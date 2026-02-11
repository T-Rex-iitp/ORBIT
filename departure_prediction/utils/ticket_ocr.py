"""
항공권 티켓 OCR - 이미지에서 비행 정보 추출
Tesseract, EasyOCR, 또는 LLM Vision을 사용하여 티켓 정보 인식
"""
import os
import re
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import base64
    import requests
    HAS_VISION = True
except ImportError:
    HAS_VISION = False


class TicketOCR:
    """항공권 티켓 정보 추출"""
    
    def __init__(self, method: str = 'auto'):
        """
        Args:
            method: 'tesseract', 'easyocr', 'vision', 'auto'
        """
        self.method = method
        
        if method == 'auto':
            # 사용 가능한 첫 번째 방법 선택 (Ollama LLaVA 우선)
            if HAS_VISION:
                self.method = 'vision'
            elif HAS_EASYOCR:
                self.method = 'easyocr'
            elif HAS_TESSERACT:
                self.method = 'tesseract'
            else:
                raise ImportError(
                    "OCR 라이브러리가 설치되지 않았습니다.\n"
                    "다음 중 하나를 설치하세요:\n"
                    "  pip install pytesseract  # Tesseract\n"
                    "  pip install easyocr      # EasyOCR\n"
                    "  pip install requests     # Ollama LLaVA (권장)"
                )
    
    def extract_with_tesseract(self, image_path: str) -> str:
        """Tesseract OCR로 텍스트 추출"""
        if not HAS_TESSERACT or not HAS_PIL:
            raise ImportError("pytesseract와 Pillow가 필요합니다")
        
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    
    def extract_with_easyocr(self, image_path: str) -> str:
        """EasyOCR로 텍스트 추출"""
        if not HAS_EASYOCR:
            raise ImportError("easyocr가 필요합니다")
        
        reader = easyocr.Reader(['en', 'ko'])
        results = reader.readtext(image_path)
        text = '\n'.join([result[1] for result in results])
        return text
    
    def extract_with_vision(self, image_path: str) -> Dict:
        """Ollama LLaVA로 정보 추출"""
        if not HAS_VISION:
            raise ImportError("requests 패키지가 필요합니다")
        
        # 이미지를 base64로 인코딩
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Ollama API 호출
        ollama_url = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
        
        prompt = """Carefully read this airline ticket image and extract the information in JSON format ONLY.

CRITICAL: Read the flight number EXACTLY as shown on the ticket!
- Flight number = 2-letter airline code + digits (e.g., QR2867, AA100, DL302, B6623)
- Look for labels: "Flight:", "Flight No:", "항공편:", or large bold text with airline code
- Common airlines: QR (Qatar), AA (American), DL (Delta), UA (United), B6 (JetBlue), SQ (Singapore)
- Double-check: Is it "QR" or "SQ"? Is it "2867" or "1481"? Read CAREFULLY!

Return ONLY this JSON format (no extra text):
{
  "departure_time": "YYYY-MM-DD HH:MM",
  "flight_number": "EXACT flight number from image (e.g., QR2867)",
  "departure_airport": "3-letter code (e.g., JFK)",
  "arrival_airport": "3-letter code (e.g., DOH)",
  "terminal": "terminal number",
  "passenger_name": null,
  "has_checked_baggage": true/false,
  "baggage_count": 0,
  "has_tsa_precheck": false
}

Reading tips:
- Flight number is usually in LARGE BOLD text or near "Flight:" label
- Departure airport: look for "From:", "출발:", or origin code before arrow (→)
- Arrival airport: look for "To:", "도착:", or destination code after arrow (→)
- Date: Convert to YYYY-MM-DD format
- Time: Use 24-hour format HH:MM
- Terminal: Extract just the number
- Baggage: true if you see "Checked Baggage", "Bags", "수하물"
- TSA: true if you see "TSA PreCheck", "TSA ✓"

Read EXACTLY what you see. Do NOT guess or infer."""
        
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "llava-phi3:latest",
                "prompt": prompt,
                "images": [image_data],
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API 오류: {response.status_code} - {response.text}")
        
        # JSON 파싱
        import json
        result = response.json()
        content = result.get('response', '')
        
        # JSON 블록 추출 (```json ... ``` 제거)
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        return json.loads(content)
    
    def parse_text_for_flight_info(self, text: str) -> Dict:
        """텍스트에서 비행 정보 파싱"""
        info = {
            'departure_time': None,
            'flight_number': None,
            'departure_airport': None,
            'arrival_airport': None,
            'terminal': None,
            'passenger_name': None
        }
        
        # 날짜/시간 패턴
        date_patterns = [
            r'(\d{4}[-/]\d{2}[-/]\d{2})\s+(\d{1,2}:\d{2})',  # 2026-02-05 19:00
            r'(\d{2}[-/]\d{2}[-/]\d{4})\s+(\d{1,2}:\d{2})',  # 02-05-2026 19:00
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\s+(\d{1,2}:\d{2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if len(match.groups()) == 2:
                        date_str = f"{match.group(1)} {match.group(2)}"
                        info['departure_time'] = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
                    elif len(match.groups()) == 4:
                        # 월 이름 형식
                        month_map = {
                            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
                            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
                            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
                        }
                        month = month_map[match.group(1)]
                        day = match.group(2).zfill(2)
                        year = match.group(3)
                        time = match.group(4)
                        date_str = f"{year}-{month}-{day} {time}"
                        info['departure_time'] = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
                    break
                except:
                    continue
        
        # 항공편 번호 패턴 (예: AA123, KE001, DL2345)
        flight_pattern = r'\b([A-Z]{2}\d{3,4})\b'
        match = re.search(flight_pattern, text)
        if match:
            info['flight_number'] = match.group(1)
        
        # 터미널 패턴
        terminal_pattern = r'Terminal\s+(\d+|[A-Z])'
        match = re.search(terminal_pattern, text, re.IGNORECASE)
        if match:
            info['terminal'] = f"Terminal {match.group(1)}"
        
        # 공항 코드 패턴 (3글자 대문자)
        airport_pattern = r'\b([A-Z]{3})\b'
        airports = re.findall(airport_pattern, text)
        if len(airports) >= 2:
            info['departure_airport'] = airports[0]
            info['arrival_airport'] = airports[1]
        
        return info
    
    def extract(self, image_path: str) -> Dict:
        """이미지에서 비행 정보 추출"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        print(f"🔍 이미지 분석 중... (방법: {self.method})")
        
        if self.method == 'vision':
            # Vision API는 직접 구조화된 데이터 반환
            return self.extract_with_vision(image_path)
        
        elif self.method == 'tesseract':
            text = self.extract_with_tesseract(image_path)
            return self.parse_text_for_flight_info(text)
        
        elif self.method == 'easyocr':
            text = self.extract_with_easyocr(image_path)
            return self.parse_text_for_flight_info(text)
        
        else:
            raise ValueError(f"지원하지 않는 OCR 방법: {self.method}")


def extract_flight_info(image_path: str, method: str = 'auto') -> Dict:
    """
    항공권 이미지에서 비행 정보 추출 (간편 함수)
    
    Args:
        image_path: 이미지 파일 경로
        method: OCR 방법 ('auto', 'tesseract', 'easyocr', 'vision')
        
    Returns:
        Dict: 추출된 비행 정보
    """
    ocr = TicketOCR(method=method)
    return ocr.extract(image_path)


def main():
    """테스트 실행"""
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python ticket_ocr.py <이미지_경로>")
        print("예시: python ticket_ocr.py ticket.jpg")
        return
    
    image_path = sys.argv[1]
    
    print("=" * 70)
    print("    🎫 항공권 티켓 OCR")
    print("=" * 70)
    print()
    
    try:
        info = extract_flight_info(image_path)
        
        print("✅ 추출 완료!\n")
        print("📋 인식된 정보:")
        print(f"   • 출발 시간: {info.get('departure_time')}")
        print(f"   • 항공편: {info.get('flight_number')}")
        print(f"   • 출발 공항: {info.get('departure_airport')}")
        print(f"   • 도착 공항: {info.get('arrival_airport')}")
        print(f"   • 터미널: {info.get('terminal')}")
        print(f"   • 승객: {info.get('passenger_name')}")
        print()
        
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
