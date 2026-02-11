"""
Google Gemini API (Direct) Integration
Vertex AI 대신 Google AI API 사용 (API Key 기반)
"""
import os
import google.generativeai as genai
from typing import Optional, Dict, Any
from PIL import Image as PILImage


class GeminiDirectClient:
    """
    Google Gemini API 클라이언트 (Direct API)
    API Key만 있으면 바로 사용 가능 (Vertex AI 불필요)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-flash-latest"
    ):
        """
        Args:
            api_key: Google AI API Key (환경변수에서 자동 로드)
            model_name: 모델 이름
                - gemini-flash-latest: 빠르고 저렴 (추천)
                - gemini-pro-latest: 더 정확, 더 비쌈
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        # Gemini API 초기화
        genai.configure(api_key=self.api_key)
        
        # 모델 초기화
        self.model = genai.GenerativeModel(model_name)
        
        print(f"✅ Gemini Direct API initialized: {model_name}")
    
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        텍스트 생성 (LLM)
        
        Args:
            prompt: 입력 프롬프트
            temperature: 창의성 (0.0~2.0)
            max_tokens: 최대 토큰 수
            
        Returns:
            생성된 텍스트
        """
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            print(f"❌ Gemini text generation error: {e}")
            raise
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail."
    ) -> str:
        """
        이미지 분석 (Vision)
        
        Args:
            image_path: 이미지 파일 경로
            prompt: 분석 프롬프트
            
        Returns:
            분석 결과 텍스트
        """
        try:
            # 이미지 로드
            image = PILImage.open(image_path)
            
            # 이미지 + 프롬프트로 분석
            response = self.model.generate_content([prompt, image])
            
            return response.text
            
        except Exception as e:
            print(f"❌ Gemini image analysis error: {e}")
            raise
    
    def extract_ticket_info(
        self,
        image_path: str
    ) -> Dict[str, Any]:
        """
        항공권 이미지에서 정보 추출
        
        Args:
            image_path: 항공권 이미지 경로
            
        Returns:
            추출된 정보 딕셔너리
        """
        prompt = """
        Extract flight information from this airline ticket image.
        Return ONLY a JSON object with these fields:
        {
            "flight_number": "string (e.g., B6123)",
            "airline": "string (e.g., JetBlue)",
            "departure_airport": "string (IATA code, e.g., JFK)",
            "arrival_airport": "string (IATA code, e.g., LAX)",
            "departure_time": "string (YYYY-MM-DD HH:MM format)",
            "arrival_time": "string (YYYY-MM-DD HH:MM format)",
            "passenger_name": "string",
            "seat": "string (e.g., 12A)",
            "gate": "string (e.g., B42)",
            "terminal": "string (e.g., 5)"
        }
        
        If any field is not found, use null.
        Return ONLY the JSON, no additional text.
        """
        
        try:
            result_text = self.analyze_image(image_path, prompt)
            
            # JSON 추출 (```json ... ``` 제거)
            import json
            import re
            
            # JSON 블록 찾기
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 블록 없이 바로 JSON
                json_str = result_text
            
            # JSON 파싱
            ticket_info = json.loads(json_str.strip())
            
            print(f"✅ Ticket info extracted: {ticket_info.get('flight_number', 'N/A')}")
            return ticket_info
            
        except Exception as e:
            print(f"❌ Ticket extraction error: {e}")
            return {}
    
    def generate_departure_recommendation(
        self,
        flight_info: Dict[str, Any],
        travel_time_minutes: int,
        tsa_wait_minutes: int,
        weather_condition: str,
        predicted_delay_minutes: int
    ) -> str:
        """
        출발 시간 추천 생성
        
        Args:
            flight_info: 항공편 정보
            travel_time_minutes: 공항까지 이동 시간 (분)
            tsa_wait_minutes: TSA 대기 시간 (분)
            weather_condition: 날씨 상태
            predicted_delay_minutes: 예상 지연 시간 (분)
            
        Returns:
            자연어 추천 텍스트
        """
        prompt = f"""
        You are a helpful travel assistant. Generate a departure recommendation for a passenger.
        
        Flight Information:
        - Flight: {flight_info.get('flight_number', 'N/A')} ({flight_info.get('airline', 'N/A')})
        - Route: {flight_info.get('departure_airport', 'N/A')} → {flight_info.get('arrival_airport', 'N/A')}
        - Scheduled Departure: {flight_info.get('departure_time', 'N/A')}
        - Gate: {flight_info.get('gate', 'N/A')}
        - Terminal: {flight_info.get('terminal', 'N/A')}
        
        Travel Conditions:
        - Travel time to airport: {travel_time_minutes} minutes
        - TSA security wait: {tsa_wait_minutes} minutes
        - Weather: {weather_condition}
        - Predicted delay: {predicted_delay_minutes} minutes
        
        Generate a friendly, concise recommendation including:
        1. Recommended departure time from home
        2. Arrival time at airport
        3. Any weather or delay warnings
        4. Tips for smooth travel
        
        Keep it under 150 words.
        """
        
        try:
            recommendation = self.generate_text(prompt, temperature=0.7)
            return recommendation
            
        except Exception as e:
            print(f"❌ Recommendation generation error: {e}")
            return "Unable to generate recommendation at this time."


class GeminiTicketOCR:
    """
    TicketOCR 호환 클래스 (drop-in replacement)
    기존 코드에서 바로 사용 가능
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = GeminiDirectClient(api_key=api_key)
    
    def extract_with_vision(self, image_path: str) -> Dict[str, Any]:
        """TicketOCR.extract_with_vision() 호환 메서드"""
        return self.client.extract_ticket_info(image_path)


# 테스트 코드
if __name__ == "__main__":
    import sys
    
    # API Key 확인
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        print("Set it with: export GEMINI_API_KEY=your-api-key")
        sys.exit(1)
    
    # 클라이언트 초기화
    client = GeminiDirectClient(api_key=api_key)
    
    # 텍스트 생성 테스트
    print("\n=== Text Generation Test ===")
    response = client.generate_text("Say hello in 5 words")
    print(f"Response: {response}")
    
    print("\n✅ Gemini Direct API test passed!")
