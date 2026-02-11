"""
항공 티켓 이미지 생성
PIL을 사용하여 현실적인 항공권 이미지 생성
"""
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
import os

def generate_ticket_image(flight_data, output_path):
    """
    항공 티켓 이미지 생성
    
    Args:
        flight_data: 항공편 정보 dict
        output_path: 저장 경로
    """
    # 티켓 크기 (가로 x 세로)
    width, height = 800, 400
    
    # 배경색 (항공사별로 다르게 설정 가능)
    bg_color = '#FFFFFF'
    primary_color = '#1E3A8A'  # 진한 파란색
    secondary_color = '#3B82F6'  # 밝은 파란색
    text_color = '#1F2937'  # 검은색
    
    # 이미지 생성
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # 폰트 설정 (시스템 기본 폰트 사용)
    try:
        title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 32)
        large_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
        medium_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
        small_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except:
        title_font = ImageFont.load_default()
        large_font = ImageFont.load_default()
        medium_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # 상단 헤더 (항공사명)
    draw.rectangle([(0, 0), (width, 80)], fill=primary_color)
    draw.text((30, 25), flight_data['airline'].upper(), fill='white', font=title_font)
    
    # 항공편 번호 (우측 상단)
    draw.text((width - 200, 25), f"Flight {flight_data['flight_number']}", fill='white', font=large_font)
    
    # 승객 정보
    y_pos = 110
    draw.text((30, y_pos), "PASSENGER", fill=secondary_color, font=small_font)
    draw.text((30, y_pos + 25), flight_data.get('passenger', 'John Smith'), fill=text_color, font=large_font)
    
    # 출발/도착 정보 (중앙)
    y_pos = 180
    
    # 출발지
    draw.text((30, y_pos), "FROM", fill=secondary_color, font=small_font)
    draw.text((30, y_pos + 25), flight_data['origin'], fill=text_color, font=title_font)
    
    # 화살표
    draw.text((200, y_pos + 25), "→", fill=secondary_color, font=title_font)
    
    # 도착지
    draw.text((280, y_pos), "TO", fill=secondary_color, font=small_font)
    draw.text((280, y_pos + 25), flight_data['destination'], fill=text_color, font=title_font)
    
    # 날짜/시간
    scheduled_dt = datetime.strptime(flight_data['scheduled_time'], '%Y-%m-%d %H:%M')
    date_str = scheduled_dt.strftime('%B %d, %Y')
    time_str = scheduled_dt.strftime('%H:%M')
    
    y_pos = 280
    draw.text((30, y_pos), "DATE", fill=secondary_color, font=small_font)
    draw.text((30, y_pos + 25), date_str, fill=text_color, font=medium_font)
    
    draw.text((280, y_pos), "DEPARTURE TIME", fill=secondary_color, font=small_font)
    draw.text((280, y_pos + 25), time_str, fill=text_color, font=medium_font)
    
    # 터미널/게이트/좌석 (우측)
    y_pos = 110
    x_pos = width - 220
    
    draw.text((x_pos, y_pos), "TERMINAL", fill=secondary_color, font=small_font)
    draw.text((x_pos, y_pos + 25), str(flight_data.get('terminal', 'N/A')), fill=text_color, font=large_font)
    
    y_pos = 180
    draw.text((x_pos, y_pos), "GATE", fill=secondary_color, font=small_font)
    draw.text((x_pos, y_pos + 25), str(flight_data.get('gate', 'N/A')), fill=text_color, font=large_font)
    
    y_pos = 250
    draw.text((x_pos, y_pos), "SEAT", fill=secondary_color, font=small_font)
    draw.text((x_pos, y_pos + 25), flight_data.get('seat', 'N/A'), fill=text_color, font=large_font)
    
    # 하단 바코드 영역 (장식용)
    draw.rectangle([(0, height - 60), (width, height)], fill=primary_color)
    draw.text((30, height - 45), "BOARDING PASS", fill='white', font=medium_font)
    draw.text((width - 250, height - 45), f"{flight_data['origin']}-{flight_data['destination']}", fill='white', font=medium_font)
    
    # 이미지 저장
    img.save(output_path)
    print(f"✅ 티켓 생성: {output_path}")
    return output_path


if __name__ == '__main__':
    # 테스트용 티켓 데이터 로드
    with open('../data/test_tickets_today.json', 'r') as f:
        tickets = json.load(f)
    
    # 출력 디렉토리 생성
    os.makedirs('../test_tickets', exist_ok=True)
    
    # 각 티켓 이미지 생성
    print("=" * 60)
    print("항공 티켓 이미지 생성")
    print("=" * 60)
    
    for i, ticket in enumerate(tickets, 1):
        output_file = f"../test_tickets/ticket_{i}_{ticket['flight_number']}.png"
        generate_ticket_image(ticket, output_file)
    
    print(f"\n✅ 총 {len(tickets)}개 티켓 이미지 생성 완료")
    print(f"📁 저장 위치: test_tickets/")
