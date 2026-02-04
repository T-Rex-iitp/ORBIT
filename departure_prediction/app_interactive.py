"""
대화형 출발 시간 추천 시스템
사용자 입력: 티켓 이미지 or 직접 입력 + 위치 + 교통수단 + 수하물 여부
"""
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.ticket_ocr import TicketOCR
from hybrid_predictor import HybridDeparturePredictor


def print_header():
    """헤더 출력"""
    print("="*70)
    print("✈️  AI 기반 출발 시간 추천 시스템")
    print("="*70)
    print()


def get_flight_info_from_user():
    """사용자로부터 항공편 정보 입력받기"""
    print("📋 항공편 정보 입력 방식을 선택하세요:")
    print("  1. 티켓 이미지 업로드 (자동 인식)")
    print("  2. 직접 입력")
    
    while True:
        choice = input("\n선택 (1 또는 2): ").strip()
        
        if choice == '1':
            return get_flight_info_from_image()
        elif choice == '2':
            return get_flight_info_manual()
        else:
            print("❌ 잘못된 입력입니다. 1 또는 2를 입력하세요.")


def get_flight_info_from_image():
    """티켓 이미지에서 정보 추출"""
    print("\n📸 티켓 이미지 업로드")
    print("이미지 파일 경로를 입력하세요 (예: /path/to/ticket.png)")
    
    while True:
        image_path = input("이미지 경로: ").strip()
        
        if not os.path.exists(image_path):
            print(f"❌ 파일을 찾을 수 없습니다: {image_path}")
            retry = input("다시 입력하시겠습니까? (y/n): ").strip().lower()
            if retry != 'y':
                return None
            continue
        
        try:
            print("\n🔍 이미지 분석 중... (LLaVA-Phi3)")
            ocr = TicketOCR(method='vision')
            flight_data = ocr.extract_with_vision(image_path)
            
            print("\n✅ 티켓 정보 추출 완료:")
            print(f"   항공편: {flight_data.get('flight_number', 'N/A')}")
            print(f"   출발: {flight_data.get('departure_time', 'N/A')}")
            print(f"   공항: {flight_data.get('departure_airport', 'N/A')} → {flight_data.get('arrival_airport', 'N/A')}")
            print(f"   수하물: {'있음' if flight_data.get('has_checked_baggage') else '없음'}")
            print(f"   TSA PreCheck: {'있음' if flight_data.get('has_tsa_precheck') else '없음'}")
            
            # 확인
            confirm = input("\n이 정보가 맞습니까? (y/n): ").strip().lower()
            if confirm == 'y':
                return flight_data
            else:
                return None
        
        except Exception as e:
            print(f"\n❌ 이미지 분석 실패: {e}")
            retry = input("수동 입력으로 전환하시겠습니까? (y/n): ").strip().lower()
            if retry == 'y':
                return None
            continue


def get_flight_info_manual():
    """수동으로 항공편 정보 입력"""
    print("\n✍️  항공편 정보 직접 입력")
    
    # 항공편 번호
    while True:
        flight_number = input("\n항공편 번호 (예: B6123, AA100): ").strip().upper()
        if len(flight_number) >= 3:
            break
        print("❌ 올바른 항공편 번호를 입력하세요.")
    
    # 출발 공항
    print("\n출발 공항 코드 (예: JFK, LAX, ORD)")
    departure_airport = input("출발 공항: ").strip().upper()
    
    # 도착 공항
    arrival_airport = input("도착 공항: ").strip().upper()
    
    # 출발 날짜 및 시간
    while True:
        print("\n출발 날짜 및 시간 (예: 2026-02-05 19:00)")
        departure_time_str = input("출발 시간: ").strip()
        try:
            datetime.strptime(departure_time_str, '%Y-%m-%d %H:%M')
            break
        except ValueError:
            print("❌ 올바른 형식으로 입력하세요 (YYYY-MM-DD HH:MM)")
    
    # TSA PreCheck
    print("\nTSA PreCheck가 있으십니까?")
    print("  1. 있음 (보안 검색 대기시간 단축)")
    print("  2. 없음")
    has_tsa = input("선택 (1 또는 2): ").strip() == '1'
    
    return {
        'flight_number': flight_number,
        'departure_airport': departure_airport,
        'arrival_airport': arrival_airport,
        'departure_time': departure_time_str,
        'terminal': None,
        'has_checked_baggage': None,  # 나중에 입력받음
        'has_tsa_precheck': has_tsa
    }


def get_location():
    """출발 위치 입력"""
    print("\n📍 출발 위치")
    print("현재 위치 또는 주소를 입력하세요 (예: Times Square, New York, NY)")
    
    while True:
        address = input("주소: ").strip()
        if len(address) > 3:
            return address
        print("❌ 올바른 주소를 입력하세요.")


def get_travel_mode():
    """교통수단 선택"""
    print("\n🚗 교통수단 선택")
    print("  1. 자동차 (DRIVE)")
    print("  2. 대중교통 (TRANSIT)")
    print("  3. 도보 (WALK)")
    print("  4. 자전거 (BICYCLE)")
    
    modes = {
        '1': 'DRIVE',
        '2': 'TRANSIT',
        '3': 'WALK',
        '4': 'BICYCLE'
    }
    
    while True:
        choice = input("\n선택 (1-4): ").strip()
        if choice in modes:
            return modes[choice]
        print("❌ 1-4 중에서 선택하세요.")


def get_baggage_info(flight_data):
    """수하물 정보 입력 (이미지에서 추출되지 않은 경우)"""
    if flight_data.get('has_checked_baggage') is not None:
        # 이미 정보가 있으면 확인만
        print(f"\n🧳 수하물 정보: {'체크인 수하물 있음' if flight_data['has_checked_baggage'] else '기내 반입만'}")
        change = input("변경하시겠습니까? (y/n): ").strip().lower()
        if change != 'y':
            return flight_data['has_checked_baggage']
    
    print("\n🧳 수하물 체크인 여부")
    print("  1. 체크인 수하물 있음 (+30분 소요)")
    print("  2. 기내 반입만 (체크인 불필요)")
    
    while True:
        choice = input("선택 (1 또는 2): ").strip()
        if choice in ['1', '2']:
            return choice == '1'
        print("❌ 1 또는 2를 선택하세요.")


def parse_flight_data(flight_data):
    """항공편 데이터 파싱"""
    # 항공편 번호에서 항공사 코드 추출
    flight_number = flight_data['flight_number']
    airline_code = ''.join([c for c in flight_number if c.isalpha()])
    
    # 항공사 이름 매핑
    airline_names = {
        'B6': 'JetBlue Airways',
        'AA': 'American Airlines',
        'DL': 'Delta Air Lines',
        'UA': 'United Airlines',
        'WN': 'Southwest Airlines',
        'NK': 'Spirit Airlines',
        'F9': 'Frontier Airlines',
        'AS': 'Alaska Airlines',
        'B': 'JetBlue Airways',  # 단축형
        'A': 'American Airlines'
    }
    
    # 날짜 시간 파싱 (여러 형식 지원)
    time_str = flight_data['departure_time']
    scheduled_time = None
    
    # 시도할 날짜 형식들
    time_formats = [
        '%Y-%m-%d %H:%M',
        '%Y-%m-%dT%H:%M',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S'
    ]
    
    for fmt in time_formats:
        try:
            scheduled_time = datetime.strptime(time_str, fmt)
            break
        except ValueError:
            continue
    
    if scheduled_time is None:
        raise ValueError(f"날짜 형식을 인식할 수 없습니다: {time_str}")
    
    return {
        'airline_code': airline_code,
        'airline_name': airline_names.get(airline_code, airline_code),
        'flight_number': flight_number,
        'origin': flight_data['departure_airport'],
        'dest': flight_data['arrival_airport'],
        'scheduled_time': scheduled_time,
        'has_checked_baggage': flight_data['has_checked_baggage'],
        'has_tsa_precheck': flight_data.get('has_tsa_precheck', False)
    }


def main():
    """메인 실행"""
    print_header()
    
    try:
        # 1. 항공편 정보 입력
        flight_data = get_flight_info_from_user()
        
        if flight_data is None:
            # 이미지 실패 시 수동 입력
            flight_data = get_flight_info_manual()
        
        # 2. 출발 위치
        address = get_location()
        
        # 3. 교통수단
        travel_mode = get_travel_mode()
        
        # 4. 수하물 정보
        has_baggage = get_baggage_info(flight_data)
        flight_data['has_checked_baggage'] = has_baggage
        
        # 5. 정보 확인
        print("\n" + "="*70)
        print("📋 입력 정보 확인")
        print("="*70)
        print(f"항공편: {flight_data['flight_number']}")
        print(f"출발: {flight_data['departure_time']}")
        print(f"공항: {flight_data['departure_airport']} → {flight_data['arrival_airport']}")
        print(f"출발지: {address}")
        print(f"교통수단: {travel_mode}")
        print(f"수하물: {'체크인 있음' if has_baggage else '기내 반입만'}")
        print(f"TSA PreCheck: {'있음' if flight_data.get('has_tsa_precheck') else '없음'}")
        print("="*70)
        
        confirm = input("\n계속 진행하시겠습니까? (y/n): ").strip().lower()
        if confirm != 'y':
            print("\n❌ 취소되었습니다.")
            return
        
        # 6. 예측 실행
        print("\n🔍 최적 출발 시간 계산 중...\n")
        
        flight_info = parse_flight_data(flight_data)
        
        predictor = HybridDeparturePredictor('models/delay_predictor_full.pkl')
        
        result = predictor.recommend_departure(
            address=address,
            flight_info=flight_info,
            travel_mode=travel_mode
        )
        
        # 7. 결과 출력
        if result['success']:
            print("\n" + "="*70)
            print("✅ 출발 시간 추천 결과")
            print("="*70)
            print(result['recommendation'])
            print("="*70)
            
            print(f"\n📊 상세 정보:")
            details = result['details']
            print(f"   - 추천 출발 시간: {details['recommended_departure']}")
            print(f"   - 항공편 출발: {details['flight_time']}")
            print(f"   - 예상 실제 출발: {details['actual_departure']}")
            print(f"   - 이동 시간: {details['travel_time']}분")
            print(f"   - TSA 대기: {details['tsa_wait']}분")
            print(f"   - 수하물 체크인: {details['baggage_check']}분")
            print(f"   - 예상 지연: {details['predicted_delay']:.0f}분")
            print(f"   - 총 소요 시간: {details['total_time']}분")
        else:
            print(f"\n❌ 오류: {result.get('error', 'Unknown error')}")
    
    except KeyboardInterrupt:
        print("\n\n❌ 사용자가 취소했습니다.")
    except FileNotFoundError:
        print(f"\n⚠️ 학습된 모델을 찾을 수 없습니다.")
        print(f"   train_delay_predictor.ipynb를 먼저 실행하여 모델을 학습시켜주세요.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("감사합니다! 안전한 여행 되세요 ✈️")
    print("="*70)


if __name__ == '__main__':
    main()
