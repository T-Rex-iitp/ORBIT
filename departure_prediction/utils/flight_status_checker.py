"""
실시간 항공편 상태 및 지연 정보 확인
AviationStack API 사용
"""
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


class FlightStatusChecker:
    """실시간 항공편 상태 확인"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: AviationStack API 키
        """
        self.api_key = api_key or os.getenv('AVIATIONSTACK_API_KEY')
        self.base_url = "http://api.aviationstack.com/v1/flights"
    
    def check_flight_status(self, flight_number: str, date: Optional[datetime] = None) -> Dict:
        """
        항공편 번호로 실시간 상태 확인
        
        Args:
            flight_number: 항공편 번호 (예: 'AA100', 'DL302')
            date: 출발 날짜 (기본값: 오늘)
            
        Returns:
            {
                'flight_number': str,
                'airline': str,
                'status': str,  # scheduled, active, landed, cancelled, delayed
                'scheduled_departure': datetime,
                'estimated_departure': datetime,
                'actual_departure': datetime,
                'delay_minutes': int,
                'is_delayed': bool,
                'delay_reason': str,  # 있을 경우
                'gate': str,
                'terminal': str
            }
        """
        if not self.api_key:
            print("⚠️ API 키가 없습니다. 샘플 데이터를 반환합니다.")
            return self._get_sample_status(flight_number)
        
        if date is None:
            date = datetime.now()
        
        params = {
            'access_key': self.api_key,
            'flight_iata': flight_number.upper()
        }
        
        try:
            print(f"🔍 {flight_number} 항공편 상태 확인 중...")
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data or len(data['data']) == 0:
                print(f"⚠️ {flight_number} 항공편을 찾을 수 없습니다.")
                return self._get_sample_status(flight_number)
            
            # 가장 최근 항공편 정보
            flight = data['data'][0]
            
            scheduled_str = flight.get('departure', {}).get('scheduled')
            estimated_str = flight.get('departure', {}).get('estimated')
            actual_str = flight.get('departure', {}).get('actual')
            
            scheduled_time = datetime.fromisoformat(scheduled_str.replace('Z', '+00:00')) if scheduled_str else None
            estimated_time = datetime.fromisoformat(estimated_str.replace('Z', '+00:00')) if estimated_str else scheduled_time
            actual_time = datetime.fromisoformat(actual_str.replace('Z', '+00:00')) if actual_str else None
            
            # 지연 시간 계산
            delay_minutes = flight.get('departure', {}).get('delay', 0) or 0
            
            # 실제로 지연되었는지 확인
            is_delayed = False
            if estimated_time and scheduled_time:
                delay_minutes = int((estimated_time - scheduled_time).total_seconds() / 60)
                is_delayed = delay_minutes > 15  # 15분 이상이면 지연으로 간주
            
            status = flight.get('flight_status', 'unknown')
            
            # 상태 한글 번역
            status_kr = {
                'scheduled': '정상 예정',
                'active': '운항 중',
                'landed': '도착 완료',
                'cancelled': '결항',
                'delayed': '지연',
                'diverted': '회항',
                'unknown': '정보 없음'
            }.get(status, status)
            
            result = {
                'flight_number': flight.get('flight', {}).get('iata', flight_number),
                'airline': flight.get('airline', {}).get('name', 'Unknown'),
                'status': status,
                'status_kr': status_kr,
                'scheduled_departure': scheduled_time,
                'estimated_departure': estimated_time,
                'actual_departure': actual_time,
                'delay_minutes': delay_minutes,
                'is_delayed': is_delayed,
                'gate': flight.get('departure', {}).get('gate', 'TBA'),
                'terminal': flight.get('departure', {}).get('terminal', 'TBA'),
                'origin': flight.get('departure', {}).get('iata', 'N/A'),
                'destination': flight.get('arrival', {}).get('iata', 'N/A'),
            }
            
            # 콘솔 출력
            print(f"✅ {result['flight_number']} - {result['airline']}")
            print(f"   상태: {result['status_kr']}")
            print(f"   출발: {result['origin']} → {result['destination']}")
            print(f"   예정: {scheduled_time.strftime('%Y-%m-%d %H:%M') if scheduled_time else 'N/A'}")
            if is_delayed:
                print(f"   ⚠️ 지연: {delay_minutes}분")
                print(f"   예상 출발: {estimated_time.strftime('%Y-%m-%d %H:%M') if estimated_time else 'N/A'}")
            print(f"   게이트: {result['terminal']} - {result['gate']}")
            
            return result
            
        except Exception as e:
            print(f"❌ API 호출 실패: {e}")
            return self._get_sample_status(flight_number)
    
    def _get_sample_status(self, flight_number: str) -> Dict:
        """샘플 상태 데이터"""
        now = datetime.now()
        scheduled = now + timedelta(hours=3)
        
        return {
            'flight_number': flight_number,
            'airline': 'Sample Airlines',
            'status': 'scheduled',
            'status_kr': '정상 예정',
            'scheduled_departure': scheduled,
            'estimated_departure': scheduled,
            'actual_departure': None,
            'delay_minutes': 0,
            'is_delayed': False,
            'gate': 'TBA',
            'terminal': 'TBA',
            'origin': 'JFK',
            'destination': 'LAX'
        }


def check_flight(flight_number: str, api_key: Optional[str] = None) -> Dict:
    """
    간편 함수: 항공편 상태 확인
    
    Args:
        flight_number: 항공편 번호
        api_key: API 키 (옵션)
        
    Returns:
        항공편 상태 정보
    """
    checker = FlightStatusChecker(api_key)
    return checker.check_flight_status(flight_number)


if __name__ == '__main__':
    # 테스트
    print("=" * 60)
    print("실시간 항공편 상태 확인 테스트")
    print("=" * 60)
    
    # 수집된 데이터에서 샘플 항공편 번호 사용
    test_flights = ['AA100', 'DL302', 'B6623']
    
    for flight_num in test_flights:
        print(f"\n{'='*60}")
        status = check_flight(flight_num)
        print()
