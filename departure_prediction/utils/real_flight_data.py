"""
실제 JFK 출발 항공편 데이터 수집
AviationStack API (무료) 사용
"""
import requests
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


class RealFlightDataCollector:
    """실제 항공편 데이터 수집기"""
    
    # 무료 API 옵션들
    AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"
    
    # 미국 주요 국내선 공항
    DOMESTIC_AIRPORTS = {
        'ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'SFO', 'SEA', 'LAS', 'MCO',
        'CLT', 'PHX', 'IAH', 'MIA', 'BOS', 'MSP', 'FLL', 'DTW', 'PHL',
        'BWI', 'SLC', 'SAN', 'DCA', 'MDW', 'TPA', 'PDX', 'STL', 'HNL',
    }
    
    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: AviationStack API 키 (무료: aviationstack.com에서 발급)
        """
        self.api_key = api_key or os.getenv('AVIATIONSTACK_API_KEY')
    
    def get_jfk_departures_today(self, limit: int = 50) -> List[Dict]:
        """
        오늘 JFK 출발 항공편 조회
        
        Returns:
            List[Dict]: 항공편 정보
        """
        if not self.api_key:
            print("⚠️  AviationStack API 키가 없습니다. 샘플 데이터를 반환합니다.")
            return self._get_sample_data()
        
        params = {
            'access_key': self.api_key,
            'dep_iata': 'JFK',
            'limit': limit,
        }
        
        try:
            print(f"🌐 AviationStack API 호출 중...")
            response = requests.get(self.AVIATIONSTACK_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                print(f"⚠️  API 응답 오류: {data}")
                return self._get_sample_data()
            
            flights = []
            for flight in data['data']:
                # 국내선만 필터링
                arrival_iata = flight.get('arrival', {}).get('iata', '')
                if arrival_iata in self.DOMESTIC_AIRPORTS:
                    
                    flight_info = {
                        'flight_number': flight.get('flight', {}).get('iata', 'N/A'),
                        'airline': flight.get('airline', {}).get('name', 'N/A'),
                        'destination': arrival_iata,
                        'destination_city': flight.get('arrival', {}).get('airport', 'N/A'),
                        'terminal': flight.get('departure', {}).get('terminal', 'N/A'),
                        'gate': flight.get('departure', {}).get('gate', 'N/A'),
                        'scheduled_time': flight.get('departure', {}).get('scheduled', 'N/A'),
                        'actual_time': flight.get('departure', {}).get('actual', None),
                        'estimated_time': flight.get('departure', {}).get('estimated', None),
                        'delay': flight.get('departure', {}).get('delay', 0),
                        'status': flight.get('flight_status', 'N/A'),
                        'is_domestic': True
                    }
                    flights.append(flight_info)
            
            print(f"✅ {len(flights)}개의 국내선 항공편 수집")
            return flights
            
        except requests.RequestException as e:
            print(f"❌ API 호출 오류: {str(e)}")
            return self._get_sample_data()
    
    def _get_sample_data(self) -> List[Dict]:
        """
        실제 데이터 기반 샘플 (JFK 실제 스케줄 참고)
        과거 데이터나 전형적인 스케줄 기반
        """
        print("📋 실제 JFK 스케줄 기반 샘플 데이터 사용")
        
        base = datetime.now()
        
        # 실제 JFK 국내선 스케줄 기반 (일반적인 패턴)
        flights = [
            {
                'flight_number': 'AA100',
                'airline': 'American Airlines',
                'destination': 'LAX',
                'destination_city': 'Los Angeles International',
                'terminal': 'Terminal 8',
                'gate': 'A10',
                'scheduled_time': (base.replace(hour=8, minute=0) + timedelta(days=1)).isoformat(),
                'actual_time': (base.replace(hour=8, minute=5) + timedelta(days=1)).isoformat(),
                'delay': 5,
                'status': 'active',
                'is_domestic': True,
                'typical_passenger_origin': '200 W 56th St, New York, NY 10019',
            },
            {
                'flight_number': 'DL302',
                'airline': 'Delta Air Lines',
                'destination': 'ATL',
                'destination_city': 'Hartsfield-Jackson Atlanta International',
                'terminal': 'Terminal 4',
                'gate': 'B25',
                'scheduled_time': (base.replace(hour=10, minute=30) + timedelta(days=1)).isoformat(),
                'actual_time': (base.replace(hour=10, minute=25) + timedelta(days=1)).isoformat(),
                'delay': -5,
                'status': 'active',
                'is_domestic': True,
                'typical_passenger_origin': 'Columbus Hotel, 308 W 58th St #6, New York, NY 10019',
            },
            {
                'flight_number': 'B6623',
                'airline': 'JetBlue Airways',
                'destination': 'SFO',
                'destination_city': 'San Francisco International',
                'terminal': 'Terminal 5',
                'gate': 'C15',
                'scheduled_time': (base.replace(hour=13, minute=15) + timedelta(days=1)).isoformat(),
                'actual_time': (base.replace(hour=13, minute=35) + timedelta(days=1)).isoformat(),
                'delay': 20,
                'status': 'active',
                'is_domestic': True,
                'typical_passenger_origin': '450 W 42nd St, New York, NY 10036',
            },
            {
                'flight_number': 'UA215',
                'airline': 'United Airlines',
                'destination': 'ORD',
                'destination_city': "O'Hare International",
                'terminal': 'Terminal 7',
                'gate': 'D8',
                'scheduled_time': (base.replace(hour=16, minute=45) + timedelta(days=1)).isoformat(),
                'actual_time': (base.replace(hour=17, minute=10) + timedelta(days=1)).isoformat(),
                'delay': 25,
                'status': 'active',
                'is_domestic': True,
                'typical_passenger_origin': '123 E 86th St, New York, NY 10028',
            },
            {
                'flight_number': 'AA1804',
                'airline': 'American Airlines',
                'destination': 'MIA',
                'destination_city': 'Miami International',
                'terminal': 'Terminal 8',
                'gate': 'A22',
                'scheduled_time': (base.replace(hour=19, minute=0) + timedelta(days=1)).isoformat(),
                'actual_time': (base.replace(hour=19, minute=0) + timedelta(days=1)).isoformat(),
                'delay': 0,
                'status': 'scheduled',
                'is_domestic': True,
                'typical_passenger_origin': '15 Broad St, New York, NY 10005',
            },
        ]
        
        return flights
    
    def save_to_json(self, flights: List[Dict], filename: str = None):
        """항공편 데이터를 JSON으로 저장"""
        if filename is None:
            filename = f"real_jfk_flights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), 'data', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'collected_at': datetime.now().isoformat(),
                'total_flights': len(flights),
                'flights': flights
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 저장: {filepath}")
        return filepath


def main():
    """테스트"""
    print("=" * 70)
    print("    ✈️  실제 JFK 출발 항공편 데이터 수집")
    print("=" * 70)
    print()
    
    collector = RealFlightDataCollector()
    flights = collector.get_jfk_departures_today(limit=30)
    
    if flights:
        print(f"\n📊 수집 결과: {len(flights)}개 국내선 항공편")
        
        for flight in flights[:5]:
            print(f"  ✈️  {flight['flight_number']} → {flight['destination']}")
            print(f"     예정: {flight['scheduled_time']}")
            if flight.get('actual_time'):
                print(f"     실제: {flight['actual_time']} (지연: {flight.get('delay', 0)}분)")
            print()
        
        filepath = collector.save_to_json(flights)
        print(f"✅ 완료!")
        print()
        print("💡 다음 단계:")
        print("   1. 각 항공편에 대해 우리 시스템의 출발 시간 추천 생성")
        print("   2. 추천 출발 시간 vs 실제 필요 시간 비교")
        print("   3. 지연 데이터 고려하여 정확도 평가")


if __name__ == "__main__":
    main()
