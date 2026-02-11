"""
Google Routes API 통합 모듈
출발지 주소에서 JFK 공항까지의 경로 정보 및 소요 시간 조회
"""
import os
import requests
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


class GoogleRoutesAPI:
    """Google Routes API 클라이언트"""
    
    # JFK 공항 터미널 좌표
    JFK_TERMINALS = {
        'Terminal 1': {'lat': 40.6441, 'lng': -73.7892},
        'Terminal 4': {'lat': 40.6441, 'lng': -73.7769},
        'Terminal 5': {'lat': 40.6399, 'lng': -73.7789},
        'Terminal 7': {'lat': 40.6505, 'lng': -73.7918},
        'Terminal 8': {'lat': 40.6472, 'lng': -73.7889},
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Google API Key (없으면 환경변수에서 읽음)
        """
        self.api_key = api_key or os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google API Key가 필요합니다. "
                "환경변수 GOOGLE_MAPS_API_KEY를 설정하거나 api_key 인자를 제공하세요."
            )
        
        self.base_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    
    def get_route_info(
        self,
        origin_address: str,
        terminal: str = 'Terminal 4',
        departure_time: Optional[datetime] = None,
        traffic_model: str = 'best_guess',
        travel_mode: str = 'DRIVE'
    ) -> Dict:
        """
        출발지에서 JFK 공항까지의 경로 정보 조회
        
        Args:
            origin_address: 출발지 주소 (예: "200 W 56th St, New York, NY 10019")
            terminal: 도착 터미널 (기본값: Terminal 4)
            departure_time: 출발 시간 (기본값: 현재 시간)
            traffic_model: 교통 예측 모델 ('best_guess', 'pessimistic', 'optimistic')
            travel_mode: 이동 수단 ('DRIVE', 'TRANSIT', 'WALK', 'BICYCLE', 'TWO_WHEELER')
            
        Returns:
            Dict: {
                'duration_in_traffic': int,  # 교통 정보 포함 소요 시간 (초)
                'duration': int,             # 일반 소요 시간 (초)
                'distance': int,             # 거리 (미터)
                'route_summary': str,        # 경로 요약
                'departure_time': str,       # 출발 시간
                'arrival_time': str,         # 예상 도착 시간
                'traffic_condition': str,    # 교통 상황
            }
        """
        if terminal not in self.JFK_TERMINALS:
            raise ValueError(f"유효하지 않은 터미널: {terminal}. 사용 가능: {list(self.JFK_TERMINALS.keys())}")
        
        if departure_time is None:
            departure_time = datetime.now()
        
        destination = self.JFK_TERMINALS[terminal]
        
        # API 요청 헤더
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.api_key,
            'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.polyline,routes.legs.steps'
        }
        
        # API 요청 본문
        payload = {
            "origin": {
                "address": origin_address
            },
            "destination": {
                "location": {
                    "latLng": {
                        "latitude": destination['lat'],
                        "longitude": destination['lng']
                    }
                }
            },
            "travelMode": travel_mode,
            "computeAlternativeRoutes": False,
            "languageCode": "en-US",
            "units": "METRIC"
        }
        
        # DRIVE 모드일 때만 routingPreference 추가
        if travel_mode == "DRIVE":
            payload["routingPreference"] = "TRAFFIC_AWARE"
            payload["departureTime"] = departure_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            payload["routeModifiers"] = {
                "avoidTolls": False,
                "avoidHighways": False,
                "avoidFerries": False
            }
        
        # TRANSIT 모드일 때만 transitPreferences 추가
        if travel_mode == "TRANSIT":
            payload["departureTime"] = departure_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            payload["transitPreferences"] = {
                "allowedTravelModes": ["BUS", "SUBWAY", "TRAIN", "RAIL"],
                "routingPreference": "LESS_WALKING"
            }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            if 'routes' not in data or len(data['routes']) == 0:
                raise Exception("경로를 찾을 수 없습니다.")
            
            route = data['routes'][0]
            duration_seconds = int(route['duration'].rstrip('s'))
            distance_meters = route['distanceMeters']
            
            # Transit 세부 경로 정보 추출
            transit_details = []
            if travel_mode == 'TRANSIT' and 'legs' in route and len(route['legs']) > 0:
                for leg in route['legs']:
                    if 'steps' in leg:
                        for step in leg['steps']:
                            if 'transitDetails' in step:
                                transit = step['transitDetails']
                                transit_line = transit.get('transitLine', {})
                                
                                # 노선 정보
                                line_name = transit_line.get('nameShort', transit_line.get('name', 'Unknown'))
                                vehicle_type = transit_line.get('vehicle', {}).get('type', 'BUS')
                                
                                # 정류장 정보
                                depart_stop = transit.get('stopDetails', {}).get('departureStop', {}).get('name', '')
                                arrival_stop = transit.get('stopDetails', {}).get('arrivalStop', {}).get('name', '')
                                
                                # 정거장 수
                                stop_count = transit.get('stopCount', 0)
                                
                                transit_details.append({
                                    'line': line_name,
                                    'vehicle_type': vehicle_type,
                                    'from': depart_stop,
                                    'to': arrival_stop,
                                    'stops': stop_count
                                })
            
            # 도착 시간 계산
            arrival_time = departure_time + timedelta(seconds=duration_seconds)
            
            # 교통 상황 판단 (기본 시간 대비)
            base_duration = distance_meters / 13.41  # 평균 속도 30mph = 13.41 m/s
            traffic_ratio = duration_seconds / base_duration
            
            if traffic_ratio < 1.2:
                traffic_condition = "원활"
            elif traffic_ratio < 1.5:
                traffic_condition = "보통"
            elif traffic_ratio < 2.0:
                traffic_condition = "혼잡"
            else:
                traffic_condition = "매우 혼잡"
            
            result = {
                'duration_in_traffic': duration_seconds,
                'duration': duration_seconds,
                'distance': distance_meters,
                'distance_miles': distance_meters * 0.000621371,
                'route_summary': f"{origin_address} → JFK {terminal}",
                'departure_time': departure_time.isoformat(),
                'arrival_time': arrival_time.isoformat(),
                'traffic_condition': traffic_condition,
                'origin': origin_address,
                'destination': terminal,
                'transit_details': transit_details if transit_details else None,
                'travel_mode': travel_mode
            }
            
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API 요청 실패: {str(e)}"
            if hasattr(e.response, 'text'):
                error_msg += f"\nResponse: {e.response.text}"
            raise Exception(error_msg)
    
    def get_multiple_departure_times(
        self,
        origin_address: str,
        terminal: str = 'Terminal 4',
        flight_time: datetime = None,
        buffer_hours: int = 3
    ) -> List[Dict]:
        """
        여러 출발 시간 옵션에 대한 경로 정보 조회
        
        Args:
            origin_address: 출발지 주소
            terminal: 도착 터미널
            flight_time: 비행 시간
            buffer_hours: 공항 도착 여유 시간 (시간)
            
        Returns:
            List[Dict]: 출발 시간별 경로 정보 리스트
        """
        if flight_time is None:
            flight_time = datetime.now() + timedelta(hours=6)
        
        # 공항 도착 목표 시간
        target_arrival = flight_time - timedelta(hours=buffer_hours)
        
        results = []
        
        # 여러 출발 시간 시도 (목표 시간 기준 ±2시간)
        for offset_minutes in [-120, -60, 0, 60]:
            test_departure = target_arrival + timedelta(minutes=offset_minutes)
            
            # 과거 시간은 건너뛰기
            if test_departure < datetime.now():
                continue
            
            try:
                route_info = self.get_route_info(
                    origin_address=origin_address,
                    terminal=terminal,
                    departure_time=test_departure
                )
                
                # 추천 점수 계산 (목표 도착 시간에 가까울수록 높음)
                arrival_time = datetime.fromisoformat(route_info['arrival_time'])
                time_diff_minutes = abs((arrival_time - target_arrival).total_seconds() / 60)
                score = max(0, 100 - time_diff_minutes)
                
                route_info['recommendation_score'] = score
                route_info['target_arrival'] = target_arrival.isoformat()
                
                results.append(route_info)
                
            except Exception as e:
                print(f"⚠️  {test_departure.strftime('%H:%M')} 출발 정보 조회 실패: {str(e)}")
        
        # 점수순으로 정렬
        results.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        return results


def format_duration(seconds: int) -> str:
    """초를 읽기 쉬운 형식으로 변환"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    
    if hours > 0:
        return f"{hours}시간 {minutes}분"
    else:
        return f"{minutes}분"


def calculate_travel_time(origin: str, destination: str, travel_mode: str = 'DRIVE', departure_time: Optional[datetime] = None) -> Dict:
    """
    간편 함수: 주소 → 공항 이동 시간 계산
    
    Args:
        origin: 출발 주소
        destination: 공항 코드 (예: 'JFK')
        travel_mode: 이동 수단 ('DRIVE', 'TRANSIT', 'WALK', 'BICYCLE')
        departure_time: 출발 시간 (없으면 현재 시간)
    
    Returns:
        {
            'success': bool,
            'duration_minutes': int,
            'distance_miles': float,
            'traffic_condition': str,
            'transit_details': list (TRANSIT일 때만),
            'travel_mode': str,
            'error': str (실패 시)
        }
    """
    try:
        routes_api = GoogleRoutesAPI()
        
        # 공항 코드를 터미널로 매핑 (기본 Terminal 4)
        terminal_map = {
            'JFK': 'Terminal 4',
            'LAX': 'LAX',
            'ORD': 'ORD'
        }
        terminal = terminal_map.get(destination, 'Terminal 4')
        
        route_info = routes_api.get_route_info(
            origin_address=origin,
            terminal=terminal,
            travel_mode=travel_mode,
            departure_time=departure_time
        )
        
        return {
            'success': True,
            'duration_minutes': route_info['duration_in_traffic'] // 60,
            'distance_miles': route_info['distance_miles'],
            'traffic_condition': route_info['traffic_condition'],
            'transit_details': route_info.get('transit_details'),
            'travel_mode': route_info['travel_mode']
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def main():
    """테스트 실행"""
    print("=== Google Routes API 테스트 ===\n")
    
    # API 키 확인
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        print("⚠️  GOOGLE_MAPS_API_KEY 환경변수를 설정하세요.")
        print("\n설정 방법:")
        print("1. Google Cloud Console에서 API 키 생성")
        print("2. Routes API 활성화")
        print("3. .env 파일에 GOOGLE_MAPS_API_KEY=your_key 추가")
        return
    
    try:
        routes_api = GoogleRoutesAPI()
        
        # 테스트 주소
        test_address = "200 W 56th St, New York, NY 10019"
        print(f"📍 출발지: {test_address}")
        print(f"📍 목적지: JFK Airport Terminal 4\n")
        
        # 경로 정보 조회
        print("🚗 경로 정보 조회 중...\n")
        route_info = routes_api.get_route_info(
            origin_address=test_address,
            terminal='Terminal 4'
        )
        
        # 결과 출력
        print("✓ 조회 완료!\n")
        print(f"거리: {route_info['distance_miles']:.1f} 마일 ({route_info['distance']:,} 미터)")
        print(f"소요 시간: {format_duration(route_info['duration_in_traffic'])}")
        print(f"교통 상황: {route_info['traffic_condition']}")
        print(f"출발 시간: {datetime.fromisoformat(route_info['departure_time']).strftime('%Y-%m-%d %H:%M')}")
        print(f"도착 예정: {datetime.fromisoformat(route_info['arrival_time']).strftime('%Y-%m-%d %H:%M')}")
        
        # 여러 출발 시간 옵션
        print("\n" + "="*50)
        print("🕐 최적 출발 시간 추천\n")
        
        flight_time = datetime.now() + timedelta(hours=6)
        print(f"비행 시간: {flight_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"공항 도착 목표: 비행 3시간 전\n")
        
        options = routes_api.get_multiple_departure_times(
            origin_address=test_address,
            terminal='Terminal 4',
            flight_time=flight_time,
            buffer_hours=3
        )
        
        for i, option in enumerate(options, 1):
            dep_time = datetime.fromisoformat(option['departure_time'])
            arr_time = datetime.fromisoformat(option['arrival_time'])
            
            print(f"{i}. 출발: {dep_time.strftime('%H:%M')} → "
                  f"도착: {arr_time.strftime('%H:%M')} "
                  f"({format_duration(option['duration_in_traffic'])}, "
                  f"{option['traffic_condition']}) "
                  f"[점수: {option['recommendation_score']:.0f}]")
        
        print("\n✅ 테스트 완료!")
        
    except ValueError as e:
        print(f"❌ 설정 오류: {str(e)}")
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
