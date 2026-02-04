"""
Google Weather API를 사용하여 공항 날씨 정보를 가져오는 모듈
"""
import os
import requests
from typing import Dict, Optional
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass


# 주요 공항 좌표
AIRPORT_COORDINATES = {
    'JFK': {'lat': 40.6413, 'lon': -73.7781, 'name': 'JFK Airport, New York'},
    'LAX': {'lat': 33.9416, 'lon': -118.4085, 'name': 'LAX Airport, Los Angeles'},
    'ORD': {'lat': 41.9742, 'lon': -87.9073, 'name': "O'Hare Airport, Chicago"},
    'ATL': {'lat': 33.6407, 'lon': -84.4277, 'name': 'ATL Airport, Atlanta'},
    'DFW': {'lat': 32.8998, 'lon': -97.0403, 'name': 'DFW Airport, Dallas'},
    'SFO': {'lat': 37.6213, 'lon': -122.3790, 'name': 'SFO Airport, San Francisco'},
    'MIA': {'lat': 25.7959, 'lon': -80.2870, 'name': 'Miami International Airport'},
    'MCO': {'lat': 28.4312, 'lon': -81.3081, 'name': 'Orlando International Airport'},
    'ORLANDO': {'lat': 28.4312, 'lon': -81.3081, 'name': 'Orlando International Airport'},
}


class GoogleWeatherAPI:
    """Google Weather API를 사용한 날씨 정보 조회"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Google Maps API 키 (Routes API와 동일)
        """
        self.api_key = api_key or os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google Maps API Key가 필요합니다. "
                "환경변수 GOOGLE_MAPS_API_KEY를 설정하거나 api_key 인자를 제공하세요."
            )
        self.base_url = "https://weather.googleapis.com/v1/currentConditions:lookup"
    
    def get_airport_weather(self, airport_code: str, flight_time: datetime) -> Dict:
        """
        공항 날씨 정보 조회
        
        Args:
            airport_code: 공항 코드 (예: 'JFK')
            flight_time: 비행 시간
            
        Returns:
            날씨 정보 딕셔너리
        """
        # 공항 좌표 가져오기
        coords = AIRPORT_COORDINATES.get(airport_code.upper())
        if not coords:
            print(f"⚠️ 공항 코드 {airport_code}를 찾을 수 없습니다.")
            return self._get_default_weather(airport_code)
        
        # 출발 시간까지 남은 시간 계산
        now = datetime.now()
        hours_until_flight = (flight_time - now).total_seconds() / 3600
        
        # 현재 날씨 조회 (Google Weather API는 현재 날씨만 제공)
        weather_data = self._get_current_weather(coords['lat'], coords['lon'])
        
        # 시간이 많이 남았으면 경고
        weather_note = ""
        if hours_until_flight > 6:
            weather_note = f"(출발까지 {hours_until_flight:.0f}시간 남음 - 출발 전 재확인 권장)"
        elif hours_until_flight < 0:
            weather_note = "(이미 지난 시간)"
        
        # 지연 위험도 평가
        delay_risk = self._assess_delay_risk(weather_data)
        warning = self._get_weather_warning(weather_data)
        
        if weather_note and warning:
            warning = f"{warning} {weather_note}"
        elif weather_note:
            warning = weather_note
        
        return {
            'airport': coords['name'],
            'condition': weather_data['condition'],
            'description': weather_data['description'],
            'temperature': weather_data['temperature'],
            'wind_speed': weather_data['wind_speed'],
            'visibility': weather_data['visibility'],
            'delay_risk': delay_risk,  # 'low', 'medium', 'high'
            'warning': warning,
            'hours_until_flight': hours_until_flight
        }
    
    def _get_current_weather(self, lat: float, lon: float) -> Dict:
        """
        좌표로 현재 날씨 조회 (Google Weather API)
        
        Args:
            lat: 위도
            lon: 경도
            
        Returns:
            날씨 정보 딕셔너리
        """
        try:
            params = {
                'key': self.api_key,
                'location.latitude': lat,
                'location.longitude': lon
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Google Weather API 응답 구조
                current = data.get('currentConditions', {})
                
                # 날씨 코드를 condition으로 변환
                weather_code = current.get('weatherCode', 'CLEAR')
                condition = self._map_weather_code(weather_code)
                
                # Temperature는 Celsius로 제공됨
                temp_celsius = current.get('temperature', {}).get('value', 15)
                
                # Wind speed: m/s
                wind_speed_mps = current.get('windSpeed', {}).get('value', 0)
                
                # Visibility: meters
                visibility_m = current.get('visibility', {}).get('value', 10000)
                
                return {
                    'condition': condition,
                    'description': weather_code.lower().replace('_', ' '),
                    'temperature': temp_celsius,
                    'feels_like': current.get('temperatureApparent', {}).get('value', temp_celsius),
                    'humidity': current.get('relativeHumidity', 50),
                    'pressure': current.get('pressureSeaLevel', {}).get('value', 1013),
                    'wind_speed': wind_speed_mps,
                    'wind_deg': current.get('windDirection', 0),
                    'visibility': visibility_m,
                    'clouds': current.get('cloudCover', 0),
                    'timestamp': datetime.now()
                }
            else:
                print(f"⚠️ Google Weather API 오류: {response.status_code} - {response.text}")
                return self._get_default_weather_data()
                
        except Exception as e:
            print(f"⚠️ 날씨 조회 실패: {e}")
            return self._get_default_weather_data()
    
    def _map_weather_code(self, code: str) -> str:
        """Google Weather Code를 간단한 condition으로 변환"""
        code_map = {
            'THUNDERSTORM': 'Thunderstorm',
            'DRIZZLE': 'Rain',
            'RAIN': 'Rain',
            'SNOW': 'Snow',
            'SLEET': 'Snow',
            'FOG': 'Fog',
            'MIST': 'Mist',
            'HAZE': 'Haze',
            'CLEAR': 'Clear',
            'CLOUDY': 'Clouds',
            'PARTLY_CLOUDY': 'Clouds',
            'MOSTLY_CLOUDY': 'Clouds'
        }
        return code_map.get(code, 'Clear')
    
    def _assess_delay_risk(self, weather: Dict) -> str:
        """
        날씨 조건으로 지연 위험도 평가
        
        Returns:
            'low', 'medium', 'high'
        """
        condition = weather['condition']
        wind_speed = weather['wind_speed']  # m/s
        visibility = weather['visibility']   # meters
        
        # High risk: 심각한 악천후
        if condition in ['Thunderstorm', 'Snow']:
            return 'high'
        if wind_speed > 15:  # 강풍 (> 54 km/h)
            return 'high'
        if visibility < 1000:  # 1km 미만
            return 'high'
        
        # Medium risk: 보통 악천후
        if condition == 'Rain':
            return 'medium'
        if wind_speed > 10:  # 중간 바람 (> 36 km/h)
            return 'medium'
        if visibility < 5000:  # 5km 미만
            return 'medium'
        
        # Low risk: 정상
        return 'low'
    
    def _get_weather_warning(self, weather: Dict) -> str:
        """날씨 경고 메시지 생성"""
        condition = weather['condition']
        wind_speed = weather['wind_speed']
        visibility = weather['visibility']
        
        warnings = []
        
        if condition == 'Thunderstorm':
            warnings.append("⚡ 뇌우 주의: 항공편 지연 가능성 높음")
        elif condition == 'Snow':
            warnings.append("❄️ 폭설 주의: 활주로 제빙으로 지연 예상")
        elif condition == 'Rain':
            warnings.append("🌧️ 비: 약간의 지연 가능")
        
        if wind_speed > 15:
            warnings.append(f"💨 강풍 ({wind_speed:.1f} m/s): 이착륙 지연 가능")
        
        if visibility < 1000:
            warnings.append(f"🌫️ 저시정 ({visibility}m): 운항 차질 우려")
        
        return " | ".join(warnings) if warnings else ""
    
    def _get_default_weather_data(self) -> Dict:
        """API 실패 시 기본 날씨 데이터"""
        return {
            'condition': 'Clear',
            'description': 'clear sky',
            'temperature': 15,
            'feels_like': 15,
            'humidity': 50,
            'pressure': 1013,
            'wind_speed': 3,
            'wind_deg': 0,
            'visibility': 10000,
            'clouds': 0,
            'timestamp': datetime.now()
        }
    
    def _get_default_weather(self, airport_code: str) -> Dict:
        """공항 코드를 찾을 수 없을 때 기본값"""
        return {
            'airport': f'{airport_code} Airport',
            'condition': 'Clear',
            'description': 'clear sky',
            'temperature': 15,
            'wind_speed': 3,
            'visibility': 10000,
            'delay_risk': 'unknown',
            'warning': ''
        }


# 편의 함수
def get_weather(airport_code: str, flight_time: datetime, api_key: Optional[str] = None) -> Dict:
    """
    공항 날씨 정보 조회 (간단한 인터페이스)
    
    Args:
        airport_code: 공항 코드
        flight_time: 비행 시간
        api_key: Google Maps API 키 (옵션)
        
    Returns:
        날씨 정보
    """
    weather_api = GoogleWeatherAPI(api_key)
    return weather_api.get_airport_weather(airport_code, flight_time)


if __name__ == '__main__':
    # 테스트
    from datetime import datetime, timedelta
    
    test_time = datetime.now() + timedelta(hours=3)
    
    print("=" * 60)
    print("Google Weather API 테스트")
    print("=" * 60)
    
    for airport in ['JFK', 'LAX', 'ORD']:
        print(f"\n📍 {airport} 공항:")
        weather = get_weather(airport, test_time)
        print(f"   - 날씨: {weather['condition']} ({weather['description']})")
        print(f"   - 온도: {weather['temperature']}°C")
        print(f"   - 풍속: {weather['wind_speed']} m/s")
        print(f"   - 가시거리: {weather['visibility']}m")
        print(f"   - 지연 위험: {weather['delay_risk'].upper()}")
        if weather['warning']:
            print(f"   - ⚠️ {weather['warning']}")
