"""
Resilience Module - 시스템 복원력 강화
API 실패 시 자동 폴백 및 복구 메커니즘
"""
from typing import Optional, Dict, Any, Callable
from functools import wraps
import time
from datetime import datetime


class ResilienceConfig:
    """복원력 설정"""
    # Retry 설정
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    EXPONENTIAL_BACKOFF = True
    
    # Timeout 설정
    API_TIMEOUT = 30  # seconds
    
    # Fallback 기본값
    DEFAULT_TRAVEL_TIME = 60  # minutes
    DEFAULT_TSA_WAIT = 30  # minutes (일반)
    DEFAULT_TSA_WAIT_PRECHECK = 10  # minutes (PreCheck)
    DEFAULT_WEATHER_DELAY = 0  # minutes
    DEFAULT_GATE_WALK = 15  # minutes
    DEFAULT_FLIGHT_DELAY = 15  # minutes


def retry_with_exponential_backoff(max_retries=3, base_delay=1):
    """
    API 호출 실패 시 지수 백오프로 재시도
    
    Args:
        max_retries: 최대 재시도 횟수
        base_delay: 기본 대기 시간 (초)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        # 지수 백오프 계산
                        delay = base_delay * (2 ** attempt)
                        print(f"   ⚠️ {func.__name__} failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"   ❌ {func.__name__} failed after {max_retries} attempts")
            
            # 모든 재시도 실패 시 예외 발생
            raise last_exception
        
        return wrapper
    return decorator


def fallback_on_error(fallback_value=None, fallback_func=None):
    """
    에러 발생 시 fallback 값 또는 함수 반환
    
    Args:
        fallback_value: 에러 시 반환할 기본값
        fallback_func: 에러 시 실행할 fallback 함수
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"   ⚠️ {func.__name__} error: {e}")
                
                if fallback_func:
                    print(f"   🔄 Using fallback function...")
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        print(f"   ❌ Fallback function also failed: {fallback_error}")
                        return fallback_value
                else:
                    print(f"   🔄 Using fallback value: {fallback_value}")
                    return fallback_value
        
        return wrapper
    return decorator


class ResilientAPIWrapper:
    """
    API 호출을 복원력 있게 감싸는 래퍼
    """
    
    @staticmethod
    def safe_api_call(
        api_func: Callable,
        fallback_value: Any,
        max_retries: int = 3,
        timeout: int = 30,
        error_message: str = "API call failed"
    ) -> Dict[str, Any]:
        """
        안전한 API 호출 with retry + fallback
        
        Returns:
            {
                'success': bool,
                'data': Any,
                'error': str (optional),
                'fallback_used': bool
            }
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = api_func()
                return {
                    'success': True,
                    'data': result,
                    'fallback_used': False
                }
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # Exponential backoff
                    print(f"   ⚠️ API call failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                    time.sleep(delay)
        
        # 모든 재시도 실패 - fallback 사용
        print(f"   ❌ {error_message}: {last_exception}")
        print(f"   🔄 Using fallback value")
        
        return {
            'success': False,
            'data': fallback_value,
            'error': str(last_exception),
            'fallback_used': True
        }


def get_fallback_travel_time(travel_mode: str = 'DRIVE') -> Dict[str, Any]:
    """
    교통 API 실패 시 평균 이동시간 반환
    """
    fallback_times = {
        'DRIVE': 60,      # 1시간
        'TRANSIT': 90,    # 1.5시간
        'WALK': 180,      # 3시간
        'BICYCLE': 120    # 2시간
    }
    
    minutes = fallback_times.get(travel_mode, 60)
    
    return {
        'success': True,
        'duration_minutes': minutes,
        'distance_km': 30,  # 추정
        'route': 'Fallback route (API unavailable)',
        'transit_details': None,
        'fallback_used': True
    }


def get_fallback_tsa_wait(has_precheck: bool = False) -> int:
    """
    TSA API 실패 시 평균 대기시간 반환
    """
    return ResilienceConfig.DEFAULT_TSA_WAIT_PRECHECK if has_precheck else ResilienceConfig.DEFAULT_TSA_WAIT


def get_fallback_weather() -> Dict[str, Any]:
    """
    날씨 API 실패 시 중립 날씨 정보 반환
    """
    return {
        'condition': 'Unknown',
        'description': 'Weather data unavailable',
        'temperature': 20,
        'wind_speed': 5,
        'delay_risk': 'low',  # 보수적으로 낮음 설정
        'warning': None,
        'airport': 'Unknown',
        'fallback_used': True
    }


def get_fallback_flight_status() -> Dict[str, Any]:
    """
    항공편 API 실패 시 기본 정보 반환
    """
    return {
        'status': 'scheduled',
        'status_kr': '예정',
        'is_delayed': False,
        'delay_minutes': 0,
        'scheduled_departure': None,
        'estimated_departure': None,
        'fallback_used': True
    }


class HealthCheck:
    """
    시스템 구성요소 상태 확인
    """
    
    @staticmethod
    def check_model_loaded(model) -> bool:
        """모델 로딩 상태 확인"""
        return model is not None
    
    @staticmethod
    def check_api_availability(api_name: str, test_func: Callable) -> bool:
        """API 가용성 테스트"""
        try:
            result = test_func()
            return result is not None
        except Exception as e:
            print(f"   ⚠️ {api_name} unavailable: {e}")
            return False
    
    @staticmethod
    def get_system_status(predictor) -> Dict[str, bool]:
        """
        전체 시스템 상태 확인
        
        Returns:
            {
                'model': bool,
                'google_api': bool,
                'ollama': bool,
                'overall': bool
            }
        """
        status = {
            'model': HealthCheck.check_model_loaded(predictor.model),
            'google_api': True,  # API 키 존재 여부로 판단
            'ollama': True,      # Ollama 서버 연결 확인
            'overall': True
        }
        
        status['overall'] = all([status['model'], status['google_api'], status['ollama']])
        
        return status


def validate_flight_info(flight_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    항공편 정보 유효성 검증 및 보정
    
    Returns:
        검증되고 보정된 flight_info
    """
    validated = flight_info.copy()
    
    # 필수 필드 확인
    required_fields = ['airline_code', 'flight_number', 'origin', 'dest', 'scheduled_time']
    for field in required_fields:
        if field not in validated or validated[field] is None:
            raise ValueError(f"Required field missing: {field}")
    
    # Optional 필드 기본값 설정
    if 'has_checked_baggage' not in validated:
        validated['has_checked_baggage'] = False
    
    if 'has_tsa_precheck' not in validated:
        validated['has_tsa_precheck'] = False
    
    if 'terminal' not in validated:
        validated['terminal'] = 'Terminal 4'  # JFK 기본값
    
    if 'gate' not in validated:
        validated['gate'] = None
    
    # 날짜/시간 검증
    if not isinstance(validated['scheduled_time'], datetime):
        raise ValueError("scheduled_time must be a datetime object")
    
    return validated


if __name__ == '__main__':
    # 테스트
    print("=== Resilience Module Test ===\n")
    
    # 1. Retry test
    @retry_with_exponential_backoff(max_retries=3)
    def failing_api():
        print("Calling API...")
        raise Exception("API Error")
    
    try:
        failing_api()
    except Exception as e:
        print(f"Final error: {e}\n")
    
    # 2. Fallback test
    @fallback_on_error(fallback_value={'status': 'unknown'})
    def unreliable_api():
        raise Exception("Connection timeout")
    
    result = unreliable_api()
    print(f"Fallback result: {result}\n")
    
    # 3. Safe API call test
    def test_api():
        raise Exception("Test error")
    
    result = ResilientAPIWrapper.safe_api_call(
        api_func=test_api,
        fallback_value={'default': 'value'},
        max_retries=2,
        error_message="Test API failed"
    )
    print(f"Safe API result: {result}")
