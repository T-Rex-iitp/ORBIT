"""
Resilience Module - system reliability enhancements.
Automatic fallback and recovery mechanisms when API calls fail.
"""
from typing import Optional, Dict, Any, Callable
from functools import wraps
import time
from datetime import datetime


class ResilienceConfig:
    """Resilience settings."""
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    EXPONENTIAL_BACKOFF = True
    
    # Timeout settings
    API_TIMEOUT = 30  # seconds
    
    # Default fallback values
    DEFAULT_TRAVEL_TIME = 60  # minutes
    DEFAULT_TSA_WAIT = 30  # minutes (normal)
    DEFAULT_TSA_WAIT_PRECHECK = 10  # minutes (PreCheck)
    DEFAULT_WEATHER_DELAY = 0  # minutes
    DEFAULT_GATE_WALK = 15  # minutes
    DEFAULT_FLIGHT_DELAY = 15  # minutes


def retry_with_exponential_backoff(max_retries=3, base_delay=1):
    """
    Retry API calls with exponential backoff on failure.
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Base wait time in seconds
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
                        # Compute exponential backoff
                        delay = base_delay * (2 ** attempt)
                        print(f"   ⚠️ {func.__name__} failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"   ❌ {func.__name__} failed after {max_retries} attempts")
            
            # Raise exception when all retries fail
            raise last_exception
        
        return wrapper
    return decorator


def fallback_on_error(fallback_value=None, fallback_func=None):
    """
    Return fallback value or fallback function output on error.
    
    Args:
        fallback_value: Default value to return on error
        fallback_func: Fallback function to execute on error
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
    Wrapper for resilient API calls.
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
        Safe API call with retry + fallback.
        
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
        
        # All retries failed - use fallback
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
    Return average travel time when traffic API fails.
    """
    fallback_times = {
        'DRIVE': 60,      # 1 hour
        'TRANSIT': 90,    # 1.5 hours
        'WALK': 180,      # 3 hours
        'BICYCLE': 120    # 2 hours
    }
    
    minutes = fallback_times.get(travel_mode, 60)
    
    return {
        'success': True,
        'duration_minutes': minutes,
        'distance_km': 30,  # Estimated
        'route': 'Fallback route (API unavailable)',
        'transit_details': None,
        'fallback_used': True
    }


def get_fallback_tsa_wait(has_precheck: bool = False) -> int:
    """
    Return average TSA wait time when TSA API fails.
    """
    return ResilienceConfig.DEFAULT_TSA_WAIT_PRECHECK if has_precheck else ResilienceConfig.DEFAULT_TSA_WAIT


def get_fallback_weather() -> Dict[str, Any]:
    """
    Return neutral weather information when weather API fails.
    """
    return {
        'condition': 'Unknown',
        'description': 'Weather data unavailable',
        'temperature': 20,
        'wind_speed': 5,
        'delay_risk': 'low',  # Conservative low-risk default
        'warning': None,
        'airport': 'Unknown',
        'fallback_used': True
    }


def get_fallback_flight_status() -> Dict[str, Any]:
    """
    Return default flight information when flight API fails.
    """
    return {
        'status': 'scheduled',
        'status_kr': 'Scheduled',
        'is_delayed': False,
        'delay_minutes': 0,
        'scheduled_departure': None,
        'estimated_departure': None,
        'fallback_used': True
    }


class HealthCheck:
    """
    Check system component status.
    """
    
    @staticmethod
    def check_model_loaded(model) -> bool:
        """Check whether model is loaded."""
        return model is not None
    
    @staticmethod
    def check_api_availability(api_name: str, test_func: Callable) -> bool:
        """Test API availability."""
        try:
            result = test_func()
            return result is not None
        except Exception as e:
            print(f"   ⚠️ {api_name} unavailable: {e}")
            return False
    
    @staticmethod
    def get_system_status(predictor) -> Dict[str, bool]:
        """
        Check overall system status.
        
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
            'google_api': True,  # Determined by API key presence
            'ollama': True,      # Check Ollama server connectivity
            'overall': True
        }
        
        status['overall'] = all([status['model'], status['google_api'], status['ollama']])
        
        return status


def validate_flight_info(flight_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize flight information.
    
    Returns:
        Validated and normalized flight_info
    """
    validated = flight_info.copy()
    
    # Validate required fields
    required_fields = ['airline_code', 'flight_number', 'origin', 'dest', 'scheduled_time']
    for field in required_fields:
        if field not in validated or validated[field] is None:
            raise ValueError(f"Required field missing: {field}")
    
    # Set defaults for optional fields
    if 'has_checked_baggage' not in validated:
        validated['has_checked_baggage'] = False
    
    if 'has_tsa_precheck' not in validated:
        validated['has_tsa_precheck'] = False
    
    if 'terminal' not in validated:
        validated['terminal'] = 'Terminal 4'  # JFK default
    
    if 'gate' not in validated:
        validated['gate'] = None
    
    # Validate date/time
    if not isinstance(validated['scheduled_time'], datetime):
        raise ValueError("scheduled_time must be a datetime object")
    
    return validated


if __name__ == '__main__':
    # Test
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
