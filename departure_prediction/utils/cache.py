"""
Cache Module - 서버 장애 시에도 시스템 작동 보장
API 응답 캐싱 및 로컬 폴백 데이터 관리
"""
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib


class CacheManager:
    """
    API 응답 캐싱 및 폴백 관리
    서버가 죽어도 이전 데이터로 계속 작동
    """
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 캐시 유효 기간 설정
        self.ttl = {
            'flight_status': timedelta(minutes=10),  # 항공편 상태: 10분
            'weather': timedelta(hours=1),           # 날씨: 1시간
            'tsa_wait': timedelta(hours=2),          # TSA 대기: 2시간
            'travel_time': timedelta(hours=6),       # 교통 시간: 6시간
            'airline_data': timedelta(days=30),      # 항공사 정보: 30일
            'airport_data': timedelta(days=90),      # 공항 정보: 90일
        }
    
    def _get_cache_key(self, category: str, **kwargs) -> str:
        """캐시 키 생성"""
        # 파라미터를 정렬하여 일관된 키 생성
        params = "_".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key = f"{category}_{params}"
        # 파일 시스템 안전을 위해 해시
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """캐시 파일 경로"""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, category: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        캐시에서 데이터 조회
        
        Returns:
            None: 캐시 없음 또는 만료됨
            Dict: 캐시된 데이터
        """
        cache_key = self._get_cache_key(category, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            
            # 만료 확인
            cached_time = datetime.fromisoformat(cached['timestamp'])
            ttl = self.ttl.get(category, timedelta(hours=1))
            
            if datetime.now() - cached_time > ttl:
                # 만료됨
                return None
            
            return cached['data']
        
        except Exception as e:
            print(f"   ⚠️ Cache read error: {e}")
            return None
    
    def set(self, category: str, data: Any, **kwargs):
        """
        캐시에 데이터 저장
        """
        cache_key = self._get_cache_key(category, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cached = {
                'timestamp': datetime.now().isoformat(),
                'category': category,
                'params': kwargs,
                'data': data
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cached, f, indent=2, default=str)
        
        except Exception as e:
            print(f"   ⚠️ Cache write error: {e}")
    
    def get_stale(self, category: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        만료된 캐시라도 반환 (서버 장애 시 사용)
        
        Returns:
            None: 캐시 없음
            Dict: 캐시된 데이터 (만료되었어도 반환)
        """
        cache_key = self._get_cache_key(category, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            
            cached_time = datetime.fromisoformat(cached['timestamp'])
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600
            
            print(f"   📦 Using stale cache ({age_hours:.1f} hours old)")
            return cached['data']
        
        except Exception as e:
            print(f"   ⚠️ Stale cache read error: {e}")
            return None
    
    def clear(self, category: Optional[str] = None):
        """캐시 삭제"""
        if category:
            # 특정 카테고리만 삭제
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                    if cached.get('category') == category:
                        cache_file.unlink()
                except:
                    pass
        else:
            # 전체 삭제
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()


class HistoricalDataFallback:
    """
    과거 데이터 기반 폴백
    서버가 완전히 죽었을 때 사용할 통계 데이터
    """
    
    def __init__(self, data_file: str = "cache/historical_data.pkl"):
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(exist_ok=True)
        self.historical_data = self._load_data()
    
    def _load_data(self) -> Dict:
        """저장된 통계 데이터 로드"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        # 기본 통계 데이터
        return {
            'flight_delays': {},      # {airline_code: {route: avg_delay}}
            'tsa_wait_times': {},     # {airport: {hour: avg_wait}}
            'travel_times': {},       # {origin_dest: {mode: avg_time}}
            'weather_patterns': {},   # {airport: {month: delay_risk}}
        }
    
    def _save_data(self):
        """통계 데이터 저장"""
        try:
            with open(self.data_file, 'wb') as f:
                pickle.dump(self.historical_data, f)
        except Exception as e:
            print(f"   ⚠️ Failed to save historical data: {e}")
    
    def update_flight_delay(self, airline: str, route: str, delay: float):
        """항공편 지연 통계 업데이트"""
        if airline not in self.historical_data['flight_delays']:
            self.historical_data['flight_delays'][airline] = {}
        
        if route not in self.historical_data['flight_delays'][airline]:
            self.historical_data['flight_delays'][airline][route] = []
        
        self.historical_data['flight_delays'][airline][route].append(delay)
        
        # 최근 100개만 유지
        if len(self.historical_data['flight_delays'][airline][route]) > 100:
            self.historical_data['flight_delays'][airline][route] = \
                self.historical_data['flight_delays'][airline][route][-100:]
        
        self._save_data()
    
    def get_avg_flight_delay(self, airline: str, route: str) -> float:
        """평균 항공편 지연 시간"""
        try:
            delays = self.historical_data['flight_delays'].get(airline, {}).get(route, [])
            return sum(delays) / len(delays) if delays else 15.0  # 기본 15분
        except:
            return 15.0
    
    def update_tsa_wait(self, airport: str, hour: int, wait_time: int):
        """TSA 대기 시간 통계 업데이트"""
        if airport not in self.historical_data['tsa_wait_times']:
            self.historical_data['tsa_wait_times'][airport] = {}
        
        if hour not in self.historical_data['tsa_wait_times'][airport]:
            self.historical_data['tsa_wait_times'][airport][hour] = []
        
        self.historical_data['tsa_wait_times'][airport][hour].append(wait_time)
        
        # 최근 50개만 유지
        if len(self.historical_data['tsa_wait_times'][airport][hour]) > 50:
            self.historical_data['tsa_wait_times'][airport][hour] = \
                self.historical_data['tsa_wait_times'][airport][hour][-50:]
        
        self._save_data()
    
    def get_avg_tsa_wait(self, airport: str, hour: int, has_precheck: bool = False) -> int:
        """평균 TSA 대기 시간"""
        try:
            waits = self.historical_data['tsa_wait_times'].get(airport, {}).get(hour, [])
            avg = sum(waits) / len(waits) if waits else (10 if has_precheck else 30)
            return int(avg)
        except:
            return 10 if has_precheck else 30
    
    def update_travel_time(self, origin: str, dest: str, mode: str, time: int):
        """교통 시간 통계 업데이트"""
        key = f"{origin}_{dest}"
        
        if key not in self.historical_data['travel_times']:
            self.historical_data['travel_times'][key] = {}
        
        if mode not in self.historical_data['travel_times'][key]:
            self.historical_data['travel_times'][key][mode] = []
        
        self.historical_data['travel_times'][key][mode].append(time)
        
        # 최근 50개만 유지
        if len(self.historical_data['travel_times'][key][mode]) > 50:
            self.historical_data['travel_times'][key][mode] = \
                self.historical_data['travel_times'][key][mode][-50:]
        
        self._save_data()
    
    def get_avg_travel_time(self, origin: str, dest: str, mode: str) -> int:
        """평균 교통 시간"""
        key = f"{origin}_{dest}"
        try:
            times = self.historical_data['travel_times'].get(key, {}).get(mode, [])
            return int(sum(times) / len(times)) if times else 60  # 기본 1시간
        except:
            return 60


# 전역 인스턴스
cache_manager = CacheManager()
historical_fallback = HistoricalDataFallback()


def cached_api_call(category: str, api_func, use_stale_on_error: bool = True, **cache_params):
    """
    캐시를 활용한 안전한 API 호출
    
    Args:
        category: 캐시 카테고리
        api_func: API 호출 함수
        use_stale_on_error: 에러 시 만료된 캐시 사용 여부
        **cache_params: 캐시 키 생성용 파라미터
    
    Returns:
        API 응답 또는 캐시된 데이터
    """
    # 1. 유효한 캐시 확인
    cached = cache_manager.get(category, **cache_params)
    if cached:
        print(f"   📦 Using cached {category} data")
        return cached
    
    # 2. API 호출 시도
    try:
        result = api_func()
        
        # 성공 시 캐시 저장
        cache_manager.set(category, result, **cache_params)
        
        return result
    
    except Exception as e:
        print(f"   ❌ API call failed: {e}")
        
        # 3. 만료된 캐시라도 사용 (서버 장애 대응)
        if use_stale_on_error:
            stale = cache_manager.get_stale(category, **cache_params)
            if stale:
                print(f"   🔄 Using stale cache due to API failure")
                return stale
        
        # 4. 캐시도 없으면 예외 발생
        raise


if __name__ == '__main__':
    # 테스트
    print("=== Cache Module Test ===\n")
    
    # 1. 캐시 저장/조회
    cache_manager.set('test', {'value': 123}, key='test_key')
    result = cache_manager.get('test', key='test_key')
    print(f"1. Cache test: {result}\n")
    
    # 2. 만료된 캐시 조회
    import time
    cache_manager.ttl['test'] = timedelta(seconds=1)
    cache_manager.set('test', {'value': 456}, key='expire_test')
    time.sleep(2)
    result = cache_manager.get('test', key='expire_test')
    print(f"2. Expired cache (should be None): {result}")
    
    stale = cache_manager.get_stale('test', key='expire_test')
    print(f"   Stale cache: {stale}\n")
    
    # 3. Historical fallback
    historical_fallback.update_flight_delay('B6', 'JFK-LAX', 20)
    historical_fallback.update_flight_delay('B6', 'JFK-LAX', 25)
    avg = historical_fallback.get_avg_flight_delay('B6', 'JFK-LAX')
    print(f"3. Historical avg delay: {avg} min")
