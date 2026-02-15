"""
Cache Module - keeps the system running even during server outages.
Manages API response caching and local fallback data.
"""
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib


class CacheManager:
    """
    API response caching and fallback management.
    Keeps the system running with previous data even when servers are down.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Cache TTL settings
        self.ttl = {
            'flight_status': timedelta(minutes=10),  # Flight status: 10 min
            'fr24_adsb_delay': timedelta(minutes=10),# FR24+ADS-B prior-leg delay: 10 min
            'weather': timedelta(hours=1),           # Weather: 1 hour
            'tsa_wait': timedelta(hours=2),          # TSA wait: 2 hours
            'travel_time': timedelta(hours=6),       # Travel time: 6 hours
            'airline_data': timedelta(days=30),      # Airline info: 30 days
            'airport_data': timedelta(days=90),      # Airport info: 90 days
        }
    
    def _get_cache_key(self, category: str, **kwargs) -> str:
        """Generate a cache key."""
        # Sort parameters to generate a consistent key
        params = "_".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key = f"{category}_{params}"
        # Hash for filesystem-safe key
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Return cache file path."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, category: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Read data from cache.
        
        Returns:
            None: Cache missing or expired.
            Dict: Cached data.
        """
        cache_key = self._get_cache_key(category, **kwargs)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            
            # Check expiration
            cached_time = datetime.fromisoformat(cached['timestamp'])
            ttl = self.ttl.get(category, timedelta(hours=1))
            
            if datetime.now() - cached_time > ttl:
                # Expired
                return None
            
            return cached['data']
        
        except Exception as e:
            print(f"   ⚠️ Cache read error: {e}")
            return None
    
    def set(self, category: str, data: Any, **kwargs):
        """
        Store data in cache.
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
        Return stale cache if available (used during server outages).
        
        Returns:
            None: No cache.
            Dict: Cached data (returned even if expired).
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
        """Clear cache."""
        if category:
            # Delete only a specific category
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                    if cached.get('category') == category:
                        cache_file.unlink()
                except:
                    pass
        else:
            # Delete all
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()


class HistoricalDataFallback:
    """
    Historical-data fallback.
    Statistical data used when upstream servers are completely unavailable.
    """
    
    def __init__(self, data_file: str = "cache/historical_data.pkl"):
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(exist_ok=True)
        self.historical_data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load saved statistical data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        # Default statistical data
        return {
            'flight_delays': {},      # {airline_code: {route: avg_delay}}
            'tsa_wait_times': {},     # {airport: {hour: avg_wait}}
            'travel_times': {},       # {origin_dest: {mode: avg_time}}
            'weather_patterns': {},   # {airport: {month: delay_risk}}
        }
    
    def _save_data(self):
        """Save statistical data."""
        try:
            with open(self.data_file, 'wb') as f:
                pickle.dump(self.historical_data, f)
        except Exception as e:
            print(f"   ⚠️ Failed to save historical data: {e}")
    
    def update_flight_delay(self, airline: str, route: str, delay: float):
        """Update flight-delay statistics."""
        if airline not in self.historical_data['flight_delays']:
            self.historical_data['flight_delays'][airline] = {}
        
        if route not in self.historical_data['flight_delays'][airline]:
            self.historical_data['flight_delays'][airline][route] = []
        
        self.historical_data['flight_delays'][airline][route].append(delay)
        
        # Keep only the most recent 100 entries
        if len(self.historical_data['flight_delays'][airline][route]) > 100:
            self.historical_data['flight_delays'][airline][route] = \
                self.historical_data['flight_delays'][airline][route][-100:]
        
        self._save_data()
    
    def get_avg_flight_delay(self, airline: str, route: str) -> float:
        """Get average flight delay."""
        try:
            delays = self.historical_data['flight_delays'].get(airline, {}).get(route, [])
            return sum(delays) / len(delays) if delays else 15.0  # Default 15 min
        except:
            return 15.0
    
    def update_tsa_wait(self, airport: str, hour: int, wait_time: int):
        """Update TSA wait-time statistics."""
        if airport not in self.historical_data['tsa_wait_times']:
            self.historical_data['tsa_wait_times'][airport] = {}
        
        if hour not in self.historical_data['tsa_wait_times'][airport]:
            self.historical_data['tsa_wait_times'][airport][hour] = []
        
        self.historical_data['tsa_wait_times'][airport][hour].append(wait_time)
        
        # Keep only the most recent 50 entries
        if len(self.historical_data['tsa_wait_times'][airport][hour]) > 50:
            self.historical_data['tsa_wait_times'][airport][hour] = \
                self.historical_data['tsa_wait_times'][airport][hour][-50:]
        
        self._save_data()
    
    def get_avg_tsa_wait(self, airport: str, hour: int, has_precheck: bool = False) -> int:
        """Get average TSA wait time."""
        try:
            waits = self.historical_data['tsa_wait_times'].get(airport, {}).get(hour, [])
            avg = sum(waits) / len(waits) if waits else (10 if has_precheck else 30)
            return int(avg)
        except:
            return 10 if has_precheck else 30
    
    def update_travel_time(self, origin: str, dest: str, mode: str, time: int):
        """Update travel-time statistics."""
        key = f"{origin}_{dest}"
        
        if key not in self.historical_data['travel_times']:
            self.historical_data['travel_times'][key] = {}
        
        if mode not in self.historical_data['travel_times'][key]:
            self.historical_data['travel_times'][key][mode] = []
        
        self.historical_data['travel_times'][key][mode].append(time)
        
        # Keep only the most recent 50 entries
        if len(self.historical_data['travel_times'][key][mode]) > 50:
            self.historical_data['travel_times'][key][mode] = \
                self.historical_data['travel_times'][key][mode][-50:]
        
        self._save_data()
    
    def get_avg_travel_time(self, origin: str, dest: str, mode: str) -> int:
        """Get average travel time."""
        key = f"{origin}_{dest}"
        try:
            times = self.historical_data['travel_times'].get(key, {}).get(mode, [])
            return int(sum(times) / len(times)) if times else 60  # Default 1 hour
        except:
            return 60


# Global instances
cache_manager = CacheManager()
historical_fallback = HistoricalDataFallback()


def cached_api_call(category: str, api_func, use_stale_on_error: bool = True, **cache_params):
    """
    Safe API call wrapper with caching.
    
    Args:
        category: Cache category.
        api_func: API call function.
        use_stale_on_error: Whether to use stale cache on failure.
        **cache_params: Parameters used to generate cache key.
    
    Returns:
        API response or cached data.
    """
    # 1. Check valid cache
    cached = cache_manager.get(category, **cache_params)
    if cached:
        print(f"   📦 Using cached {category} data")
        return cached
    
    # 2. Try API call
    try:
        result = api_func()
        
        # Cache on success
        cache_manager.set(category, result, **cache_params)
        
        return result
    
    except Exception as e:
        print(f"   ❌ API call failed: {e}")
        
        # 3. Use stale cache on failure (outage resilience)
        if use_stale_on_error:
            stale = cache_manager.get_stale(category, **cache_params)
            if stale:
                print(f"   🔄 Using stale cache due to API failure")
                return stale
        
        # 4. Raise if no cache is available
        raise


if __name__ == '__main__':
    # Test
    print("=== Cache Module Test ===\n")
    
    # 1. Cache write/read
    cache_manager.set('test', {'value': 123}, key='test_key')
    result = cache_manager.get('test', key='test_key')
    print(f"1. Cache test: {result}\n")
    
    # 2. Expired cache read
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
