"""
Module to fetch airport weather information using Google Weather API.
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


# Major airport coordinates
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
    """Weather lookup via Google Weather API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Google Maps API key (same key used for Routes API)
        """
        self.api_key = api_key or os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google Maps API key is required. "
                "Set GOOGLE_MAPS_API_KEY in environment or pass api_key."
            )
        self.base_url = "https://weather.googleapis.com/v1/currentConditions:lookup"
    
    def get_airport_weather(self, airport_code: str, flight_time: datetime) -> Dict:
        """
        Get airport weather information.
        
        Args:
            airport_code: Airport code (e.g., 'JFK')
            flight_time: Flight time
            
        Returns:
            Weather information dictionary
        """
        # Get airport coordinates
        coords = AIRPORT_COORDINATES.get(airport_code.upper())
        if not coords:
            print(f"⚠️ Airport code {airport_code} not found.")
            return self._get_default_weather(airport_code)
        
        # Calculate remaining time until departure
        now = datetime.now()
        hours_until_flight = (flight_time - now).total_seconds() / 3600
        
        # Fetch current weather (Google Weather API provides current weather only)
        weather_data = self._get_current_weather(coords['lat'], coords['lon'])
        
        # Add warning when departure is still far away
        weather_note = ""
        if hours_until_flight > 6:
            weather_note = f"({hours_until_flight:.0f} hours until departure - recheck before leaving)"
        elif hours_until_flight < 0:
            weather_note = "(time already passed)"
        
        # Evaluate delay risk
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
        Get current weather by coordinates (Google Weather API).
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Weather information dictionary
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
                
                # Google Weather API response fields are at the root level
                # Convert weather code to `condition`
                weather_condition = data.get('weatherCondition', {})
                weather_type = weather_condition.get('type', 'CLEAR')
                condition = self._map_weather_code(weather_type)
                description = weather_condition.get('description', {}).get('text', 'clear')
                
                # Temperature: use the `degrees` field
                temp_data = data.get('temperature', {})
                temp_celsius = temp_data.get('degrees', 15)
                
                # Wind speed: m/s
                wind_data = data.get('wind', {})
                wind_speed_mps = wind_data.get('speedMetersPerSecond', 0)
                
                # Visibility: meters
                visibility_data = data.get('visibility', {})
                visibility_m = visibility_data.get('distanceMeters', 10000)
                
                return {
                    'condition': condition,
                    'description': description.lower(),
                    'temperature': round(temp_celsius, 1),
                    'feels_like': round(data.get('feelsLikeTemperature', {}).get('degrees', temp_celsius), 1),
                    'humidity': data.get('relativeHumidity', 50),
                    'pressure': 1013,  # Default value
                    'wind_speed': round(wind_speed_mps, 1),
                    'wind_deg': wind_data.get('directionDegrees', 0),
                    'visibility': visibility_m,
                    'clouds': 0,  # Not provided by this API
                    'timestamp': datetime.now()
                }
            else:
                print(f"⚠️ Google Weather API error: {response.status_code} - {response.text}")
                return self._get_default_weather_data()
                
        except Exception as e:
            print(f"⚠️ Weather lookup failed: {e}")
            return self._get_default_weather_data()
    
    def _map_weather_code(self, code: str) -> str:
        """Map Google weather code to simple condition label."""
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
        Assess delay risk based on weather conditions.
        
        Returns:
            'low', 'medium', 'high'
        """
        condition = weather['condition']
        wind_speed = weather['wind_speed']  # m/s
        visibility = weather['visibility']   # meters
        
        # High risk: severe weather
        if condition in ['Thunderstorm', 'Snow']:
            return 'high'
        if wind_speed > 15:  # Strong wind (> 54 km/h)
            return 'high'
        if visibility < 1000:  # Under 1 km
            return 'high'
        
        # Medium risk: moderate weather issues
        if condition == 'Rain':
            return 'medium'
        if wind_speed > 10:  # Moderate wind (> 36 km/h)
            return 'medium'
        if visibility < 5000:  # Under 5 km
            return 'medium'
        
        # Low risk: normal conditions
        return 'low'
    
    def _get_weather_warning(self, weather: Dict) -> str:
        """Generate weather warning message."""
        condition = weather['condition']
        wind_speed = weather['wind_speed']
        visibility = weather['visibility']
        
        warnings = []
        
        if condition == 'Thunderstorm':
            warnings.append("⚡ Thunderstorm alert: high chance of flight delay")
        elif condition == 'Snow':
            warnings.append("❄️ Heavy snow alert: runway de-icing may cause delays")
        elif condition == 'Rain':
            warnings.append("🌧️ Rain: possible minor delays")
        
        if wind_speed > 15:
            warnings.append(f"💨 Strong wind ({wind_speed:.1f} m/s): takeoff/landing delays possible")
        
        if visibility < 1000:
            warnings.append(f"🌫️ Low visibility ({visibility}m): operational disruptions possible")
        
        return " | ".join(warnings) if warnings else ""
    
    def _get_default_weather_data(self) -> Dict:
        """Return default weather data if API fails."""
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
        """Return defaults when airport code is unknown."""
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


# Convenience function
def get_weather(airport_code: str, flight_time: datetime, api_key: Optional[str] = None) -> Dict:
    """
    Get airport weather information (simple interface).
    
    Args:
        airport_code: Airport code
        flight_time: Flight time
        api_key: Google Maps API key (optional)
        
    Returns:
        Weather information
    """
    weather_api = GoogleWeatherAPI(api_key)
    return weather_api.get_airport_weather(airport_code, flight_time)


if __name__ == '__main__':
    # Test
    from datetime import datetime, timedelta
    
    test_time = datetime.now() + timedelta(hours=3)
    
    print("=" * 60)
    print("Google Weather API Test")
    print("=" * 60)
    
    for airport in ['JFK', 'LAX', 'ORD']:
        print(f"\n📍 {airport} Airport:")
        weather = get_weather(airport, test_time)
        print(f"   - Weather: {weather['condition']} ({weather['description']})")
        print(f"   - Temperature: {weather['temperature']}°C")
        print(f"   - Wind speed: {weather['wind_speed']} m/s")
        print(f"   - Visibility: {weather['visibility']}m")
        print(f"   - Delay risk: {weather['delay_risk'].upper()}")
        if weather['warning']:
            print(f"   - ⚠️ {weather['warning']}")
