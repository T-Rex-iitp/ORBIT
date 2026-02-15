"""
Check real-time flight status and delay information.
Uses the AviationStack API.
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
    """Real-time flight status checker."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: AviationStack API key
        """
        self.api_key = api_key or os.getenv('AVIATIONSTACK_API_KEY')
        self.base_url = "http://api.aviationstack.com/v1/flights"
    
    def check_flight_status(self, flight_number: str, date: Optional[datetime] = None) -> Dict:
        """
        Check real-time status by flight number.
        
        Args:
            flight_number: Flight number (e.g., 'AA100', 'DL302')
            date: Departure date (default: today)
            
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
                'delay_reason': str,  # If available
                'gate': str,
                'terminal': str
            }
        """
        if not self.api_key:
            print("⚠️ API key is missing. Returning sample data.")
            return self._get_sample_status(flight_number)
        
        if date is None:
            date = datetime.now()
        
        params = {
            'access_key': self.api_key,
            'flight_iata': flight_number.upper()
        }
        
        try:
            print(f"🔍 Checking status for flight {flight_number}...")
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data or len(data['data']) == 0:
                print(f"⚠️ Flight {flight_number} not found.")
                return self._get_sample_status(flight_number)
            
            # Most recent flight info
            flight = data['data'][0]
            
            scheduled_str = flight.get('departure', {}).get('scheduled')
            estimated_str = flight.get('departure', {}).get('estimated')
            actual_str = flight.get('departure', {}).get('actual')
            
            scheduled_time = datetime.fromisoformat(scheduled_str.replace('Z', '+00:00')) if scheduled_str else None
            estimated_time = datetime.fromisoformat(estimated_str.replace('Z', '+00:00')) if estimated_str else scheduled_time
            actual_time = datetime.fromisoformat(actual_str.replace('Z', '+00:00')) if actual_str else None
            
            # Calculate delay minutes
            delay_minutes = flight.get('departure', {}).get('delay', 0) or 0
            
            # Check whether it is truly delayed
            is_delayed = False
            if estimated_time and scheduled_time:
                delay_minutes = int((estimated_time - scheduled_time).total_seconds() / 60)
                is_delayed = delay_minutes > 15  # Treat as delayed if 15+ minutes
            
            status = flight.get('flight_status', 'unknown')
            
            # Human-readable status label
            status_kr = {
                'scheduled': 'On schedule',
                'active': 'In flight',
                'landed': 'Landed',
                'cancelled': 'Cancelled',
                'delayed': 'Delayed',
                'diverted': 'Diverted',
                'unknown': 'No information'
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
            
            # Console output
            print(f"✅ {result['flight_number']} - {result['airline']}")
            print(f"   Status: {result['status_kr']}")
            print(f"   Route: {result['origin']} → {result['destination']}")
            print(f"   Scheduled: {scheduled_time.strftime('%Y-%m-%d %H:%M') if scheduled_time else 'N/A'}")
            if is_delayed:
                print(f"   ⚠️ Delay: {delay_minutes} min")
                print(f"   Estimated departure: {estimated_time.strftime('%Y-%m-%d %H:%M') if estimated_time else 'N/A'}")
            print(f"   Gate: {result['terminal']} - {result['gate']}")
            
            return result
            
        except Exception as e:
            print(f"❌ API call failed: {e}")
            return self._get_sample_status(flight_number)
    
    def _get_sample_status(self, flight_number: str) -> Dict:
        """Sample status data."""
        now = datetime.now()
        scheduled = now + timedelta(hours=3)
        
        return {
            'flight_number': flight_number,
            'airline': 'Sample Airlines',
            'status': 'scheduled',
            'status_kr': 'On schedule',
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
    Convenience function: check flight status.
    
    Args:
        flight_number: Flight number
        api_key: API key (optional)
        
    Returns:
        Flight status information
    """
    checker = FlightStatusChecker(api_key)
    return checker.check_flight_status(flight_number)


if __name__ == '__main__':
    # Test
    print("=" * 60)
    print("Real-time Flight Status Test")
    print("=" * 60)
    
    # Use sample flight number from collected data
    test_flights = ['AA100', 'DL302', 'B6623']
    
    for flight_num in test_flights:
        print(f"\n{'='*60}")
        status = check_flight(flight_num)
        print()
