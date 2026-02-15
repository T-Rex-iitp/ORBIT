"""
Collect real JFK departure flight data.
Uses the AviationStack API (free tier).
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
    """Real flight data collector."""
    
    # Free API options
    AVIATIONSTACK_URL = "http://api.aviationstack.com/v1/flights"
    
    # Major U.S. domestic airports
    DOMESTIC_AIRPORTS = {
        'ATL', 'LAX', 'ORD', 'DFW', 'DEN', 'SFO', 'SEA', 'LAS', 'MCO',
        'CLT', 'PHX', 'IAH', 'MIA', 'BOS', 'MSP', 'FLL', 'DTW', 'PHL',
        'BWI', 'SLC', 'SAN', 'DCA', 'MDW', 'TPA', 'PDX', 'STL', 'HNL',
    }
    
    def __init__(self, api_key: str = None):
        """
        Args:
            api_key: AviationStack API key (free at aviationstack.com)
        """
        self.api_key = api_key or os.getenv('AVIATIONSTACK_API_KEY')
    
    def get_jfk_departures_today(self, limit: int = 50) -> List[Dict]:
        """
        Fetch today's JFK departures.
        
        Returns:
            List[Dict]: Flight information
        """
        if not self.api_key:
            print("⚠️  AviationStack API key is missing. Returning sample data.")
            return self._get_sample_data()
        
        params = {
            'access_key': self.api_key,
            'dep_iata': 'JFK',
            'limit': limit,
        }
        
        try:
            print("🌐 Calling AviationStack API...")
            response = requests.get(self.AVIATIONSTACK_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                print(f"⚠️  API response error: {data}")
                return self._get_sample_data()
            
            flights = []
            for flight in data['data']:
                # Filter to domestic flights only
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
            
            print(f"✅ Collected {len(flights)} domestic flights")
            return flights
            
        except requests.RequestException as e:
            print(f"❌ API call error: {str(e)}")
            return self._get_sample_data()
    
    def _get_sample_data(self) -> List[Dict]:
        """
        Sample data based on real JFK schedules.
        Built from historical or typical scheduling patterns.
        """
        print("📋 Using sample data based on real JFK schedules")
        
        base = datetime.now()
        
        # Based on real JFK domestic schedules (typical pattern)
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
        """Save flight data to JSON."""
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
        
        print(f"💾 Saved: {filepath}")
        return filepath


def main():
    """Test runner."""
    print("=" * 70)
    print("    ✈️  Real JFK Departure Flight Data Collection")
    print("=" * 70)
    print()
    
    collector = RealFlightDataCollector()
    flights = collector.get_jfk_departures_today(limit=30)
    
    if flights:
        print(f"\n📊 Collection result: {len(flights)} domestic flights")
        
        for flight in flights[:5]:
            print(f"  ✈️  {flight['flight_number']} → {flight['destination']}")
            print(f"     Scheduled: {flight['scheduled_time']}")
            if flight.get('actual_time'):
                print(f"     Actual: {flight['actual_time']} (Delay: {flight.get('delay', 0)} min)")
            print()
        
        filepath = collector.save_to_json(flights)
        print("✅ Done!")
        print()
        print("💡 Next steps:")
        print("   1. Generate departure-time recommendations for each flight in our system")
        print("   2. Compare recommended departure time vs actual required time")
        print("   3. Evaluate accuracy considering delay data")


if __name__ == "__main__":
    main()
