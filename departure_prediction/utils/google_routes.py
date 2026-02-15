"""
Google Routes API integration module.
Retrieves route details and travel time from an origin address to JFK airport.
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
    """Google Routes API client."""
    
    # JFK airport terminal coordinates
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
            api_key: Google API key (reads from environment if missing)
        """
        self.api_key = api_key or os.getenv('GOOGLE_MAPS_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google API key is required. "
                "Set GOOGLE_MAPS_API_KEY in environment or pass api_key."
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
        Get route information from origin address to JFK airport.
        
        Args:
            origin_address: Origin address (e.g., "200 W 56th St, New York, NY 10019")
            terminal: Destination terminal (default: Terminal 4)
            departure_time: Departure time (default: now)
            traffic_model: Traffic prediction model ('best_guess', 'pessimistic', 'optimistic')
            travel_mode: Travel mode ('DRIVE', 'TRANSIT', 'WALK', 'BICYCLE', 'TWO_WHEELER')
            
        Returns:
            Dict: {
                'duration_in_traffic': int,  # Travel time with traffic info (seconds)
                'duration': int,             # Base travel time (seconds)
                'distance': int,             # Distance (meters)
                'route_summary': str,        # Route summary
                'departure_time': str,       # Departure time
                'arrival_time': str,         # Estimated arrival time
                'traffic_condition': str,    # Traffic condition
            }
        """
        if terminal not in self.JFK_TERMINALS:
            raise ValueError(f"Invalid terminal: {terminal}. Available: {list(self.JFK_TERMINALS.keys())}")
        
        if departure_time is None:
            departure_time = datetime.now()
        
        destination = self.JFK_TERMINALS[terminal]
        
        # API request headers
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.api_key,
            'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.polyline,routes.legs.steps'
        }
        
        # API request payload
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
        
        # Add routingPreference only for DRIVE mode
        if travel_mode == "DRIVE":
            payload["routingPreference"] = "TRAFFIC_AWARE"
            payload["departureTime"] = departure_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            payload["routeModifiers"] = {
                "avoidTolls": False,
                "avoidHighways": False,
                "avoidFerries": False
            }
        
        # Add transitPreferences only for TRANSIT mode
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
                raise Exception("No route found.")
            
            route = data['routes'][0]
            duration_seconds = int(route['duration'].rstrip('s'))
            distance_meters = route['distanceMeters']
            
            # Extract detailed transit route info
            transit_details = []
            if travel_mode == 'TRANSIT' and 'legs' in route and len(route['legs']) > 0:
                for leg in route['legs']:
                    if 'steps' in leg:
                        for step in leg['steps']:
                            if 'transitDetails' in step:
                                transit = step['transitDetails']
                                transit_line = transit.get('transitLine', {})
                                
                                # Line information
                                line_name = transit_line.get('nameShort', transit_line.get('name', 'Unknown'))
                                vehicle_type = transit_line.get('vehicle', {}).get('type', 'BUS')
                                
                                # Stop information
                                depart_stop = transit.get('stopDetails', {}).get('departureStop', {}).get('name', '')
                                arrival_stop = transit.get('stopDetails', {}).get('arrivalStop', {}).get('name', '')
                                
                                # Number of stops
                                stop_count = transit.get('stopCount', 0)
                                
                                transit_details.append({
                                    'line': line_name,
                                    'vehicle_type': vehicle_type,
                                    'from': depart_stop,
                                    'to': arrival_stop,
                                    'stops': stop_count
                                })
            
            # Calculate arrival time
            arrival_time = departure_time + timedelta(seconds=duration_seconds)
            
            # Determine traffic condition (compared to base travel time)
            base_duration = distance_meters / 13.41  # Average speed 30 mph = 13.41 m/s
            traffic_ratio = duration_seconds / base_duration
            
            if traffic_ratio < 1.2:
                traffic_condition = "Light"
            elif traffic_ratio < 1.5:
                traffic_condition = "Moderate"
            elif traffic_ratio < 2.0:
                traffic_condition = "Heavy"
            else:
                traffic_condition = "Severe"
            
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
            error_msg = f"API request failed: {str(e)}"
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
        Get route options for multiple departure times.
        
        Args:
            origin_address: Origin address
            terminal: Destination terminal
            flight_time: Flight time
            buffer_hours: Airport arrival buffer (hours)
            
        Returns:
            List[Dict]: Route information list by departure time
        """
        if flight_time is None:
            flight_time = datetime.now() + timedelta(hours=6)
        
        # Target airport arrival time
        target_arrival = flight_time - timedelta(hours=buffer_hours)
        
        results = []
        
        # Try multiple departure times (target time +/- 2 hours)
        for offset_minutes in [-120, -60, 0, 60]:
            test_departure = target_arrival + timedelta(minutes=offset_minutes)
            
            # Skip times in the past
            if test_departure < datetime.now():
                continue
            
            try:
                route_info = self.get_route_info(
                    origin_address=origin_address,
                    terminal=terminal,
                    departure_time=test_departure
                )
                
                # Compute recommendation score (higher when closer to target arrival time)
                arrival_time = datetime.fromisoformat(route_info['arrival_time'])
                time_diff_minutes = abs((arrival_time - target_arrival).total_seconds() / 60)
                score = max(0, 100 - time_diff_minutes)
                
                route_info['recommendation_score'] = score
                route_info['target_arrival'] = target_arrival.isoformat()
                
                results.append(route_info)
                
            except Exception as e:
                print(f"⚠️  Failed to fetch departure info for {test_departure.strftime('%H:%M')}: {str(e)}")
        
        # Sort by score
        results.sort(key=lambda x: x['recommendation_score'], reverse=True)
        
        return results


def format_duration(seconds: int) -> str:
    """Convert seconds to a human-readable format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def calculate_travel_time(origin: str, destination: str, travel_mode: str = 'DRIVE', departure_time: Optional[datetime] = None) -> Dict:
    """
    Convenience function: calculate travel time from address to airport.
    
    Args:
        origin: Origin address
        destination: Airport code (e.g., 'JFK')
        travel_mode: Travel mode ('DRIVE', 'TRANSIT', 'WALK', 'BICYCLE')
        departure_time: Departure time (uses current time if missing)
    
    Returns:
        {
            'success': bool,
            'duration_minutes': int,
            'distance_miles': float,
            'traffic_condition': str,
            'transit_details': list (TRANSIT only),
            'travel_mode': str,
            'error': str (on failure)
        }
    """
    try:
        routes_api = GoogleRoutesAPI()
        
        # Map airport code to terminal (default: Terminal 4)
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
    """Run test."""
    print("=== Google Routes API Test ===\n")
    
    # Check API key
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        print("⚠️  Set the GOOGLE_MAPS_API_KEY environment variable.")
        print("\nSetup steps:")
        print("1. Create an API key in Google Cloud Console")
        print("2. Enable the Routes API")
        print("3. Add GOOGLE_MAPS_API_KEY=your_key to your .env file")
        return
    
    try:
        routes_api = GoogleRoutesAPI()
        
        # Test address
        test_address = "200 W 56th St, New York, NY 10019"
        print(f"📍 Origin: {test_address}")
        print(f"📍 Destination: JFK Airport Terminal 4\n")
        
        # Fetch route info
        print("🚗 Fetching route information...\n")
        route_info = routes_api.get_route_info(
            origin_address=test_address,
            terminal='Terminal 4'
        )
        
        # Print results
        print("✓ Fetch complete!\n")
        print(f"Distance: {route_info['distance_miles']:.1f} miles ({route_info['distance']:,} meters)")
        print(f"Travel time: {format_duration(route_info['duration_in_traffic'])}")
        print(f"Traffic: {route_info['traffic_condition']}")
        print(f"Departure: {datetime.fromisoformat(route_info['departure_time']).strftime('%Y-%m-%d %H:%M')}")
        print(f"Estimated arrival: {datetime.fromisoformat(route_info['arrival_time']).strftime('%Y-%m-%d %H:%M')}")
        
        # Multiple departure-time options
        print("\n" + "="*50)
        print("🕐 Recommended Departure Times\n")
        
        flight_time = datetime.now() + timedelta(hours=6)
        print(f"Flight time: {flight_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"Target airport arrival: 3 hours before flight\n")
        
        options = routes_api.get_multiple_departure_times(
            origin_address=test_address,
            terminal='Terminal 4',
            flight_time=flight_time,
            buffer_hours=3
        )
        
        for i, option in enumerate(options, 1):
            dep_time = datetime.fromisoformat(option['departure_time'])
            arr_time = datetime.fromisoformat(option['arrival_time'])
            
            print(f"{i}. Depart: {dep_time.strftime('%H:%M')} → "
                  f"Arrive: {arr_time.strftime('%H:%M')} "
                  f"({format_duration(option['duration_in_traffic'])}, "
                  f"{option['traffic_condition']}) "
                  f"[score: {option['recommendation_score']:.0f}]")
        
        print("\n✅ Test complete!")
        
    except ValueError as e:
        print(f"❌ Configuration error: {str(e)}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
