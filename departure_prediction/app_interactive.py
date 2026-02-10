"""
Interactive Departure Time Recommendation System
User input: Ticket image or manual input + location + travel mode + baggage status
"""
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

# LLM/VLM 선택: Gemini 또는 Ollama
USE_GEMINI = os.getenv('USE_GEMINI', 'false').lower() == 'true'

if USE_GEMINI:
    from utils.gemini_direct_client import GeminiTicketOCR as TicketOCR
    print("🤖 Using Google Gemini Vision for ticket OCR")
else:
    from utils.ticket_ocr import TicketOCR
    print("🤖 Using Ollama Vision for ticket OCR")

from hybrid_predictor import HybridDeparturePredictor


def print_header():
    """Print header"""
    print("="*70)
    print("✈️  AI-Based Departure Time Recommendation System")
    print("="*70)
    print()


def get_flight_info_from_user():
    """Get flight information from user"""
    print("📋 Select flight information input method:")
    print("  1. Upload ticket image (automatic recognition)")
    print("  2. Manual input")
    
    while True:
        choice = input("\nSelect (1 or 2): ").strip()
        
        if choice == '1':
            return get_flight_info_from_image()
        elif choice == '2':
            return get_flight_info_manual()
        else:
            print("❌ Invalid input. Please enter 1 or 2.")


def get_flight_info_from_image():
    """Extract information from ticket image"""
    print("\n📸 Ticket Image Upload")
    print("Enter the image file path (e.g., /path/to/ticket.png)")
    
    while True:
        image_path = input("Image path: ").strip()
        
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None
            continue
        
        try:
            print(f"\n🔍 Analyzing image... ({'Gemini' if USE_GEMINI else 'LLaVA-Phi3'})")
            if USE_GEMINI:
                ocr = TicketOCR()
            else:
                ocr = TicketOCR(method='vision')
            flight_data = ocr.extract_with_vision(image_path)
            
            print("\n✅ Ticket information extracted:")
            print(f"   Flight: {flight_data.get('flight_number', 'N/A')}")
            print(f"   Departure: {flight_data.get('departure_time', 'N/A')}")
            print(f"   Airports: {flight_data.get('departure_airport', 'N/A')} → {flight_data.get('arrival_airport', 'N/A')}")
            print(f"   Terminal: {flight_data.get('terminal', 'N/A')}")
            print(f"   Gate: {flight_data.get('gate', 'N/A')}")
            print(f"   Checked baggage: {'Yes' if flight_data.get('has_checked_baggage') else 'No'}")
            print(f"   TSA PreCheck: {'Yes' if flight_data.get('has_tsa_precheck') else 'No'}")
            
            # Confirmation
            confirm = input("\nIs this information correct? (y/n): ").strip().lower()
            if confirm == 'y':
                return flight_data
            else:
                return None
        
        except Exception as e:
            print(f"\n❌ Image analysis failed: {e}")
            retry = input("Switch to manual input? (y/n): ").strip().lower()
            if retry == 'y':
                return None
            continue


def get_flight_info_manual():
    """Manually enter flight information"""
    print("\n✍️  Manual Flight Information Input")
    
    # Flight number
    while True:
        flight_number = input("\nFlight number (e.g., B6123, AA100): ").strip().upper()
        if len(flight_number) >= 3:
            break
        print("❌ Please enter a valid flight number.")
    
    # Departure airport
    print("\nDeparture airport code (e.g., JFK, LAX, ORD)")
    departure_airport = input("Departure airport: ").strip().upper()
    
    # Arrival airport
    arrival_airport = input("Arrival airport: ").strip().upper()
    
    # Departure date and time
    while True:
        print("\nDeparture date and time (e.g., 2026-02-05 19:00)")
        departure_time_str = input("Departure time: ").strip()
        try:
            datetime.strptime(departure_time_str, '%Y-%m-%d %H:%M')
            break
        except ValueError:
            print("❌ Please use the correct format (YYYY-MM-DD HH:MM)")
    
    # TSA PreCheck
    print("\nDo you have TSA PreCheck?")
    print("  1. Yes (reduced security wait time)")
    print("  2. No")
    has_tsa = input("Select (1 or 2): ").strip() == '1'
    
    return {
        'flight_number': flight_number,
        'departure_airport': departure_airport,
        'arrival_airport': arrival_airport,
        'departure_time': departure_time_str,
        'terminal': None,
        'has_checked_baggage': None,  # 나중에 입력받음
        'has_tsa_precheck': has_tsa
    }


def get_location():
    """Get departure location"""
    print("\n📍 Departure Location")
    print("Enter your current location or address (e.g., Times Square, New York, NY)")
    
    while True:
        address = input("Address: ").strip()
        if len(address) > 3:
            return address
        print("❌ Please enter a valid address.")


def get_travel_mode():
    """Select travel mode"""
    print("\n🚗 Travel Mode Selection")
    print("  1. Driving (DRIVE)")
    print("  2. Public Transit (TRANSIT)")
    print("  3. Walking (WALK)")
    print("  4. Bicycle (BICYCLE)")
    
    modes = {
        '1': 'DRIVE',
        '2': 'TRANSIT',
        '3': 'WALK',
        '4': 'BICYCLE'
    }
    
    while True:
        choice = input("\nSelect (1-4): ").strip()
        if choice in modes:
            return modes[choice]
        print("❌ Please select 1-4.")


def get_baggage_info(flight_data):
    """Get baggage information (if not extracted from image)"""
    if flight_data.get('has_checked_baggage') is not None:
        # If info already exists, just confirm
        print(f"\n🧳 Baggage info: {'Checked baggage' if flight_data['has_checked_baggage'] else 'Carry-on only'}")
        change = input("Change? (y/n): ").strip().lower()
        if change != 'y':
            return flight_data['has_checked_baggage']
    
    print("\n🧳 Checked Baggage")
    print("  1. Checked baggage (+30 min required)")
    print("  2. Carry-on only (no check-in)")
    
    while True:
        choice = input("Select (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice == '1'
        print("❌ Please select 1 or 2.")


def parse_flight_data(flight_data):
    """항공편 데이터 파싱"""
    # 항공편 번호에서 항공사 코드 추출
    flight_number = flight_data['flight_number']
    airline_code = ''.join([c for c in flight_number if c.isalpha()])
    
    # 항공사 이름 매핑
    airline_names = {
        'B6': 'JetBlue Airways',
        'AA': 'American Airlines',
        'DL': 'Delta Air Lines',
        'UA': 'United Airlines',
        'WN': 'Southwest Airlines',
        'NK': 'Spirit Airlines',
        'F9': 'Frontier Airlines',
        'AS': 'Alaska Airlines',
        'B': 'JetBlue Airways',  # 단축형
        'A': 'American Airlines'
    }
    
    # 날짜 시간 파싱 (여러 형식 지원)
    time_str = flight_data['departure_time']
    scheduled_time = None
    
    # 시도할 날짜 형식들
    time_formats = [
        '%Y-%m-%d %H:%M',
        '%Y-%m-%dT%H:%M',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S'
    ]
    
    for fmt in time_formats:
        try:
            scheduled_time = datetime.strptime(time_str, fmt)
            break
        except ValueError:
            continue
    
    if scheduled_time is None:
        raise ValueError(f"Cannot recognize date format: {time_str}")
    
    return {
        'airline_code': airline_code,
        'airline_name': airline_names.get(airline_code, airline_code),
        'flight_number': flight_number,
        'origin': flight_data['departure_airport'],
        'dest': flight_data['arrival_airport'],
        'scheduled_time': scheduled_time,
        'terminal': flight_data.get('terminal'),
        'gate': flight_data.get('gate'),
        'has_checked_baggage': flight_data['has_checked_baggage'],
        'has_tsa_precheck': flight_data.get('has_tsa_precheck', False)
    }


def main():
    """Main execution"""
    print_header()
    
    try:
        # 1. Flight information input
        flight_data = get_flight_info_from_user()
        
        if flight_data is None:
            # Manual input if image fails
            flight_data = get_flight_info_manual()
        
        # 2. Departure location
        address = get_location()
        
        # 3. Travel mode
        travel_mode = get_travel_mode()
        
        # 4. Baggage information
        has_baggage = get_baggage_info(flight_data)
        flight_data['has_checked_baggage'] = has_baggage
        
        # 5. Information confirmation
        print("\n" + "="*70)
        print("📋 Information Confirmation")
        print("="*70)
        print(f"Flight: {flight_data['flight_number']}")
        print(f"Departure: {flight_data['departure_time']}")
        print(f"Airports: {flight_data['departure_airport']} → {flight_data['arrival_airport']}")
        print(f"Departure location: {address}")
        print(f"Travel mode: {travel_mode}")
        print(f"Baggage: {'Checked baggage' if has_baggage else 'Carry-on only'}")
        print(f"TSA PreCheck: {'Yes' if flight_data.get('has_tsa_precheck') else 'No'}")
        print("="*70)
        
        confirm = input("\nProceed? (y/n): ").strip().lower()
        if confirm != 'y':
            print("\n❌ Cancelled.")
            return
        
        # 6. Run prediction
        print("\n🔍 Calculating optimal departure time...\n")
        
        flight_info = parse_flight_data(flight_data)
        
        predictor = HybridDeparturePredictor('models/delay_predictor_full.pkl')
        
        result = predictor.recommend_departure(
            address=address,
            flight_info=flight_info,
            travel_mode=travel_mode
        )
        
        # 7. Output results
        if result['success']:
            print("\n" + "="*70)
            print("✅ Departure Time Recommendation")
            print("="*70)
            print(result['recommendation'])
            print("="*70)
            
            print(f"\n📊 Detailed Information:")
            details = result['details']
            print(f"   - Recommended departure: {details['recommended_departure']}")
            print(f"   - Flight departure: {details['flight_time']}")
            print(f"   - Actual expected departure: {details['actual_departure']}")
            print(f"   - Travel time: {details['travel_time']} min")
            print(f"   - TSA wait: {details['tsa_wait']} min")
            print(f"   - Baggage check-in: {details['baggage_check']} min")
            print(f"   - Predicted delay: {details['predicted_delay']:.0f} min")
            print(f"   - Total time: {details['total_time']} min")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
    
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user.")
    except FileNotFoundError:
        print(f"\n⚠️ Trained model not found.")
        print(f"   Please run train_delay_predictor.ipynb first to train the model.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Thank you! Have a safe trip ✈️")
    print("="*70)


if __name__ == '__main__':
    main()
