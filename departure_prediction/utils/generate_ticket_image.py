"""
Generate airline ticket images.
Creates realistic airline ticket images using PIL.
"""
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
import os

def generate_ticket_image(flight_data, output_path):
    """
    Generate an airline ticket image.
    
    Args:
        flight_data: Flight information dict
        output_path: Output path
    """
    # Ticket size (width x height)
    width, height = 800, 400
    
    # Background colors (can vary by airline)
    bg_color = '#FFFFFF'
    primary_color = '#1E3A8A'  # Dark blue
    secondary_color = '#3B82F6'  # Light blue
    text_color = '#1F2937'  # Black
    
    # Create image
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Configure fonts (use system default fonts)
    try:
        title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 32)
        large_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
        medium_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
        small_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except:
        title_font = ImageFont.load_default()
        large_font = ImageFont.load_default()
        medium_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Top header (airline name)
    draw.rectangle([(0, 0), (width, 80)], fill=primary_color)
    draw.text((30, 25), flight_data['airline'].upper(), fill='white', font=title_font)
    
    # Flight number (top-right)
    draw.text((width - 200, 25), f"Flight {flight_data['flight_number']}", fill='white', font=large_font)
    
    # Passenger info
    y_pos = 110
    draw.text((30, y_pos), "PASSENGER", fill=secondary_color, font=small_font)
    draw.text((30, y_pos + 25), flight_data.get('passenger', 'John Smith'), fill=text_color, font=large_font)
    
    # Departure/arrival info (center)
    y_pos = 180
    
    # Departure
    draw.text((30, y_pos), "FROM", fill=secondary_color, font=small_font)
    draw.text((30, y_pos + 25), flight_data['origin'], fill=text_color, font=title_font)
    
    # Arrow
    draw.text((200, y_pos + 25), "→", fill=secondary_color, font=title_font)
    
    # Arrival
    draw.text((280, y_pos), "TO", fill=secondary_color, font=small_font)
    draw.text((280, y_pos + 25), flight_data['destination'], fill=text_color, font=title_font)
    
    # Date/time
    scheduled_dt = datetime.strptime(flight_data['scheduled_time'], '%Y-%m-%d %H:%M')
    date_str = scheduled_dt.strftime('%B %d, %Y')
    time_str = scheduled_dt.strftime('%H:%M')
    
    y_pos = 280
    draw.text((30, y_pos), "DATE", fill=secondary_color, font=small_font)
    draw.text((30, y_pos + 25), date_str, fill=text_color, font=medium_font)
    
    draw.text((280, y_pos), "DEPARTURE TIME", fill=secondary_color, font=small_font)
    draw.text((280, y_pos + 25), time_str, fill=text_color, font=medium_font)
    
    # Terminal/gate/seat (right)
    y_pos = 110
    x_pos = width - 220
    
    draw.text((x_pos, y_pos), "TERMINAL", fill=secondary_color, font=small_font)
    draw.text((x_pos, y_pos + 25), str(flight_data.get('terminal', 'N/A')), fill=text_color, font=large_font)
    
    y_pos = 180
    draw.text((x_pos, y_pos), "GATE", fill=secondary_color, font=small_font)
    draw.text((x_pos, y_pos + 25), str(flight_data.get('gate', 'N/A')), fill=text_color, font=large_font)
    
    y_pos = 250
    draw.text((x_pos, y_pos), "SEAT", fill=secondary_color, font=small_font)
    draw.text((x_pos, y_pos + 25), flight_data.get('seat', 'N/A'), fill=text_color, font=large_font)
    
    # Bottom barcode area (decorative)
    draw.rectangle([(0, height - 60), (width, height)], fill=primary_color)
    draw.text((30, height - 45), "BOARDING PASS", fill='white', font=medium_font)
    draw.text((width - 250, height - 45), f"{flight_data['origin']}-{flight_data['destination']}", fill='white', font=medium_font)
    
    # Save image
    img.save(output_path)
    print(f"✅ Ticket created: {output_path}")
    return output_path


if __name__ == '__main__':
    # Load sample ticket data for testing
    with open('../data/test_tickets_today.json', 'r') as f:
        tickets = json.load(f)
    
    # Create output directory
    os.makedirs('../test_tickets', exist_ok=True)
    
    # Generate each ticket image
    print("=" * 60)
    print("Airline Ticket Image Generation")
    print("=" * 60)
    
    for i, ticket in enumerate(tickets, 1):
        output_file = f"../test_tickets/ticket_{i}_{ticket['flight_number']}.png"
        generate_ticket_image(ticket, output_file)
    
    print(f"\n✅ Successfully generated {len(tickets)} ticket images")
    print(f"📁 Saved to: test_tickets/")
