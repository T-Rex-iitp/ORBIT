"""
Google Gemini API (Direct) integration.
Uses Google AI API instead of Vertex AI (API key based).
"""
import os
import google.generativeai as genai
from typing import Optional, Dict, Any
from PIL import Image as PILImage


class GeminiDirectClient:
    """
    Google Gemini API client (Direct API).
    Works with only an API key (no Vertex AI required).
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-flash-latest"
    ):
        """
        Args:
            api_key: Google AI API key (auto-loaded from environment)
            model_name: Model name
                - gemini-flash-latest: Fast and low cost (recommended)
                - gemini-pro-latest: More accurate, higher cost
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model_name
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        # Initialize Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(model_name)
        
        print(f"✅ Gemini Direct API initialized: {model_name}")
    
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Generate text (LLM).
        
        Args:
            prompt: Input prompt
            temperature: Creativity (0.0~2.0)
            max_tokens: Maximum output tokens
            
        Returns:
            Generated text
        """
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            print(f"❌ Gemini text generation error: {e}")
            raise
    
    def analyze_image(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail."
    ) -> str:
        """
        Analyze image (Vision).
        
        Args:
            image_path: Image file path
            prompt: Analysis prompt
            
        Returns:
            Analysis result text
        """
        try:
            # Load image
            image = PILImage.open(image_path)
            
            # Analyze with image + prompt
            response = self.model.generate_content([prompt, image])
            
            return response.text
            
        except Exception as e:
            print(f"❌ Gemini image analysis error: {e}")
            raise
    
    def extract_ticket_info(
        self,
        image_path: str
    ) -> Dict[str, Any]:
        """
        Extract information from an airline ticket image.
        
        Args:
            image_path: Airline ticket image path
            
        Returns:
            Extracted info dictionary
        """
        prompt = """
        Extract flight information from this airline ticket image.
        Return ONLY a JSON object with these fields:
        {
            "flight_number": "string (e.g., B6123)",
            "airline": "string (e.g., JetBlue)",
            "departure_airport": "string (IATA code, e.g., JFK)",
            "arrival_airport": "string (IATA code, e.g., LAX)",
            "departure_time": "string (YYYY-MM-DD HH:MM format)",
            "arrival_time": "string (YYYY-MM-DD HH:MM format)",
            "passenger_name": "string",
            "seat": "string (e.g., 12A)",
            "gate": "string (e.g., B42)",
            "terminal": "string (e.g., 5)"
        }
        
        If any field is not found, use null.
        Return ONLY the JSON, no additional text.
        """
        
        try:
            result_text = self.analyze_image(image_path, prompt)
            
            # Extract JSON (remove ```json ... ```)
            import json
            import re
            
            # Find JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Raw JSON without fenced block
                json_str = result_text
            
            # Parse JSON
            ticket_info = json.loads(json_str.strip())
            
            print(f"✅ Ticket info extracted: {ticket_info.get('flight_number', 'N/A')}")
            return ticket_info
            
        except Exception as e:
            print(f"❌ Ticket extraction error: {e}")
            return {}
    
    def generate_departure_recommendation(
        self,
        flight_info: Dict[str, Any],
        travel_time_minutes: int,
        tsa_wait_minutes: int,
        weather_condition: str,
        predicted_delay_minutes: int
    ) -> str:
        """
        Generate departure-time recommendation.
        
        Args:
            flight_info: Flight information
            travel_time_minutes: Travel time to airport (minutes)
            tsa_wait_minutes: TSA wait time (minutes)
            weather_condition: Weather condition
            predicted_delay_minutes: Predicted delay (minutes)
            
        Returns:
            Natural-language recommendation text
        """
        prompt = f"""
        You are a helpful travel assistant. Generate a departure recommendation for a passenger.
        
        Flight Information:
        - Flight: {flight_info.get('flight_number', 'N/A')} ({flight_info.get('airline', 'N/A')})
        - Route: {flight_info.get('departure_airport', 'N/A')} → {flight_info.get('arrival_airport', 'N/A')}
        - Scheduled Departure: {flight_info.get('departure_time', 'N/A')}
        - Gate: {flight_info.get('gate', 'N/A')}
        - Terminal: {flight_info.get('terminal', 'N/A')}
        
        Travel Conditions:
        - Travel time to airport: {travel_time_minutes} minutes
        - TSA security wait: {tsa_wait_minutes} minutes
        - Weather: {weather_condition}
        - Predicted delay: {predicted_delay_minutes} minutes
        
        Generate a friendly, concise recommendation including:
        1. Recommended departure time from home
        2. Arrival time at airport
        3. Any weather or delay warnings
        4. Tips for smooth travel
        
        Keep it under 150 words.
        """
        
        try:
            recommendation = self.generate_text(prompt, temperature=0.7)
            return recommendation
            
        except Exception as e:
            print(f"❌ Recommendation generation error: {e}")
            return "Unable to generate recommendation at this time."


class GeminiTicketOCR:
    """
    TicketOCR-compatible class (drop-in replacement).
    Can be used immediately in existing code.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = GeminiDirectClient(api_key=api_key)
    
    def extract_with_vision(self, image_path: str) -> Dict[str, Any]:
        """Compatible method for TicketOCR.extract_with_vision()."""
        return self.client.extract_ticket_info(image_path)


# Test code
if __name__ == "__main__":
    import sys
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        print("Set it with: export GEMINI_API_KEY=your-api-key")
        sys.exit(1)
    
    # Initialize client
    client = GeminiDirectClient(api_key=api_key)
    
    # Text generation test
    print("\n=== Text Generation Test ===")
    response = client.generate_text("Say hello in 5 words")
    print(f"Response: {response}")
    
    print("\n✅ Gemini Direct API test passed!")
