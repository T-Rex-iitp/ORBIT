"""
Data collection module.
Collects JFK airport security wait times and flight information, then saves them to CSV.
"""
import os
import time
import csv
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

JFK_URL = "https://www.jfkairport.com"
SECURITY_WAIT_FORM_XPATH = '//*[@id="security-wait"]/form'


class JFKDataCollector:
    """JFK airport data collection class."""
    
    def __init__(self, output_dir: str = "data/collected"):
        """
        Args:
            output_dir: Directory where collected data will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.driver = None
        
    def create_driver(self, headless: bool = True) -> webdriver.Chrome:
        """Create a Chrome WebDriver."""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Configure ChromeDriver path
        crawl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crawl")
        driver_path = os.path.join(crawl_dir, "chromedriver-mac-x64", "chromedriver")
        
        if os.path.exists(driver_path):
            service = Service(executable_path=driver_path)
            return webdriver.Chrome(service=service, options=chrome_options)
        else:
            # If ChromeDriver is available in PATH
            return webdriver.Chrome(options=chrome_options)
    
    def collect_security_wait_time(self) -> dict:
        """
        Collect security wait-time data.
        
        Returns:
            dict: Collected data (timestamp, terminal, wait_time, etc.)
        """
        self.driver = self.create_driver(headless=True)
        wait = WebDriverWait(self.driver, 10)
        
        collected_data = {
            'timestamp': datetime.now().isoformat(),
            'terminals': []
        }
        
        try:
            self.driver.get(JFK_URL)
            
            form_element = wait.until(
                EC.presence_of_element_located((By.XPATH, SECURITY_WAIT_FORM_XPATH))
            )
            
            # Parse tables
            tables = form_element.find_elements(By.TAG_NAME, "table")
            
            for table in tables:
                rows = table.find_elements(By.TAG_NAME, "tr")
                
                # Extract headers
                headers = []
                for row in rows:
                    header_cells = row.find_elements(By.TAG_NAME, "th")
                    if header_cells:
                        headers = [h.text.strip() for h in header_cells]
                        break
                
                # Extract data rows
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if cells and len(cells) >= 2:
                        terminal_data = {
                            'terminal': cells[0].text.strip(),
                            'wait_time': cells[1].text.strip()
                        }
                        # Add TSA Pre time if present (optional)
                        if len(cells) > 2:
                            terminal_data['tsa_pre_wait_time'] = cells[2].text.strip()
                        
                        collected_data['terminals'].append(terminal_data)
            
            print(f"✓ Data collection complete: {len(collected_data['terminals'])} terminals")
            return collected_data
            
        except Exception as e:
            print(f"✗ Data collection failed: {str(e)}")
            return None
        finally:
            if self.driver:
                self.driver.quit()
    
    def save_to_csv(self, data: dict, filename: str = None):
        """
        Save collected data to a CSV file.
        
        Args:
            data: Collected data.
            filename: File name to save (default: timestamp.csv).
        """
        if not data or not data.get('terminals'):
            print("No data to save.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"security_wait_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Write CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['timestamp', 'terminal', 'wait_time', 'tsa_pre_wait_time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for terminal_data in data['terminals']:
                row = {
                    'timestamp': data['timestamp'],
                    'terminal': terminal_data['terminal'],
                    'wait_time': terminal_data['wait_time'],
                    'tsa_pre_wait_time': terminal_data.get('tsa_pre_wait_time', '')
                }
                writer.writerow(row)
        
        print(f"✓ Data saved successfully: {filepath}")
    
    def collect_continuous(self, interval_minutes: int = 15, duration_hours: int = 24):
        """
        Collect data continuously.
        
        Args:
            interval_minutes: Collection interval in minutes.
            duration_hours: Total collection duration in hours.
        """
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        # Save into one consolidated CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"continuous_data_{timestamp}.csv"
        
        print(f"⏰ Starting data collection for {duration_hours} hours at {interval_minutes}-minute intervals...")
        
        iteration = 0
        while time.time() < end_time:
            iteration += 1
            print(f"\n[Iteration {iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            data = self.collect_security_wait_time()
            if data:
                self.save_to_csv(data, filename)
            
            # Wait until the next collection cycle
            if time.time() < end_time:
                sleep_time = interval_minutes * 60
                print(f"⏳ Waiting for {interval_minutes} minutes...")
                time.sleep(sleep_time)


def main():
    """Main entry point."""
    collector = JFKDataCollector()
    
    # Single collection example
    print("=== JFK Security Wait Time Data Collection ===\n")
    data = collector.collect_security_wait_time()
    if data:
        collector.save_to_csv(data)
    
    # Continuous collection (uncomment to use)
    # collector.collect_continuous(interval_minutes=15, duration_hours=24)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted.")
