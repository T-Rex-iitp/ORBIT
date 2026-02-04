"""
데이터 수집 모듈
JFK 공항의 보안 대기 시간 및 항공편 정보를 수집하여 CSV로 저장
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
    """JFK 공항 데이터 수집 클래스"""
    
    def __init__(self, output_dir: str = "data/collected"):
        """
        Args:
            output_dir: 수집된 데이터를 저장할 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.driver = None
        
    def create_driver(self, headless: bool = True) -> webdriver.Chrome:
        """Chrome WebDriver 생성"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # ChromeDriver 경로 설정
        crawl_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "crawl")
        driver_path = os.path.join(crawl_dir, "chromedriver-mac-x64", "chromedriver")
        
        if os.path.exists(driver_path):
            service = Service(executable_path=driver_path)
            return webdriver.Chrome(service=service, options=chrome_options)
        else:
            # ChromeDriver가 PATH에 있는 경우
            return webdriver.Chrome(options=chrome_options)
    
    def collect_security_wait_time(self) -> dict:
        """
        보안 대기 시간 데이터 수집
        
        Returns:
            dict: 수집된 데이터 (timestamp, terminal, wait_time 등)
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
            
            # 테이블 파싱
            tables = form_element.find_elements(By.TAG_NAME, "table")
            
            for table in tables:
                rows = table.find_elements(By.TAG_NAME, "tr")
                
                # 헤더 추출
                headers = []
                for row in rows:
                    header_cells = row.find_elements(By.TAG_NAME, "th")
                    if header_cells:
                        headers = [h.text.strip() for h in header_cells]
                        break
                
                # 데이터 행 추출
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if cells and len(cells) >= 2:
                        terminal_data = {
                            'terminal': cells[0].text.strip(),
                            'wait_time': cells[1].text.strip()
                        }
                        # TSA Pre 시간이 있으면 추가 (선택적)
                        if len(cells) > 2:
                            terminal_data['tsa_pre_wait_time'] = cells[2].text.strip()
                        
                        collected_data['terminals'].append(terminal_data)
            
            print(f"✓ 데이터 수집 완료: {len(collected_data['terminals'])}개 터미널")
            return collected_data
            
        except Exception as e:
            print(f"✗ 데이터 수집 실패: {str(e)}")
            return None
        finally:
            if self.driver:
                self.driver.quit()
    
    def save_to_csv(self, data: dict, filename: str = None):
        """
        수집된 데이터를 CSV 파일로 저장
        
        Args:
            data: 수집된 데이터
            filename: 저장할 파일명 (기본값: timestamp.csv)
        """
        if not data or not data.get('terminals'):
            print("저장할 데이터가 없습니다.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"security_wait_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # CSV 작성
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
        
        print(f"✓ 데이터 저장 완료: {filepath}")
    
    def collect_continuous(self, interval_minutes: int = 15, duration_hours: int = 24):
        """
        지속적으로 데이터 수집
        
        Args:
            interval_minutes: 수집 간격 (분)
            duration_hours: 총 수집 시간 (시간)
        """
        start_time = time.time()
        end_time = start_time + (duration_hours * 3600)
        
        # 하나의 통합 CSV 파일에 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"continuous_data_{timestamp}.csv"
        
        print(f"⏰ {duration_hours}시간 동안 {interval_minutes}분 간격으로 데이터 수집 시작...")
        
        iteration = 0
        while time.time() < end_time:
            iteration += 1
            print(f"\n[반복 {iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            data = self.collect_security_wait_time()
            if data:
                self.save_to_csv(data, filename)
            
            # 다음 수집까지 대기
            if time.time() < end_time:
                sleep_time = interval_minutes * 60
                print(f"⏳ {interval_minutes}분 대기 중...")
                time.sleep(sleep_time)


def main():
    """메인 실행 함수"""
    collector = JFKDataCollector()
    
    # 단일 수집 예제
    print("=== JFK 보안 대기 시간 데이터 수집 ===\n")
    data = collector.collect_security_wait_time()
    if data:
        collector.save_to_csv(data)
    
    # 지속적 수집 (주석 해제하여 사용)
    # collector.collect_continuous(interval_minutes=15, duration_hours=24)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
