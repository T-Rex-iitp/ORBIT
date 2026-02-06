"""
TSA 보안검색 대기시간 실시간 크롤링 및 통계 기반 예측
"""
from datetime import datetime
from typing import Dict, Optional
import re


# 공항별 통계 대기시간
TSA_WAIT_TIMES = {
    'JFK': {
        'peak': 45,
        'normal': 25,
        'off_peak': 15,
        'holiday_peak': 35
    },
    'LAX': {
        'peak': 40,
        'normal': 20,
        'off_peak': 12,
        'holiday_peak': 30
    },
    'DEFAULT': {
        'peak': 35,
        'normal': 20,
        'off_peak': 10,
        'holiday_peak': 28
    }
}


class TSAWaitTime:
    """TSA 보안검색 대기시간 조회"""
    
    def __init__(self, use_live_data: bool = True):
        self.use_live_data = use_live_data
    
    def _crawl_jfk_tsa(self) -> Optional[Dict[str, int]]:
        """JFK 실시간 TSA 대기시간 크롤링"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            import os
            
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            
            # Chrome 경로
            chrome_bin = os.path.expanduser("~/chrome_install/opt/google/chrome/chrome")
            driver_path = os.path.expanduser("~/chrome_install/chromedriver-linux64/chromedriver")
            
            if os.path.exists(chrome_bin):
                options.binary_location = chrome_bin
            
            service = Service(driver_path) if os.path.exists(driver_path) else Service()
            driver = webdriver.Chrome(service=service, options=options)
            
            try:
                driver.get("https://www.jfkairport.com")
                wait = WebDriverWait(driver, 10)
                form = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="security-wait"]/form')))
                
                results = {}
                tables = form.find_elements(By.TAG_NAME, "table")
                
                for table in tables:
                    rows = table.find_elements(By.TAG_NAME, "tr")
                    for row in rows:
                        cells = row.find_elements(By.TAG_NAME, "td")
                        if len(cells) >= 4:
                            term = cells[0].text.strip()
                            wait_text = cells[3].text.strip()
                            
                            if "Terminal" in term:
                                nums = re.findall(r'\d+', wait_text)
                                if nums:
                                    results[term] = int(nums[0])
                
                return results if results else None
            finally:
                driver.quit()
        except:
            return None
    
    def get_wait_time(self, airport_code: str, departure_time: datetime) -> Dict:
        """TSA 대기시간 조회"""
        # JFK 실시간 크롤링 시도
        if airport_code == 'JFK' and self.use_live_data:
            live_data = self._crawl_jfk_tsa()
            if live_data:
                avg_wait = int(sum(live_data.values()) / len(live_data))
                hour = departure_time.hour
                dow = departure_time.weekday()
                
                if dow < 5:
                    period = 'peak' if (6 <= hour < 9 or 16 <= hour < 19) else 'normal' if (5 <= hour < 12 or 15 <= hour < 20) else 'off_peak'
                else:
                    period = 'normal' if 9 <= hour < 14 else 'off_peak'
                
                return {
                    'wait_time': avg_wait,
                    'period': period,
                    'source': 'real-time_crawl',
                    'terminals': live_data,
                    'timestamp': datetime.now().isoformat()
                }
        
        # 통계 기반
        stats = TSA_WAIT_TIMES.get(airport_code, TSA_WAIT_TIMES['DEFAULT'])
        hour = departure_time.hour
        dow = departure_time.weekday()
        
        if dow < 5:
            if 6 <= hour < 9 or 16 <= hour < 19:
                period = 'peak'
            elif 5 <= hour < 12 or 15 <= hour < 20:
                period = 'normal'
            else:
                period = 'off_peak'
        else:
            period = 'normal' if 9 <= hour < 14 else 'off_peak'
        
        # 공휴일 체크
        month, day = departure_time.month, departure_time.day
        if self._is_holiday_season(month, day):
            period = 'holiday_peak'
        
        return {
            'wait_time': stats[period],
            'period': period,
            'source': 'historical_average',
            'airport': airport_code
        }
    
    def _is_holiday_season(self, month: int, day: int) -> bool:
        """공휴일 시즌 판단"""
        # 추수감사절 (11월 넷째 목요일 전후)
        if month == 11 and 20 <= day <= 30:
            return True
        # 크리스마스/신년 (12월 20일 - 1월 5일)
        if (month == 12 and day >= 20) or (month == 1 and day <= 5):
            return True
        # 여름 성수기 (6/15 - 8/15)
        if (month == 7) or (month == 6 and day >= 15) or (month == 8 and day <= 15):
            return True
        return False
    
    def get_precheck_wait_time(self, airport_code: str, departure_time: datetime) -> Dict:
        """TSA PreCheck 대기시간 (일반의 35%)"""
        regular = self.get_wait_time(airport_code, departure_time)
        precheck_wait = max(5, int(regular['wait_time'] * 0.35))
        
        return {
            'wait_time': precheck_wait,
            'period': regular['period'],
            'source': regular.get('source', 'historical_average'),
            'airport': airport_code,
            'type': 'TSA_PreCheck'
        }


def get_tsa_wait_time(airport_code: str, departure_time: datetime, 
                      has_precheck: bool = False, use_live_data: bool = True) -> int:
    """편의 함수: TSA 대기시간 반환"""
    tsa = TSAWaitTime(use_live_data=use_live_data)
    
    if has_precheck:
        result = tsa.get_precheck_wait_time(airport_code, departure_time)
    else:
        result = tsa.get_wait_time(airport_code, departure_time)
    
    return result['wait_time']


if __name__ == '__main__':
    print("=== 실시간 크롤링 테스트 ===")
    tsa_live = TSAWaitTime(use_live_data=True)
    dt = datetime.now()
    result = tsa_live.get_wait_time('JFK', dt)
    print(f"JFK 실시간: {result}")
    
    print("\n=== 통계 데이터 테스트 ===")
    tsa_stat = TSAWaitTime(use_live_data=False)
    dt = datetime(2026, 2, 10, 7, 0)
    result = tsa_stat.get_wait_time('JFK', dt)
    print(f"JFK 평일 7am (통계): {result}")
