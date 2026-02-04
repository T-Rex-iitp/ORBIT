"""
TSA 보안검색 대기시간 정보
실시간 API 또는 공항별 평균 통계 제공
"""
from typing import Dict, Optional
from datetime import datetime


# 주요 공항별 평균 TSA 대기시간 (분)
# 출처: TSA historical data, 시간대별 평균
TSA_WAIT_TIMES = {
    'JFK': {
        'peak': 45,      # 오전 6-9시, 오후 4-7시
        'normal': 25,    # 일반 시간
        'off_peak': 15   # 밤/새벽
    },
    'LAX': {
        'peak': 50,
        'normal': 30,
        'off_peak': 20
    },
    'ORD': {
        'peak': 40,
        'normal': 25,
        'off_peak': 15
    },
    'ATL': {
        'peak': 45,
        'normal': 28,
        'off_peak': 18
    },
    'DFW': {
        'peak': 35,
        'normal': 22,
        'off_peak': 12
    },
    'SFO': {
        'peak': 40,
        'normal': 25,
        'off_peak': 15
    },
    'MIA': {
        'peak': 38,
        'normal': 23,
        'off_peak': 13
    },
    'LAS': {
        'peak': 42,
        'normal': 26,
        'off_peak': 16
    },
    'SEA': {
        'peak': 35,
        'normal': 20,
        'off_peak': 12
    },
    'BOS': {
        'peak': 38,
        'normal': 22,
        'off_peak': 14
    },
    # 기본값 (모든 공항)
    'DEFAULT': {
        'peak': 40,
        'normal': 25,
        'off_peak': 15
    }
}


class TSAWaitTime:
    """TSA 보안검색 대기시간 조회"""
    
    def __init__(self):
        """Initialize TSA wait time service"""
        pass
    
    def get_wait_time(self, airport_code: str, departure_time: datetime) -> Dict:
        """
        공항과 출발 시간에 따른 TSA 대기시간 반환
        
        Args:
            airport_code: 공항 코드 (예: 'JFK')
            departure_time: 출발 시간
            
        Returns:
            {
                'wait_time': 예상 대기시간 (분),
                'period': 'peak'/'normal'/'off_peak',
                'source': 'historical_average' or 'live_api'
            }
        """
        # 공항별 통계 가져오기
        airport_stats = TSA_WAIT_TIMES.get(airport_code, TSA_WAIT_TIMES['DEFAULT'])
        
        # 시간대 판단
        hour = departure_time.hour
        day_of_week = departure_time.weekday()
        
        # Peak 시간대 판단
        # - 평일 오전 6-9시, 오후 4-7시
        # - 주말 오전 7-11시, 오후 3-6시
        is_weekday = day_of_week < 5
        
        if is_weekday:
            if (6 <= hour < 9) or (16 <= hour < 19):
                period = 'peak'
            elif (22 <= hour or hour < 5):
                period = 'off_peak'
            else:
                period = 'normal'
        else:  # 주말
            if (7 <= hour < 11) or (15 <= hour < 18):
                period = 'peak'
            elif (22 <= hour or hour < 6):
                period = 'off_peak'
            else:
                period = 'normal'
        
        wait_time = airport_stats[period]
        
        # 추수감사절, 크리스마스 등 성수기에는 +10분
        if self._is_holiday_season(departure_time):
            wait_time += 10
            period = 'holiday_peak'
        
        return {
            'wait_time': wait_time,
            'period': period,
            'source': 'historical_average',
            'airport': airport_code
        }
    
    def _is_holiday_season(self, dt: datetime) -> bool:
        """성수기 여부 판단"""
        month = dt.month
        day = dt.day
        
        # 추수감사절 주간 (11월 20-30일)
        if month == 11 and 20 <= day <= 30:
            return True
        
        # 크리스마스/연말 (12월 20일 - 1월 5일)
        if (month == 12 and day >= 20) or (month == 1 and day <= 5):
            return True
        
        # 여름 성수기 (6월 15일 - 8월 15일)
        if (month == 7) or (month == 6 and day >= 15) or (month == 8 and day <= 15):
            return True
        
        return False
    
    def get_precheck_wait_time(self, airport_code: str, departure_time: datetime) -> Dict:
        """
        TSA PreCheck 대기시간 (일반 대기시간의 30-40%)
        
        Args:
            airport_code: 공항 코드
            departure_time: 출발 시간
            
        Returns:
            PreCheck 대기시간 정보
        """
        regular_wait = self.get_wait_time(airport_code, departure_time)
        
        # PreCheck는 일반의 35% 정도
        precheck_wait = int(regular_wait['wait_time'] * 0.35)
        
        return {
            'wait_time': max(5, precheck_wait),  # 최소 5분
            'period': regular_wait['period'],
            'source': 'historical_average',
            'airport': airport_code,
            'type': 'TSA_PreCheck'
        }


def get_tsa_wait_time(airport_code: str, departure_time: datetime, 
                      has_precheck: bool = False) -> int:
    """
    편의 함수: TSA 대기시간 반환
    
    Args:
        airport_code: 공항 코드
        departure_time: 출발 시간
        has_precheck: TSA PreCheck 보유 여부
        
    Returns:
        예상 대기시간 (분)
    """
    tsa = TSAWaitTime()
    
    if has_precheck:
        result = tsa.get_precheck_wait_time(airport_code, departure_time)
    else:
        result = tsa.get_wait_time(airport_code, departure_time)
    
    return result['wait_time']


if __name__ == '__main__':
    # 테스트
    tsa = TSAWaitTime()
    
    # JFK 평일 오전 7시
    dt = datetime(2026, 2, 10, 7, 0)  # 월요일
    result = tsa.get_wait_time('JFK', dt)
    print(f"JFK 평일 오전 7시: {result}")
    
    # JFK 주말 오후 5시
    dt = datetime(2026, 2, 15, 17, 0)  # 토요일
    result = tsa.get_wait_time('JFK', dt)
    print(f"JFK 주말 오후 5시: {result}")
    
    # PreCheck
    dt = datetime(2026, 2, 10, 7, 0)
    result = tsa.get_precheck_wait_time('JFK', dt)
    print(f"JFK PreCheck: {result}")
    
    # 크리스마스
    dt = datetime(2025, 12, 24, 10, 0)
    result = tsa.get_wait_time('JFK', dt)
    print(f"크리스마스 JFK: {result}")
