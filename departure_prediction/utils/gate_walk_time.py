"""
JFK 공항 게이트 이동 시간 정보
보안검색대에서 게이트까지의 평균 도보 시간
"""
from typing import Optional
import re


# JFK 터미널별 게이트 이동 시간 (분)
# 출처: JFK Airport 공식 웹사이트
GATE_WALK_TIMES = {
    'Terminal 1': {
        'Gate 1-4': (1, 2),
        'Gate 5-11': (3, 4),
    },
    'Terminal 4': {
        'Gates A3-B25': (3, 5),
        'Gates B26-B33': (7, 11),
        'Gates B34-B42': (12, 14),
        'Gates B43-B55': (15, 18),
    },
    'Terminal 5': {
        'Gates 1-11, 18-27': (2, 4),
        'Gates 12-17, 28-30': (5, 7),
    },
    'Terminal 7': {
        'All Gates': (1, 2),
    },
    'Terminal 8': {
        'Gates 5-16': (2, 4),
        'Gates 1-4, 40-47': (5, 8),
        'Gates 31A-39': (8, 11),
    },
}


def get_gate_walk_time(terminal: str, gate: Optional[str] = None) -> int:
    """
    게이트 이동 시간 계산
    
    Args:
        terminal: 터미널 이름 (예: 'Terminal 4', 'T4', '4')
        gate: 게이트 번호 (예: 'B42', '42', 'A10')
        
    Returns:
        예상 이동 시간 (분)
    """
    # 터미널 이름 정규화
    terminal_normalized = _normalize_terminal_name(terminal)
    
    if not terminal_normalized or terminal_normalized not in GATE_WALK_TIMES:
        # 알 수 없는 터미널 - 기본값 15분
        return 15
    
    terminal_data = GATE_WALK_TIMES[terminal_normalized]
    
    # 게이트 정보가 없으면 터미널 평균
    if not gate or gate == 'N/A':
        return _get_terminal_average(terminal_data)
    
    # 게이트 번호 파싱
    gate_num = _parse_gate_number(gate)
    if gate_num is None:
        return _get_terminal_average(terminal_data)
    
    # 게이트 범위에서 매칭
    for gate_range, (min_time, max_time) in terminal_data.items():
        if _is_gate_in_range(gate, gate_num, gate_range):
            # 평균값 반환
            return (min_time + max_time) // 2
    
    # 매칭 실패 - 터미널 평균
    return _get_terminal_average(terminal_data)


def _normalize_terminal_name(terminal: str) -> Optional[str]:
    """터미널 이름 정규화"""
    if not terminal:
        return None
    
    # "Terminal 4", "T4", "4" 등을 "Terminal 4"로 통일
    match = re.search(r'(\d+)', str(terminal))
    if match:
        term_num = match.group(1)
        return f"Terminal {term_num}"
    
    return None


def _parse_gate_number(gate: str) -> Optional[int]:
    """게이트에서 숫자 추출 (B42 -> 42, A10 -> 10)"""
    match = re.search(r'(\d+)', str(gate))
    return int(match.group(1)) if match else None


def _is_gate_in_range(gate: str, gate_num: int, gate_range: str) -> bool:
    """게이트가 범위 내에 있는지 확인"""
    gate = gate.upper()
    gate_range = gate_range.upper()
    
    # "All Gates"
    if "ALL" in gate_range:
        return True
    
    # 범위 파싱
    # "Gates A3-B25" -> A3부터 B25까지
    # "Gates 1-4, 40-47" -> 1-4 또는 40-47
    # "Gates 1-11, 18-27" -> 1-11 또는 18-27
    
    ranges = gate_range.replace("GATES", "").replace("GATE", "").strip().split(",")
    
    for range_part in ranges:
        range_part = range_part.strip()
        
        # 단일 범위 "A3-B25" or "1-4"
        if "-" in range_part:
            parts = range_part.split("-")
            if len(parts) == 2:
                start_str, end_str = parts[0].strip(), parts[1].strip()
                
                # 숫자만 추출
                start_num = _parse_gate_number(start_str)
                end_num = _parse_gate_number(end_str)
                
                if start_num and end_num:
                    # 숫자 범위 체크
                    if start_num <= gate_num <= end_num:
                        # 문자 prefix 체크 (A, B 등)
                        gate_prefix = re.match(r'([A-Z]+)', gate)
                        start_prefix = re.match(r'([A-Z]+)', start_str)
                        end_prefix = re.match(r'([A-Z]+)', end_str)
                        
                        # prefix가 있으면 매칭 확인
                        if gate_prefix and (start_prefix or end_prefix):
                            gate_letter = gate_prefix.group(1)
                            # 범위에 포함되는지 확인
                            if start_prefix and gate_letter >= start_prefix.group(1):
                                if end_prefix and gate_letter <= end_prefix.group(1):
                                    return True
                                elif not end_prefix:
                                    return True
                        else:
                            # prefix 없으면 숫자만 비교
                            return True
    
    return False


def _get_terminal_average(terminal_data: dict) -> int:
    """터미널의 평균 이동 시간"""
    all_times = []
    for min_time, max_time in terminal_data.values():
        all_times.append((min_time + max_time) // 2)
    
    return sum(all_times) // len(all_times) if all_times else 15


if __name__ == '__main__':
    # 테스트
    print("=== 게이트 이동 시간 테스트 ===\n")
    
    test_cases = [
        ('Terminal 4', 'B42'),
        ('Terminal 4', 'B25'),
        ('Terminal 4', 'B55'),
        ('Terminal 4', 'A10'),
        ('Terminal 1', '5'),
        ('Terminal 1', '10'),
        ('Terminal 5', '15'),
        ('Terminal 5', '25'),
        ('Terminal 7', 'A1'),
        ('Terminal 8', '10'),
        ('Terminal 8', '45'),
        ('Terminal 8', '35'),
        ('T4', 'B42'),  # 축약형 터미널명
        ('4', 'B42'),   # 숫자만
        ('Terminal 4', None),  # 게이트 정보 없음
        ('Unknown', 'A1'),  # 알 수 없는 터미널
    ]
    
    for terminal, gate in test_cases:
        walk_time = get_gate_walk_time(terminal, gate)
        print(f"{terminal:15} Gate {gate if gate else 'N/A':8} -> {walk_time:2}분")
