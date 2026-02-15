"""
JFK gate walking time information.
Average walking time from security checkpoint to gate.
"""
from typing import Optional
import re


# Gate walk times by JFK terminal (minutes)
# Source: official JFK Airport website
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
    Calculate gate walking time.
    
    Args:
        terminal: Terminal name (e.g., 'Terminal 4', 'T4', '4')
        gate: Gate identifier (e.g., 'B42', '42', 'A10')
        
    Returns:
        Estimated walking time (minutes)
    """
    # Normalize terminal name
    terminal_normalized = _normalize_terminal_name(terminal)
    
    if not terminal_normalized or terminal_normalized not in GATE_WALK_TIMES:
        # Unknown terminal - default 15 minutes
        return 15
    
    terminal_data = GATE_WALK_TIMES[terminal_normalized]
    
    # Use terminal average when gate info is unavailable
    if not gate or gate == 'N/A':
        return _get_terminal_average(terminal_data)
    
    # Parse gate number
    gate_num = _parse_gate_number(gate)
    if gate_num is None:
        return _get_terminal_average(terminal_data)
    
    # Match against gate ranges
    for gate_range, (min_time, max_time) in terminal_data.items():
        if _is_gate_in_range(gate, gate_num, gate_range):
            # Return average value (rounded)
            return round((min_time + max_time) / 2)
    
    # Match failed - use terminal average
    return _get_terminal_average(terminal_data)


def _normalize_terminal_name(terminal: str) -> Optional[str]:
    """Normalize terminal name."""
    if not terminal:
        return None
    
    # Normalize forms like "Terminal 4", "T4", and "4" to "Terminal 4"
    match = re.search(r'(\d+)', str(terminal))
    if match:
        term_num = match.group(1)
        return f"Terminal {term_num}"
    
    return None


def _parse_gate_number(gate: str) -> Optional[int]:
    """Extract the numeric part from gate string (B42 -> 42, A10 -> 10)."""
    match = re.search(r'(\d+)', str(gate))
    return int(match.group(1)) if match else None


def _is_gate_in_range(gate: str, gate_num: int, gate_range: str) -> bool:
    """Check whether gate is within the specified range."""
    gate = gate.upper()
    gate_range = gate_range.upper()
    
    # "All Gates"
    if "ALL" in gate_range:
        return True
    
    # Parse range
    # "Gates A3-B25" -> from A3 to B25
    # "Gates 1-4, 40-47" -> 1-4 or 40-47
    # "Gates 1-11, 18-27" -> 1-11 or 18-27
    
    ranges = gate_range.replace("GATES", "").replace("GATE", "").strip().split(",")
    
    for range_part in ranges:
        range_part = range_part.strip()
        
        # Single range: "A3-B25" or "1-4"
        if "-" in range_part:
            parts = range_part.split("-")
            if len(parts) == 2:
                start_str, end_str = parts[0].strip(), parts[1].strip()
                
                # Extract digits only
                start_num = _parse_gate_number(start_str)
                end_num = _parse_gate_number(end_str)
                
                if start_num and end_num:
                    # Check numeric range
                    if start_num <= gate_num <= end_num:
                        # Check letter prefix (A, B, etc.)
                        gate_prefix = re.match(r'([A-Z]+)', gate)
                        start_prefix = re.match(r'([A-Z]+)', start_str)
                        end_prefix = re.match(r'([A-Z]+)', end_str)
                        
                        # If prefix exists, verify prefix match
                        if gate_prefix and (start_prefix or end_prefix):
                            gate_letter = gate_prefix.group(1)
                            # Check whether it falls in range
                            if start_prefix and gate_letter >= start_prefix.group(1):
                                if end_prefix and gate_letter <= end_prefix.group(1):
                                    return True
                                elif not end_prefix:
                                    return True
                        else:
                            # If no prefix, compare numbers only
                            return True
    
    return False


def _get_terminal_average(terminal_data: dict) -> int:
    """Return average walking time for terminal."""
    all_times = []
    for min_time, max_time in terminal_data.values():
        all_times.append((min_time + max_time) / 2)
    
    return round(sum(all_times) / len(all_times)) if all_times else 15


if __name__ == '__main__':
    # Test
    print("=== Gate Walk Time Test ===\n")
    
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
        ('T4', 'B42'),  # Abbreviated terminal name
        ('4', 'B42'),   # Numeric form only
        ('Terminal 4', None),  # No gate info
        ('Unknown', 'A1'),  # Unknown terminal
    ]
    
    for terminal, gate in test_cases:
        walk_time = get_gate_walk_time(terminal, gate)
        print(f"{terminal:15} Gate {gate if gate else 'N/A':8} -> {walk_time:2} min")
