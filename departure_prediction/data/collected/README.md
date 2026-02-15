# Departure Time Prediction System

This folder stores collected data.

## File Format

Data is saved in the following CSV format:

```csv
timestamp,terminal,wait_time,tsa_pre_wait_time
2026-02-04T12:00:00,Terminal 1,15 mins,5 mins
2026-02-04T12:00:00,Terminal 4,20 mins,8 mins
```

## File Naming Rules

- Single collection: `security_wait_YYYYMMDD_HHMMSS.csv`
- Continuous collection: `continuous_data_YYYYMMDD_HHMMSS.csv`

## How to Collect Data

```bash
# Run from the parent directory
python -m data.data_collector
```

Or

```bash
# Use main.py
python main.py collect --duration 24 --interval 15
```
