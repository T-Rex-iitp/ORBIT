# 출발 시간 예측 시스템

이 폴더는 수집된 데이터를 저장하는 공간입니다.

## 파일 형식

데이터는 다음과 같은 CSV 형식으로 저장됩니다:

```csv
timestamp,terminal,wait_time,tsa_pre_wait_time
2026-02-04T12:00:00,Terminal 1,15 mins,5 mins
2026-02-04T12:00:00,Terminal 4,20 mins,8 mins
```

## 파일 명명 규칙

- 단일 수집: `security_wait_YYYYMMDD_HHMMSS.csv`
- 지속적 수집: `continuous_data_YYYYMMDD_HHMMSS.csv`

## 데이터 수집 방법

```bash
# 상위 디렉토리에서 실행
python -m data.data_collector
```

또는

```bash
# main.py 사용
python main.py collect --duration 24 --interval 15
```
