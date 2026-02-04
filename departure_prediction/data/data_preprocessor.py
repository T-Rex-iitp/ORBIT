"""
데이터 전처리 모듈
수집된 데이터를 Transformer 모델 학습에 적합한 형태로 변환
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List
import os


class DepartureDataPreprocessor:
    """출발 시간 예측을 위한 데이터 전처리 클래스"""
    
    def __init__(self, sequence_length: int = 24):
        """
        Args:
            sequence_length: 시계열 시퀀스 길이 (기본 24시간)
        """
        self.sequence_length = sequence_length
        
    def parse_wait_time(self, wait_time_str: str) -> int:
        """
        대기 시간 문자열을 분(minute)으로 변환
        
        Args:
            wait_time_str: "15 mins", "1 hour", "< 5 mins" 등의 문자열
            
        Returns:
            int: 대기 시간 (분)
        """
        if pd.isna(wait_time_str) or wait_time_str == '':
            return 0
        
        wait_time_str = wait_time_str.lower().strip()
        
        # "< X mins" 형태 처리
        if '<' in wait_time_str:
            wait_time_str = wait_time_str.replace('<', '').strip()
        
        # 시간(hour) 단위 처리
        if 'hour' in wait_time_str:
            try:
                hours = int(''.join(filter(str.isdigit, wait_time_str.split('hour')[0])))
                return hours * 60
            except:
                return 0
        
        # 분(mins) 단위 처리
        elif 'min' in wait_time_str:
            try:
                mins = int(''.join(filter(str.isdigit, wait_time_str.split('min')[0])))
                return mins
            except:
                return 0
        
        # 숫자만 있는 경우
        try:
            return int(''.join(filter(str.isdigit, wait_time_str)))
        except:
            return 0
    
    def extract_time_features(self, timestamp: pd.Timestamp) -> dict:
        """
        타임스탬프에서 시간 관련 특성 추출
        
        Args:
            timestamp: pandas Timestamp
            
        Returns:
            dict: 시간 특성 (hour, day_of_week, is_weekend, etc.)
        """
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.dayofweek,  # 0=Monday, 6=Sunday
            'is_weekend': 1 if timestamp.dayofweek >= 5 else 0,
            'day': timestamp.day,
            'month': timestamp.month,
            'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
            'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
            'day_sin': np.sin(2 * np.pi * timestamp.dayofweek / 7),
            'day_cos': np.cos(2 * np.pi * timestamp.dayofweek / 7),
        }
    
    def load_and_preprocess(self, csv_path: str) -> pd.DataFrame:
        """
        CSV 파일을 로드하고 전처리
        
        Args:
            csv_path: CSV 파일 경로
            
        Returns:
            pd.DataFrame: 전처리된 데이터프레임
        """
        # CSV 로드
        df = pd.read_csv(csv_path)
        
        # 타임스탬프 파싱
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 대기 시간 파싱
        df['wait_time_mins'] = df['wait_time'].apply(self.parse_wait_time)
        
        # TSA Pre 대기 시간 파싱 (있는 경우)
        if 'tsa_pre_wait_time' in df.columns:
            df['tsa_pre_wait_time_mins'] = df['tsa_pre_wait_time'].apply(self.parse_wait_time)
        
        # 시간 특성 추출
        time_features = df['timestamp'].apply(self.extract_time_features)
        time_features_df = pd.DataFrame(list(time_features))
        df = pd.concat([df, time_features_df], axis=1)
        
        # 정렬
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, 
                        feature_columns: List[str],
                        target_column: str = 'wait_time_mins') -> Tuple[np.ndarray, np.ndarray]:
        """
        시계열 시퀀스 데이터 생성
        
        Args:
            df: 전처리된 데이터프레임
            feature_columns: 입력 특성 컬럼 리스트
            target_column: 예측 대상 컬럼
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) - 입력 시퀀스, 타겟 값
        """
        X, y = [], []
        
        for i in range(len(df) - self.sequence_length):
            # 과거 sequence_length 시간의 데이터를 입력으로 사용
            sequence = df.iloc[i:i+self.sequence_length][feature_columns].values
            # 다음 시점의 대기 시간을 예측 타겟으로 사용
            target = df.iloc[i+self.sequence_length][target_column]
            
            X.append(sequence)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def normalize_features(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        특성 정규화
        
        Args:
            X: 입력 데이터
            
        Returns:
            Tuple[np.ndarray, dict]: 정규화된 데이터, 정규화 파라미터
        """
        # 각 특성별 평균과 표준편차 계산
        mean = np.mean(X, axis=(0, 1))
        std = np.std(X, axis=(0, 1))
        std[std == 0] = 1  # 0으로 나누기 방지
        
        # 정규화
        X_normalized = (X - mean) / std
        
        normalization_params = {
            'mean': mean,
            'std': std
        }
        
        return X_normalized, normalization_params
    
    def prepare_for_training(self, csv_path: str, 
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15) -> dict:
        """
        학습을 위한 전체 데이터 준비
        
        Args:
            csv_path: CSV 파일 경로
            train_ratio: 학습 데이터 비율
            val_ratio: 검증 데이터 비율
            
        Returns:
            dict: 학습/검증/테스트 데이터 및 메타정보
        """
        # 데이터 로드 및 전처리
        df = self.load_and_preprocess(csv_path)
        
        # 특성 컬럼 선택
        feature_columns = [
            'wait_time_mins',
            'hour', 'day_of_week', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        # TSA Pre 시간이 있으면 추가
        if 'tsa_pre_wait_time_mins' in df.columns:
            feature_columns.append('tsa_pre_wait_time_mins')
        
        # 시퀀스 생성
        X, y = self.create_sequences(df, feature_columns)
        
        # 정규화
        X_normalized, norm_params = self.normalize_features(X)
        
        # 데이터 분할
        n_samples = len(X_normalized)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        X_train = X_normalized[:train_size]
        y_train = y[:train_size]
        
        X_val = X_normalized[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X_normalized[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        print(f"✓ 데이터 준비 완료:")
        print(f"  - 학습: {X_train.shape}")
        print(f"  - 검증: {X_val.shape}")
        print(f"  - 테스트: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'normalization_params': norm_params,
            'feature_columns': feature_columns,
            'sequence_length': self.sequence_length
        }


def main():
    """테스트 실행"""
    preprocessor = DepartureDataPreprocessor(sequence_length=24)
    
    # 예제 데이터 경로 (실제 파일 경로로 변경 필요)
    csv_path = "collected/continuous_data_20260204_120000.csv"
    
    if os.path.exists(csv_path):
        data = preprocessor.prepare_for_training(csv_path)
        print("\n데이터 형태:")
        print(f"입력 시퀀스: {data['X_train'].shape}")
        print(f"타겟: {data['y_train'].shape}")
    else:
        print(f"⚠️  파일을 찾을 수 없습니다: {csv_path}")
        print("먼저 data_collector.py를 실행하여 데이터를 수집하세요.")


if __name__ == "__main__":
    main()
