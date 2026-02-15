"""
Data preprocessing module.
Transforms collected data into a format suitable for Transformer model training.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List
import os


class DepartureDataPreprocessor:
    """Data preprocessor for departure-time prediction."""
    
    def __init__(self, sequence_length: int = 24):
        """
        Args:
            sequence_length: Time-series sequence length (default: 24 hours).
        """
        self.sequence_length = sequence_length
        
    def parse_wait_time(self, wait_time_str: str) -> int:
        """
        Convert a wait-time string to minutes.
        
        Args:
            wait_time_str: A string such as "15 mins", "1 hour", or "< 5 mins"
            
        Returns:
            int: Wait time in minutes.
        """
        if pd.isna(wait_time_str) or wait_time_str == '':
            return 0
        
        wait_time_str = wait_time_str.lower().strip()
        
        # Handle "< X mins" format
        if '<' in wait_time_str:
            wait_time_str = wait_time_str.replace('<', '').strip()
        
        # Handle hour units
        if 'hour' in wait_time_str:
            try:
                hours = int(''.join(filter(str.isdigit, wait_time_str.split('hour')[0])))
                return hours * 60
            except:
                return 0
        
        # Handle minute units
        elif 'min' in wait_time_str:
            try:
                mins = int(''.join(filter(str.isdigit, wait_time_str.split('min')[0])))
                return mins
            except:
                return 0
        
        # Handle numeric-only values
        try:
            return int(''.join(filter(str.isdigit, wait_time_str)))
        except:
            return 0
    
    def extract_time_features(self, timestamp: pd.Timestamp) -> dict:
        """
        Extract time-related features from a timestamp.
        
        Args:
            timestamp: pandas Timestamp
            
        Returns:
            dict: Time features (hour, day_of_week, is_weekend, etc.)
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
        Load and preprocess a CSV file.
        
        Args:
            csv_path: Path to the CSV file.
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Parse wait times
        df['wait_time_mins'] = df['wait_time'].apply(self.parse_wait_time)
        
        # Parse TSA Pre wait times (if available)
        if 'tsa_pre_wait_time' in df.columns:
            df['tsa_pre_wait_time_mins'] = df['tsa_pre_wait_time'].apply(self.parse_wait_time)
        
        # Extract time features
        time_features = df['timestamp'].apply(self.extract_time_features)
        time_features_df = pd.DataFrame(list(time_features))
        df = pd.concat([df, time_features_df], axis=1)
        
        # Sort
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, 
                        feature_columns: List[str],
                        target_column: str = 'wait_time_mins') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time-series sequence data.
        
        Args:
            df: Preprocessed DataFrame.
            feature_columns: List of input feature columns.
            target_column: Target column for prediction.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (X, y) - input sequences and target values.
        """
        X, y = [], []
        
        for i in range(len(df) - self.sequence_length):
            # Use the previous `sequence_length` hours as input
            sequence = df.iloc[i:i+self.sequence_length][feature_columns].values
            # Use the next time step's wait time as the prediction target
            target = df.iloc[i+self.sequence_length][target_column]
            
            X.append(sequence)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def normalize_features(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Normalize features.
        
        Args:
            X: Input data.
            
        Returns:
            Tuple[np.ndarray, dict]: Normalized data and normalization parameters.
        """
        # Compute per-feature mean and standard deviation
        mean = np.mean(X, axis=(0, 1))
        std = np.std(X, axis=(0, 1))
        std[std == 0] = 1  # Prevent division by zero
        
        # Normalize
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
        Prepare full dataset for training.
        
        Args:
            csv_path: Path to the CSV file.
            train_ratio: Training split ratio.
            val_ratio: Validation split ratio.
            
        Returns:
            dict: Train/validation/test data and metadata.
        """
        # Load and preprocess data
        df = self.load_and_preprocess(csv_path)
        
        # Select feature columns
        feature_columns = [
            'wait_time_mins',
            'hour', 'day_of_week', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        # Add TSA Pre time if available
        if 'tsa_pre_wait_time_mins' in df.columns:
            feature_columns.append('tsa_pre_wait_time_mins')
        
        # Create sequences
        X, y = self.create_sequences(df, feature_columns)
        
        # Normalize
        X_normalized, norm_params = self.normalize_features(X)
        
        # Split data
        n_samples = len(X_normalized)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        X_train = X_normalized[:train_size]
        y_train = y[:train_size]
        
        X_val = X_normalized[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X_normalized[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        print(f"✓ Data preparation complete:")
        print(f"  - Train: {X_train.shape}")
        print(f"  - Validation: {X_val.shape}")
        print(f"  - Test: {X_test.shape}")
        
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
    """Run a quick test."""
    preprocessor = DepartureDataPreprocessor(sequence_length=24)
    
    # Example data path (replace with an actual file path)
    csv_path = "collected/continuous_data_20260204_120000.csv"
    
    if os.path.exists(csv_path):
        data = preprocessor.prepare_for_training(csv_path)
        print("\nData shapes:")
        print(f"Input sequence: {data['X_train'].shape}")
        print(f"Target: {data['y_train'].shape}")
    else:
        print(f"⚠️  File not found: {csv_path}")
        print("Run data_collector.py first to collect data.")


if __name__ == "__main__":
    main()
