"""
Transformer 기반 출발 시간 예측 모델
시계열 데이터를 입력받아 미래의 대기 시간을 예측
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """위치 인코딩 레이어"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: 임베딩 차원
            max_len: 최대 시퀀스 길이
            dropout: 드롭아웃 비율
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 인코딩 계산
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerTimeSeriesModel(nn.Module):
    """Transformer 기반 시계열 예측 모델"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        """
        Args:
            input_dim: 입력 특성 차원
            d_model: Transformer 임베딩 차원
            nhead: 멀티헤드 어텐션 헤드 수
            num_encoder_layers: 인코더 레이어 수
            dim_feedforward: 피드포워드 네트워크 차원
            dropout: 드롭아웃 비율
            max_seq_length: 최대 시퀀스 길이
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 입력 프로젝션
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # [seq_len, batch, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: [batch_size, 1] - 예측된 대기 시간
        """
        # [batch, seq, features] -> [seq, batch, features]
        x = x.transpose(0, 1)
        
        # 입력 프로젝션
        x = self.input_projection(x)  # [seq, batch, d_model]
        
        # 위치 인코딩 추가
        x = self.pos_encoder(x)
        
        # Transformer 인코더
        encoded = self.transformer_encoder(x)  # [seq, batch, d_model]
        
        # 마지막 타임스텝의 출력만 사용
        last_output = encoded[-1, :, :]  # [batch, d_model]
        
        # 최종 예측
        output = self.output_layer(last_output)  # [batch, 1]
        
        return output


class LSTMBaselineModel(nn.Module):
    """비교를 위한 LSTM 베이스라인 모델"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 입력 특성 차원
            hidden_dim: LSTM 은닉 차원
            num_layers: LSTM 레이어 수
            dropout: 드롭아웃 비율
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: [batch_size, 1] - 예측된 대기 시간
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 마지막 타임스텝의 출력 사용
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # 최종 예측
        output = self.output_layer(last_output)  # [batch, 1]
        
        return output


def create_model(model_type: str = 'transformer', input_dim: int = 8, **kwargs) -> nn.Module:
    """
    모델 생성 팩토리 함수
    
    Args:
        model_type: 모델 타입 ('transformer' 또는 'lstm')
        input_dim: 입력 특성 차원
        **kwargs: 모델별 추가 파라미터
        
    Returns:
        nn.Module: 생성된 모델
    """
    if model_type == 'transformer':
        return TransformerTimeSeriesModel(input_dim=input_dim, **kwargs)
    elif model_type == 'lstm':
        return LSTMBaselineModel(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_model():
    """모델 테스트"""
    # 테스트 데이터
    batch_size = 16
    seq_len = 24
    input_dim = 8
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("=== Transformer 모델 테스트 ===")
    transformer_model = create_model('transformer', input_dim=input_dim)
    transformer_output = transformer_model(x)
    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {transformer_output.shape}")
    print(f"파라미터 수: {sum(p.numel() for p in transformer_model.parameters()):,}")
    
    print("\n=== LSTM 모델 테스트 ===")
    lstm_model = create_model('lstm', input_dim=input_dim)
    lstm_output = lstm_model(x)
    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {lstm_output.shape}")
    print(f"파라미터 수: {sum(p.numel() for p in lstm_model.parameters()):,}")


if __name__ == "__main__":
    test_model()
