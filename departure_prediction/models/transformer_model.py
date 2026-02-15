"""
Transformer-based departure-time prediction model.
Takes time-series data as input and predicts future wait time.
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding layer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout ratio.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encoding
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
    """Transformer-based time-series prediction model."""
    
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
            input_dim: Input feature dimension.
            d_model: Transformer embedding dimension.
            nhead: Number of multi-head attention heads.
            num_encoder_layers: Number of encoder layers.
            dim_feedforward: Feed-forward network dimension.
            dropout: Dropout ratio.
            max_seq_length: Maximum sequence length.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
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
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: [batch_size, 1] - predicted wait time.
        """
        # [batch, seq, features] -> [seq, batch, features]
        x = x.transpose(0, 1)
        
        # Input projection
        x = self.input_projection(x)  # [seq, batch, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # [seq, batch, d_model]
        
        # Use only the output of the last time step
        last_output = encoded[-1, :, :]  # [batch, d_model]
        
        # Final prediction
        output = self.output_layer(last_output)  # [batch, 1]
        
        return output


class LSTMBaselineModel(nn.Module):
    """LSTM baseline model for comparison."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension.
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout ratio.
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
        
        # Output layer
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
            torch.Tensor: [batch_size, 1] - predicted wait time.
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use output from the last time step
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # Final prediction
        output = self.output_layer(last_output)  # [batch, 1]
        
        return output


def create_model(model_type: str = 'transformer', input_dim: int = 8, **kwargs) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_type: Model type ('transformer' or 'lstm').
        input_dim: Input feature dimension.
        **kwargs: Additional model-specific parameters.
        
    Returns:
        nn.Module: Created model.
    """
    if model_type == 'transformer':
        return TransformerTimeSeriesModel(input_dim=input_dim, **kwargs)
    elif model_type == 'lstm':
        return LSTMBaselineModel(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_model():
    """Test model outputs."""
    # Test data
    batch_size = 16
    seq_len = 24
    input_dim = 8
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("=== Transformer Model Test ===")
    transformer_model = create_model('transformer', input_dim=input_dim)
    transformer_output = transformer_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {transformer_output.shape}")
    print(f"Parameter count: {sum(p.numel() for p in transformer_model.parameters()):,}")
    
    print("\n=== LSTM Model Test ===")
    lstm_model = create_model('lstm', input_dim=input_dim)
    lstm_output = lstm_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {lstm_output.shape}")
    print(f"Parameter count: {sum(p.numel() for p in lstm_model.parameters()):,}")


if __name__ == "__main__":
    test_model()
