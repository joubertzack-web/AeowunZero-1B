from dataclasses import dataclass

@dataclass
class AEOWUNConfig:
    vocab_size: int = 8192
    max_seq_len: int = 512
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 8
    multiple_of: int = 32  # For SwiGLU FFN hidden dimension
    norm_eps: float = 1e-5
    
    # Training params
    batch_size: int = 4
    learning_rate: float = 0.02 # As recommended for Muon
    weight_decay: float = 0.01
    device: str = "cuda"
