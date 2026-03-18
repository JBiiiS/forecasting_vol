import torch
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass 
class BaseConfig:
    # ----------------------------------
    # [1] 공통 환경 설정 (Environment)
    # ----------------------------------
    seed: int = 42
    
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ----------------------------------
    # [2] 데이터 공통 설정 (Data)
    # ----------------------------------
    num_assets: int = 100       # d: 자산 개수
    batch_size: int = 16
    total_samples: int = 20000
    T: float = 30/252           # T: 윈도우 길이   
    steps: int = 30           # N: steps
    rv_window: int = 10
    r: float = 0.0
    num_epochs: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3   # Optimizer 설정
    dropout: float = 0.1

    
    # ----------------------------------
    # [3] 유틸리티 메서드
    # ----------------------------------
    @property # making a function call look like a variable access
    def device(self):
        return torch.device(self.device_name)

    def to_dict(self):
        return asdict(self)
    
    def __post_init__(self):
        self.dt = self.T / self.steps