import torch
from dataclasses import dataclass
from dlqf_rnn.dlqf_rnn_config import DLQFRNNConfig


@dataclass
class DLQFRNNWithODEConfig(DLQFRNNConfig):
    # ----------------------------------
    # Neural SDE Settings
    # ----------------------------------


    ode_hidden_dim: int = 16

    exp_deno_init: float = 8.5
    
    output_dim: int = 1      

    lambda_gan: float = 1.0
    lambda_l2: float = 0.5

    w_coef: float = 50.0
    l2_coef: float = 1.0

    only_d_epoch: int = 20


    def __post_init__(self):
        super().__post_init__()
        self.lstm_hidden_dim: int = self.hidden_dim
    
        self.ode_times = torch.linspace(0, 1, self.total_quantile).to(self.device)


        