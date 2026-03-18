import torch
import torch.nn as nn
import torchcde 
from dlqf_rnn.dlqf_rnn_with_ode_config import DLQFRNNWithODEConfig


class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)



# =============================================================================
# [1] CDE Vector Field  f_φ(t, h)
# =============================================================================

class CDEFunc(nn.Module):

    def __init__(self, config: DLQFRNNWithODEConfig):
        super().__init__()
        self.cde_hidden_dim = config.ode_hidden_dim
        self.output_dim     = config.output_dim

        self.net = nn.Sequential(
            nn.Linear(self.cde_hidden_dim + 1, self.cde_hidden_dim),  # +1 for time
            LipSwish(),
            nn.Linear(self.cde_hidden_dim, self.cde_hidden_dim),
            LipSwish(),
            nn.Linear(self.cde_hidden_dim, self.cde_hidden_dim * self.output_dim),
            nn.Tanh()
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, h):
       
        t_batch = torch.full((h.size(0), 1), float(t.detach()), device=h.device, dtype=h.dtype)
        th  = torch.cat([t_batch, h], dim=-1)               
        out = self.net(th)                                  
        return out.view(h.size(0), self.cde_hidden_dim, self.output_dim)
    
# =============================================================================
# [2] CDEDiscriminator — full Neural CDE discriminator
# =============================================================================

class CDEDiscriminator(nn.Module):

    def __init__(self, config:DLQFRNNWithODEConfig):
        super().__init__()
        self.config = config
        self.cde_hidden_dim = config.ode_hidden_dim
        self.cde_func  = CDEFunc(config)

        # Initial hidden state — linear projection of the first observation
        self.h0_linear = nn.Sequential(
            nn.Linear(config.output_dim, self.cde_hidden_dim),
            LipSwish(),
            nn.Linear(self.cde_hidden_dim, self.cde_hidden_dim),
            LipSwish(),
            nn.Linear(self.cde_hidden_dim, self.cde_hidden_dim)
        )

        for m in self.h0_linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)

        # Final readout — no activation
        self.readout   = nn.Linear(self.cde_hidden_dim, 1)
        for m in self.readout.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, val=0)
   

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
       
        # ------------------------------------------------------------------
        # [1] Build continuous interpolation from discrete path
        # ------------------------------------------------------------------
        coeffs = torchcde.linear_interpolation_coeffs(x, t=times)
        interpolated_x = torchcde.LinearInterpolation(coeffs, t=times)

        # ------------------------------------------------------------------
        # [2] Initial hidden state from first observation
        # -------------------------------------------------
        x0 = interpolated_x.evaluate(interpolated_x.interval[0]) 
        h0 = self.h0_linear(x0)                # (B, cde_hidden_dim)

        adjoint_params = tuple(self.cde_func.parameters()) + (coeffs,)

        # ------------------------------------------------------------------
        # [3] Integrate CDE 
        # ------------------------------------------------------------------
        h = torchcde.cdeint(
            X       = interpolated_x,
            func    = self.cde_func,
            z0      = h0,
            t       = interpolated_x.interval,
            method  = 'dopri5',    
            adjoint = True,        
            adjoint_params = adjoint_params, 
            atol    = 1e-4,      
            rtol    = 1e-4       
        )

        # ------------------------------------------------------------------
        # [4] Readout from final hidden state h(T)
        # ------------------------------------------------------------------
        h_T   = h[:, -1, :]                            # (B, cde_hidden_dim)
        score = self.readout(h_T)                      # (B, 1)
        
        return score

