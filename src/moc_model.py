# moc_model.py
"""
The Mixture of Classifiers (MoC) model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MixtureOfClassifiers(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=64, bias=True):
        """
        Args:
            input_dim (int): Dimensionality of input
            num_experts (int): Number of experts in the mixture
            hidden_dim (int): Number of hidden units in the router
            bias (bool): Whether experts have bias terms
        """
        super().__init__()
        self.num_experts = num_experts

        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )

        # Experts
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, 2, bias=bias) for _ in range(num_experts)
        ])

    def gumbel_softmax_sample(self, logits, temperature=2.0, eps=1e-8):
        """
        Sample from a Gumbel-Softmax distribution (a continuous relaxation 
        of sampling a single index).
        """
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)

    def forward(self, x, train_mode=True, temperature=1.0):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)
            train_mode (bool): If True, use Gumbel-Softmax (soft mixture);
                               if False, do hard selection.
            temperature (float): Temperature for Gumbel-Softmax.

        Returns:
            outputs (Tensor): Shape (batch_size, num_classes) = (batch_size, 2)
            selected_experts (Tensor or None): Selected expert indices or None
        """
        routing_logits = self.router(x)  # (batch_size, num_experts)

        if train_mode:
            routing_weights = self.gumbel_softmax_sample(routing_logits, temperature)
            experts_output = torch.stack([expert(x) for expert in self.experts], dim=1)
            routing_weights = routing_weights.unsqueeze(-1)  # (batch_size, num_experts, 1)
            outputs = torch.sum(experts_output * routing_weights, dim=1)
            selected_experts = None
        else:
            selected_experts = torch.argmax(routing_logits, dim=-1)  # (batch_size,)
            outputs = torch.zeros(x.size(0), 2, device=x.device)
            for i in range(x.size(0)):
                expert_idx = selected_experts[i].item()
                outputs[i] = self.experts[expert_idx](x[i])

        return outputs, selected_experts
