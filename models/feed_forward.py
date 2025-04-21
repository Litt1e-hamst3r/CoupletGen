import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    device,
    d_model,
    d_ff,
    d_k,
    d_v,
    n_layers,
    n_heads,
    dropout,
    moe_k,
    capacity_factor,
    noise_epsilon,
)

# 实现了Noisy Top-K门控机制(选择top 2专家)
# 增加了专家容量控制，防止单个专家过载
# 改进了专家间的负载均衡
# 训练时加入噪声增强探索能力


class MoeLayer(nn.Module):
    def __init__(self, num_experts=4, k=2, capacity_factor=1.0, noise_epsilon=1e-2):
        super(MoeLayer, self).__init__()
        self.num_experts = num_experts
        self.k = min(k, num_experts)  # Can't use more experts than available
        self.capacity_factor = capacity_factor
        self.noise_epsilon = noise_epsilon

        # Expert networks
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff, bias=False),
                    nn.ReLU(),
                    nn.Linear(d_ff, d_model, bias=False),
                )
                for _ in range(num_experts)
            ]
        )

        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.layer_norm = nn.LayerNorm(d_model).to(device)

    def noisy_top_k_gating(self, x):
        # Add noise for exploration
        clean_logits = self.gate(x)
        if self.training:
            noise = torch.randn_like(clean_logits) * self.noise_epsilon
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits

        # Top-k gating
        top_k_logits, top_k_indices = noisy_logits.topk(self.k, dim=1)
        top_k_gates = F.softmax(top_k_logits, dim=1)

        # Create mask for selected experts
        zeros = torch.zeros_like(noisy_logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        return gates, top_k_indices

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        batch_size, seq_len, d_model = inputs.shape
        inputs_flat = inputs.view(-1, d_model)  # [batch_size*seq_len, d_model]

        # Get gate values and expert indices
        gates, expert_indices = self.noisy_top_k_gating(inputs_flat)

        # Calculate expert capacity
        expert_capacity = int(
            (batch_size * seq_len * self.capacity_factor) / self.num_experts
        )
        expert_capacity = max(expert_capacity, 4)  # Minimum capacity

        # Process through experts with load balancing
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Get inputs for this expert
            expert_mask = (expert_indices == i).any(dim=1)
            expert_inputs = inputs_flat[expert_mask]

            if len(expert_inputs) > 0:
                # Process with expert (truncate to capacity if needed)
                expert_inputs = expert_inputs[:expert_capacity]
                expert_output = expert(expert_inputs)
                expert_outputs.append(expert_output)
            else:
                expert_outputs.append(torch.zeros((0, d_model), device=device))

        # Combine expert outputs
        moe_output = torch.zeros_like(inputs_flat)
        for i, expert_out in enumerate(expert_outputs):
            # Get gate values for this expert
            expert_gates = gates[:, i].unsqueeze(1)
            expert_mask = (expert_indices == i).any(dim=1)

            if len(expert_out) > 0:
                # Apply gates to expert outputs
                # Ensure we only apply to the first expert_capacity inputs
                expert_mask = expert_mask.nonzero().squeeze(1)
                if len(expert_mask) > expert_capacity:
                    expert_mask = expert_mask[:expert_capacity]
                moe_output[expert_mask] += expert_out * expert_gates[expert_mask]

        moe_output = moe_output.view(batch_size, seq_len, d_model)
        return self.layer_norm(moe_output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, use_moe=False, num_experts=4):
        super(PoswiseFeedForwardNet, self).__init__()
        if use_moe:
            self.ffn = MoeLayer(
                num_experts=num_experts,
                k=moe_k,
                capacity_factor=capacity_factor,
                noise_epsilon=noise_epsilon,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.ReLU(),
                nn.Linear(d_ff, d_model, bias=False),
            )
        self.layer_norm = nn.LayerNorm(d_model).to(device)

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.ffn(inputs)
        return self.layer_norm(output + residual)
