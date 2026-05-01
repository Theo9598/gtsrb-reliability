from __future__ import annotations

import torch
from torch import nn


class TemperatureScaler(nn.Module):
    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(initial_temperature))))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp_min(1e-4)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 80,
    learning_rate: float = 0.01,
) -> TemperatureScaler:
    scaler = TemperatureScaler().to(logits.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=learning_rate, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler

