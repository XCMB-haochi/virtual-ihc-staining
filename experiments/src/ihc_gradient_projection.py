"""Asymmetric auxiliary projection after effective-batch accumulation.

The primary task gradient is preserved. All gradients share the same AMP scale;
the projection coefficient is scale invariant. AdamW may change the resulting
direction, so this is not a guarantee of primary-loss descent after an update.
"""
from __future__ import annotations

import torch


class PrimaryGradientProjection:
    def __init__(self, parameters):
        self.parameters = tuple(p for p in parameters if p.requires_grad)
        if not self.parameters:
            raise ValueError("Projection requires trainable parameters.")
        self.reset()

    def reset(self):
        self.primary = None

    def accumulate(self, primary_loss):
        """Accumulate unscaled primary gradients before AMP scales total loss."""
        gradients = torch.autograd.grad(primary_loss, self.parameters, retain_graph=True)
        if self.primary is None:
            self.primary = [gradient.detach().clone() for gradient in gradients]
        else:
            for summed, gradient in zip(self.primary, gradients):
                summed.add_(gradient.detach())

    @torch.no_grad()
    def project(self):
        if self.primary is None or any(p.grad is None for p in self.parameters):
            raise RuntimeError("Both accumulated primary and total gradients are required.")
        dots, norms = [], []
        for parameter, primary in zip(self.parameters, self.primary):
            # Compute reductions in float64 to avoid squared AMP-scale overflow.
            gp = primary.double()
            ga = parameter.grad.double() - gp
            dots.append((gp * ga).sum())
            norms.append(gp.square().sum())
        dot, norm = torch.stack(dots).sum(), torch.stack(norms).sum()
        if not bool(torch.isfinite(dot) & torch.isfinite(norm)):
            # Leave the original total gradient untouched. GradScaler decides
            # whether an actually non-finite optimizer update should be skipped.
            return {"finite": False, "coefficient": None, "conflict": False}
        coefficient = max(0.0, -float(dot) / float(norm)) if float(norm) > 0 else 0.0
        if coefficient:
            for parameter, primary in zip(self.parameters, self.primary):
                parameter.grad.add_(primary, alpha=coefficient)
        return {"finite": True, "coefficient": coefficient, "conflict": coefficient > 0}
