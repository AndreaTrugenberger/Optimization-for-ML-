"""
optimizers.py — Clean implementations of SignSGD, Lion, and AdaHessian.
Adam is provided by PyTorch; we import it from torch.optim.
"""

import torch
from torch.optim import Optimizer


# ─── SignSGD (signum variant from the paper) ──────────────────────────────────

class SignSGD(Optimizer):
    """
    SignSGD with momentum — exactly the 'signum' algorithm from
    Bernstein et al. (2018) "signSGD: Compressed Optimisation for Non-Convex
    Problems" (https://arxiv.org/abs/1802.04434, Algorithm 2):

        m_{k+1} ← β m_k + (1 - β) g_k         (EMA momentum)
        θ_{k+1} ← θ_k - η · sign(m_{k+1})     (sign update)

    Note the EMA momentum: the gradient enters with weight (1 - β), so β and
    (1 - β) sum to 1. This is the convex-combination form from the paper,
    not the classical heavy-ball form where the gradient is added at full
    weight.
    """
    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            beta         = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if weight_decay != 0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                # EMA momentum update:   m ← β·m + (1 - β)·g
                buf.mul_(beta).add_(g, alpha=1 - beta)
                # Sign update:           θ ← θ - η · sign(m)
                p.add_(buf.sign(), alpha=-lr)

        return loss


# ─── Lion ─────────────────────────────────────────────────────────────────────

class Lion(Optimizer):
    """
    Lion (EvoLved Sign Momentum) — Chen et al. (2023)
    "Symbolic Discovery of Optimization Algorithms"
    (https://arxiv.org/abs/2302.06675).

    Distinct from SignSGD in two ways:
      (1) The parameter update uses a normalised combination of momentum
          and current gradient (weights β1 and 1-β1 sum to 1).
      (2) The momentum buffer is updated separately with DIFFERENT weights
          (β2 and 1-β2), decoupling memory evolution from update direction.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                m = state["exp_avg"]

                # Update direction: sign of β1·m + (1-β1)·g
                update = (beta1 * m + (1 - beta1) * g).sign_()

                # Decoupled weight decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                p.add_(update, alpha=-lr)

                # Momentum update uses DIFFERENT weights (β2, 1-β2)
                m.mul_(beta2).add_(g, alpha=1 - beta2)

        return loss


# ─── AdaHessian ───────────────────────────────────────────────────────────────

class AdaHessian(Optimizer):
    """
    AdaHessian — Yao et al. (2021)
    "ADAHESSIAN: An Adaptive Second Order Optimizer for ML"
    (https://arxiv.org/abs/2006.00719).

    Approximates the diagonal of the Hessian via Hutchinson's trick:
    a single Rademacher vector z is sampled per step, and the diagonal is
    estimated as diag(H) ≈ z ⊙ (Hz). Uses this as a pre-conditioner for
    coordinate-wise adaptive learning rates.

    NOTE: requires loss.backward(create_graph=True) in the training loop.
    """
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0.0, hessian_power=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power)
        super().__init__(params, defaults)

    def _get_hessian_diag(self, params_with_grad):
        """
        Hutchinson estimator of the Hessian diagonal:
            diag(H) ≈ z ⊙ (Hz),  z ~ Rademacher(±1)
        Computed using a single random sample per step.
        """
        zs = [torch.randint_like(p, high=2).float() * 2 - 1
              for p in params_with_grad]
        grads = [p.grad for p in params_with_grad]
        hvps  = torch.autograd.grad(
            outputs=grads, inputs=params_with_grad,
            grad_outputs=zs, only_inputs=True, retain_graph=False,
        )
        return [z * hvp for z, hvp in zip(zs, hvps)]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        params_with_grad = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)

        if not params_with_grad:
            return loss

        with torch.enable_grad():
            hess_diags = self._get_hessian_diag(params_with_grad)

        p_idx = 0
        for group in self.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            eps          = group["eps"]
            weight_decay = group["weight_decay"]
            hess_power   = group["hessian_power"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g  = p.grad
                hd = hess_diags[p_idx].abs()
                p_idx += 1

                if weight_decay != 0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["step"]     = 0
                    state["exp_avg"]  = torch.zeros_like(p)
                    state["exp_hess"] = torch.zeros_like(p)

                state["step"] += 1
                m = state["exp_avg"]
                v = state["exp_hess"]
                t = state["step"]

                m.mul_(beta1).add_(g,  alpha=1 - beta1)
                v.mul_(beta2).add_(hd, alpha=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                denom = v_hat.pow(hess_power).add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)

        return loss
