"""
Implementations of SignSGD, Lion, and AdaHessian.
We use Adam implementation from PyTorch.
"""

import torch
from torch.optim import Optimizer


class SignSGD(Optimizer):
    # SignSGD with momentum, Bernstein et al. (2018) (https://arxiv.org/abs/1802.04434)
    
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



class Lion(Optimizer):
    # Lion, Chen et al. (2023) (https://arxiv.org/abs/2302.06675).

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


class AdaHessian(Optimizer):
    # AdaHessian, Yao et al. (2021) (https://arxiv.org/abs/2006.00719).

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
