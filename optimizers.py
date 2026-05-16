"""
optimizers.py — Clean implementations of SignSGD, Lion, and AdaHessian.
Adam is provided by PyTorch; we import it from torch.optim.
"""

import torch
from torch.optim import Optimizer


# ─── SignSGD ──────────────────────────────────────────────────────────────────

class SignSGD(Optimizer):
    """
    signSGD: uses only the *sign* of the gradient for each coordinate.
    Each coordinate gets ±lr, completely ignoring gradient magnitude.
    Ref: Bernstein et al. (2018) "signSGD: Compressed Optimisation for Non-Convex Problems"
         https://arxiv.org/abs/1802.04434
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
            momentum     = group["momentum"]
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
                buf.mul_(momentum).add_(g)          # heavy-ball momentum on gradient
                p.add_(buf.sign(), alpha=-lr)        # step = -lr * sign(momentum_buffer)

        return loss


# ─── Lion ─────────────────────────────────────────────────────────────────────

class Lion(Optimizer):
    """
    Lion (EvoLved Sign Momentum): uses sign of an EMA of gradients.
    More memory-efficient than Adam; works well with larger batch sizes.
    Ref: Chen et al. (2023) "Symbolic Discovery of Optimization Algorithms"
         https://arxiv.org/abs/2302.06675
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

                # update: sign of interpolation between EMA and current gradient
                update = (beta1 * m + (1 - beta1) * g).sign_()

                # weight decay applied directly to parameter
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                p.add_(update, alpha=-lr)

                # update EMA (not the same as the update above!)
                m.mul_(beta2).add_(g, alpha=1 - beta2)

        return loss


# ─── AdaHessian ───────────────────────────────────────────────────────────────

class AdaHessian(Optimizer):
    """
    AdaHessian: diagonal Hessian estimated via Hutchinson's trick (random Rademacher vectors).
    Uses Hessian curvature info per coordinate to set adaptive learning rates,
    similar to Adagrad/Adam but with second-order information.
    Ref: Yao et al. (2021) "ADAHESSIAN: An Adaptive Second Order Optimizer for ML"
         https://arxiv.org/abs/2006.00719

    NOTE: requires loss.backward(create_graph=True) in the training loop.
    """
    def __init__(self, params, lr=0.1, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0.0, hessian_power=1.0, spatial_average_blocksize=1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        hessian_power=hessian_power,
                        spatial_average_blocksize=spatial_average_blocksize)
        super().__init__(params, defaults)

    def _get_hessian_diag(self, params_with_grad):
        """
        Hutchinson estimator of the diagonal of the Hessian.
        Draws one Rademacher vector z ∈ {±1}^d and computes
        diag(H) ≈ z ⊙ (H z),  where H z = ∇(∇L · z).
        """
        # Rademacher vector
        zs = [torch.randint_like(p, high=2).float() * 2 - 1
              for p in params_with_grad]

        # Hessian-vector product: grad of (grad · z) w.r.t. params
        grads = [p.grad for p in params_with_grad]
        hvps  = torch.autograd.grad(
            outputs=grads,
            inputs=params_with_grad,
            grad_outputs=zs,
            only_inputs=True,
            retain_graph=False,
        )

        # Hutchinson estimate: element-wise z ⊙ Hv
        return [z * hvp for z, hvp in zip(zs, hvps)]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Collect params that have gradients (need enable_grad for hvp)
        params_with_grad = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)

        if not params_with_grad:
            return loss

        # Hessian diagonal (runs autograd, so do it before @no_grad context kills grads)
        with torch.enable_grad():
            hess_diags = self._get_hessian_diag(params_with_grad)

        # Apply updates
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
                hd = hess_diags[p_idx].abs()     # absolute value of Hessian diagonal
                p_idx += 1

                if weight_decay != 0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["step"]      = 0
                    state["exp_avg"]   = torch.zeros_like(p)
                    state["exp_hess"]  = torch.zeros_like(p)   # EMA of |hessian diag|

                state["step"] += 1
                m  = state["exp_avg"]
                v  = state["exp_hess"]
                t  = state["step"]

                m.mul_(beta1).add_(g,  alpha=1 - beta1)
                v.mul_(beta2).add_(hd, alpha=1 - beta2)

                # Bias correction
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                denom = v_hat.pow(hess_power).add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)

        return loss
