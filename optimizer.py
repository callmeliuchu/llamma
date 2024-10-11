from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        # state = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Update first and second moments of the gradients
                beta1,beta2 = group['betas']
                if 'm1' not in state:
                    state['m1'] = torch.zeros_like(grad)
                    state['t'] = 0
                    # print('m1111',self.m1.shape,grad.shape)
                if 'm2' not in state:
                    state['m2'] = torch.zeros_like(grad)
                state['t'] += 1
                # print('m1',self.m1.shape,'grad',grad.shape)
                state['m1'] = beta1 * state['m1'] + (1-beta1) * grad
                state['m2'] = beta2 * state['m2'] + (1-beta2) * (grad ** 2)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                correct_bias = group['correct_bias']
                m1_hat = state['m1']
                m2_hat = state['m2']
                if correct_bias:
                    m1_hat = state['m1'] / (1-beta1**state['t'])
                    m2_hat = state['m2'] / (1-beta2**state['t'])

                # Update parameters
                eps = group['eps']
                p.data -= (m1_hat / (m2_hat**0.5 + eps) ) * alpha


                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                weight_decay = group['weight_decay']
                p.data -= alpha * (1 * weight_decay * p.data)

        return loss