from typing import Tuple, Dict, Iterable, List

import torch

class Optimizer:
    def __init__(self, params: Iterable[Dict[str, torch.Tensor]], lr: float = 1e-3) -> None:
        self.lr = lr
        self.param_groups: List[Dict[str, torch.Tensor]] = [dict(group) for group in params]

    def zero_grad(self):
        for group in self.param_groups:
            grad = group.get("grad")
            if grad is not None:
                grad.zero_()

    def step(self):
        raise NotImplementedError

class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float, 
        beta: Tuple[float] = [0.9, 0.999],
        eps: float = 1e-8,
        weight: float = 0.01,
    ):
        super().__init__(params=params, lr=lr)
        self.beta_1, self.beta_2 = beta
        self.eps = eps
        self.weight = weight
        for group in self.param_groups:
            param = group["param"]
            if param is None:
                continue
            group["m"] = torch.zeros_like(param)
            group["v"] = torch.zeros_like(param)
            group["t"] = 0

    def step(self):
        for group in self.param_groups:
            grad = group["grad"]
            param = group["param"]
            m = group["m"]
            v = group["v"]
            t = group["t"]
            t += 1

            m.mul_(self.beta_1).add_(grad, alpha=1-self.beta_1)
            v.mul_(self.beta_2).addcmul_(grad, grad, value=1-self.beta_2)
            m_hat = m / (1-self.beta_1 ** t)
            v_hat = v / (1-self.beta_2 ** t)

            param.mul_(1-self.lr*self.weight)
            param.addcdiv_(m_hat, torch.sqrt(v_hat) + self.eps, value=-self.lr)
            group["t"] = t