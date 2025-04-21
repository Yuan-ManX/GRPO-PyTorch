import math

import torch
from torch.optim import AdamW


class MemoryEfficientAdamW(AdamW):
    """
    MemoryEfficientAdamW 是一个内存高效的 AdamW 优化器实现。
    它将参数和梯度保留在 GPU 上，但当启用时，将优化器状态存储在 CPU 上。
    当禁用时，其行为与标准的 AdamW 优化器完全相同。
    
    参数:
        params (iterable): 待优化的参数或定义参数组的字典。
        lr (float, 可选): 学习率。默认值为 1e-3。
        betas (Tuple[float, float], 可选): 一阶和二阶矩估计的衰减率。默认值为 (0.9, 0.999)。
        eps (float, 可选): 为了数值稳定性而加到分母上的小常数。默认值为 1e-8。
        weight_decay (float, 可选): 权重衰减（L2惩罚）。默认值为 1e-2。
        amsgrad (bool, 可选): 是否使用 AMSGrad 变体。默认值为 False。
        pin_memory (bool, 可选): 是否将优化器状态张量固定到内存中。默认值为 True。
        enabled (bool, 可选): 是否启用内存高效模式。默认值为 True。
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        pin_memory=True,
        enabled=True,
    ):
        # 调用父类 AdamW 的初始化方法，传递必要的参数
        super(MemoryEfficientAdamW, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        # 是否将优化器状态张量固定到内存中
        self.pin_memory = pin_memory
        # 是否启用内存高效模式
        self.enabled = enabled

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行单步优化。
        
        参数:
            closure (callable, 可选): 一个可调用对象，可以重新评估模型并返回损失。默认值为 None。
        
        返回:
            loss (torch.Tensor, 可选): 如果提供了 closure，则返回损失值。
        """
        if not self.enabled:
            # 当内存高效模式禁用时，使用父类 AdamW 的 step 方法
            return super(MemoryEfficientAdamW, self).step(closure)

        loss = None
        if closure is not None:
            # 如果提供了 closure，则在启用梯度的情况下执行 closure
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # 初始化用于存储参数、梯度、一阶矩、二阶矩、最大二阶矩和步骤计数的列表
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            # 获取当前参数组的一阶和二阶矩估计的衰减率
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue  # 如果参数没有梯度，则跳过

                params_with_grad.append(p)
                grads.append(p.grad)

                # 初始化状态字典，如果尚未初始化
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # 将优化器状态存储在 CPU 上，并固定到内存中（如果启用）
                    device = "cpu"
                    pin_memory = self.pin_memory
                    dtype = torch.float32

                    state["exp_avg"] = torch.zeros_like(
                        p.data, device=device, pin_memory=pin_memory, dtype=dtype
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, device=device, pin_memory=pin_memory, dtype=dtype
                    )
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p.data, device=device, pin_memory=pin_memory, dtype=dtype
                        )

                # 获取当前参数的状态中的值
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                # 更新步骤计数
                state["step"] += 1
                state_steps.append(state["step"])

            # 对当前参数组中的所有参数执行内存高效的更新
            self._memory_efficient_update(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

        return loss

    def _memory_efficient_update(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad,
        beta1,
        beta2,
        lr,
        weight_decay,
        eps,
    ):
        """
        在 GPU 上执行 AdamW 参数更新，同时将优化器状态存储在 CPU 上。
        使用固定内存以实现优化器状态的高效 CPU 到 GPU 传输。
        
        参数:
            params (List[torch.Tensor]): 需要更新的参数列表。
            grads (List[torch.Tensor]): 对应参数的梯度列表。
            exp_avgs (List[torch.Tensor]): 一阶矩估计列表。
            exp_avg_sqs (List[torch.Tensor]): 二阶矩估计列表。
            max_exp_avg_sqs (List[torch.Tensor]): 最大二阶矩估计列表（如果启用 amsgrad）。
            state_steps (List[int]): 每个参数的步骤计数列表。
            amsgrad (bool): 是否使用 AMSGrad 变体。
            beta1 (float): 一阶矩估计的衰减率。
            beta2 (float): 二阶矩估计的衰减率。
            lr (float): 学习率。
            weight_decay (float): 权重衰减。
            eps (float): 小常数，用于数值稳定性。
        """
        for i, param in enumerate(params):
            grad = grads[i]
            param_device = param.device

            # 访问优化器状态 - 由于固定内存，传输效率高
            exp_avg = exp_avgs[i].to(param_device, non_blocking=True)
            exp_avg_sq = exp_avg_sqs[i].to(param_device, non_blocking=True)

            step = state_steps[i]

            # 衰减一阶和二阶矩估计
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if amsgrad:
                # 访问 max_exp_avg_sq - 由于固定内存，传输效率高
                max_exp_avg_sq = max_exp_avg_sqs[i].to(param_device, non_blocking=True)
                # 维护到目前为止所有二阶矩估计的最大值
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # 使用最大值来规范化梯度的运行平均值
                denom = max_exp_avg_sq.sqrt().add_(eps)
                # 将最大值存储回 CPU
                max_exp_avg_sqs[i].copy_(max_exp_avg_sq, non_blocking=True)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            # 直接对参数应用权重衰减（AdamW）
            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)

            # 更新参数（直接在 GPU 上）
            param.addcdiv_(exp_avg, denom, value=-step_size)

            # 将优化器状态存储回 CPU
            exp_avgs[i].copy_(exp_avg, non_blocking=True)
            exp_avg_sqs[i].copy_(exp_avg_sq, non_blocking=True)
