import torch
import math

def adjust_tensor_statistic(tensor, scale_factor):
    interpolated_tensor = torch.zeros_like(tensor)
    for i in range(tensor.shape[1]):
        index = max(min(round((i)*scale_factor), tensor.shape[1]-1), 0)
        interpolated_tensor[:,index] += tensor[:,i]

    return interpolated_tensor

# 示例使用
tensor = torch.tensor([[0.1, 0.15, 0.2, 0.25, 0.3]], dtype=torch.float32)
tensor /= torch.sum(tensor)  # 保证总和为 1
scale_factor =  1.5  # 放缩因子
adjusted_tensor = adjust_tensor_statistic(tensor, scale_factor)
print(f"Original tensor: {tensor}")
print(f"Adjusted tensor: {adjusted_tensor}")
print(f"Sum of adjusted tensor: {torch.sum(adjusted_tensor)}")

# 打印统计值变化
indices = torch.arange(tensor.shape[1], dtype=tensor.dtype, device=tensor.device)
original_statistic = torch.sum(indices * tensor)
new_statistic = torch.sum(indices * adjusted_tensor)
print(f"Original Statistic: {original_statistic.item()}, New Statistic: {new_statistic.item()}")
