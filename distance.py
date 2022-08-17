from cmath import isnan
from numpy import double
from regex import P
import torch

def calculate_group_to_group_relative_distance_asymmetric_test(batch_data, perplexity=10):
    batch_data = batch_data.view(batch_data.shape[0], batch_data.shape[1], -1)
    new_batch_data = torch.zeros((batch_data.shape[0], batch_data.shape[1], batch_data.shape[1]), dtype=torch.float32)
    for i in range(batch_data.shape[1]):
        for j in range(batch_data.shape[1]):
            if i != j:
                new_batch_data[..., i, j] = torch.exp(-torch.sum((batch_data[..., i, :] - batch_data[..., j, :]) ** 2, dim=1) / (2 * (perplexity ** 2)))
    new_batch_data = new_batch_data / (torch.sum(new_batch_data, dim=1).unsqueeze(-1) * batch_data.shape[1])
    new_batch_data = new_batch_data.view(batch_data.shape[0], batch_data.shape[1] ** 2)
    return new_batch_data

def calculate_group_to_group_relative_distance_asymmetric(batch_data, perplexity=10):
    # batch_data.shape: torch.Size((32, 53, 13, 768))
    batch_size = batch_data.shape[0]
    n = batch_data.shape[1]
    batch_data = batch_data.view(batch_size, n, -1)
    data_size = batch_data.shape[-1]
    batch_data_buffer_i = batch_data.unsqueeze(2).expand(batch_size, n, n, data_size)
    batch_data_buffer_j = batch_data.unsqueeze(1).expand(batch_size, n, n, data_size)
    new_batch_data = torch.exp(-torch.sum((batch_data_buffer_i - batch_data_buffer_j) ** 2, dim=-1) / (2 * (perplexity ** 2))).view(batch_size, n, n)
    drop_position_b = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n).reshape(-1)
    drop_position_n = torch.arange(n).unsqueeze(0).expand(batch_size, n).reshape(-1)
    mask = torch.ones_like(new_batch_data)
    mask[(drop_position_b, drop_position_n, drop_position_n)] = 0
    new_batch_data = new_batch_data * mask
    new_batch_data = new_batch_data / (torch.sum(new_batch_data, dim=1).unsqueeze(-1) * batch_data.shape[1])
    new_batch_data = new_batch_data.view(batch_data.shape[0], batch_data.shape[1] ** 2)
    # new_batch_data.shape: torch.Size([32, 2809])
    return new_batch_data

def calculate_group_to_one_relative_distance_asymmetric_test(neutral_batch_data, attribute_data, perplexity=10):
    # neutral_batch_data.shape: torch.Size([32, 1, 13, 768])
    # attribute_data.shape: torch.Size([2, 13, 768])
    n_shape0 = neutral_batch_data.shape[0]
    a_shape0 = attribute_data.shape[0]
    neutral_batch_data = neutral_batch_data.view(n_shape0, -1)
    attribute_data = attribute_data.view(a_shape0, -1)
    assert neutral_batch_data.shape[1] == attribute_data.shape[1]
    new_batch_data = torch.zeros((a_shape0, n_shape0), dtype=torch.float32)
    for i in range(a_shape0):
        for j in range(n_shape0):
            new_batch_data[i, j] = torch.exp(-torch.sum((attribute_data[i] - neutral_batch_data[j]) ** 2) / (2 * (perplexity ** 2)))
    new_batch_data = new_batch_data / torch.sum(new_batch_data, dim=1).unsqueeze(-1)
    return new_batch_data

def calculate_group_to_one_relative_distance_asymmetric(neutral_batch_data, attribute_data, perplexity=10):
    # neutral_batch_data.shape: torch.Size([32, 1, 13, 768])
    # attribute_data.shape: torch.Size([2, 13, 768])
    n_shape0 = neutral_batch_data.shape[0]
    a_shape0 = attribute_data.shape[0]
    neutral_batch_data = neutral_batch_data.view(n_shape0, -1)
    attribute_data = attribute_data.view(a_shape0, -1)
    assert neutral_batch_data.shape[1] == attribute_data.shape[1]
    data_size = neutral_batch_data.shape[1]
    _neutral = neutral_batch_data.unsqueeze(0).expand(a_shape0, n_shape0, data_size)
    _neutral = _neutral.reshape(-1, data_size)
    _attribute = attribute_data.unsqueeze(1).expand(a_shape0, n_shape0, data_size)
    _attribute = _attribute.reshape(-1, data_size)
    new_batch_data = torch.exp(-torch.sum((_attribute - _neutral) ** 2, dim=-1) / (2 * (perplexity ** 2))).view(a_shape0, n_shape0)
    new_batch_data = new_batch_data / torch.sum(new_batch_data, dim=1).unsqueeze(-1)
    # new_batch_data.shape: torch.Size([2, 32])
    return new_batch_data

def KL_divergence(p_distribution, q_distribution):
    sum_total = torch.sum(p_distribution)
    _p = p_distribution[torch.where((p_distribution!=0)&(q_distribution!=0))]
    _q = q_distribution[torch.where((p_distribution!=0)&(q_distribution!=0))]
    result = _p * torch.log2(_p / _q)
    result = torch.sum(result) / sum_total
    return result

def JS_divergence(p_distribution, q_distribution):
    m = 0.5 * (p_distribution + q_distribution)
    return 0.5 * KL_divergence(p_distribution, m) + 0.5 * KL_divergence(q_distribution, m)

if __name__ == "__main__":
    a = torch.randn((32, 53, 13, 768)).view(32, 53, 13, 768)
    b = calculate_group_to_group_relative_distance_asymmetric(a)
    c = calculate_group_to_group_relative_distance_asymmetric_test(a)
    d = KL_divergence(b, c)
    print(d)

    n = torch.randn((32, 1, 13, 768)).view(32, 1, 13, 768)
    a = torch.randn((2, 13, 768)).view(2, 13, 768)
    b = calculate_group_to_one_relative_distance_asymmetric(n, a)
    c = calculate_group_to_one_relative_distance_asymmetric_test(n, a)
    d = KL_divergence(b, c)
    print(d)
