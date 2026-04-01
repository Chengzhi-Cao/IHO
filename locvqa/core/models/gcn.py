import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math
import numpy as np
import torch.nn.functional as F
 
 
class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True, device=None, dtype=None) -> None:
        """
        空域图卷积：
        :param input_dim 输入单节点特征数
        :param output_dim 输出单节点特征数
        :param use_bias 是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = bias
        self.weight = Parameter(torch.empty((input_dim, output_dim), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(output_dim, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
 
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
 
    def forward(self, input_feature, adjacency):
        """
        adjacency: torch.FloatTensor 邻接矩阵 [nodes, nodes]
        input_feature: torch.Tensor   输入节点集合: [batch, nodes,  features]
        process:
            H(l+1) = D^(-1/2)*A*D^(-1/2)*H(l)*W
        """
        adjacency = self.standard_adjacency(adjacency)  # (8, 18, 8, 8)
        support = torch.einsum("bij,bjk->bik", [adjacency, input_feature])
        output = torch.matmul(support, self.weight)
        if self.use_bias:
            output += self.bias
        return output
 
    def standard_adjacency(self, adjacency):  # 注意这里的邻接矩阵应带有自环
        """
        :param adjacency: 邻接矩阵 [batch, nodes, nodes]
        :return: 标准化邻接矩阵: [batch, nodes, nodes]
        """
        degree_matrix = torch.sum(adjacency, dim=-1, keepdim=False)  # [8, 18, 8]
        degree_matrix = degree_matrix.pow(-0.5)
        degree_matrix[degree_matrix == float("inf")] = 0.  # [64, 18]
        degree_matrix = degree_matrix.reshape(-1, len(degree_matrix[0]), 1) * torch.eye(
            len(degree_matrix[0]), len(degree_matrix[0])).cuda() # (16, 8, 8)
        return torch.matmul(torch.matmul(degree_matrix, 1.0 * adjacency), degree_matrix)  # [64, 18, 18]
 
 
# if __name__ == "__main__":
#     graph_conv = GraphConvolution(49, 49)
#     # 这是我们随意设置的一个邻接矩阵，16 表示batch
#     adjacency = torch.zeros(16, 2048, 2048) + torch.Tensor(np.eye(2048, k=-1) + np.eye(2048) + np.eye(2048, k=1))
#     x = torch.randn(16, 2048, 49)  # 输入一个batch 的节点特征
#     h = F.relu(graph_conv(x, adjacency))  # 输出更新后的节点特征, 并使用relu作为激活函数
    
#     print('X=',x.shape)
#     print('h=',h.shape)
#     print(h.shape)
#     # print(h)