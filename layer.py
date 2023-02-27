import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.utils.degree import maybe_num_nodes

def B_degree(index, num_nodes, weight, dtype):
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    one = one * weight
    return out.scatter_add_(0, index, one)


class Scalar_Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, B=False):
        super(Scalar_Conv, self).__init__(aggr="add", node_dim=0)
        self.s_to_s = nn.Linear(2 * in_channels, out_channels)
        self.v_to_s = nn.Linear(2 * in_channels, out_channels)
        self.scalar_linear = nn.Linear(2 * out_channels, out_channels)
        self.activation = nn.SiLU()
        self.B = B
        self.out_channels = out_channels

    def forward(self, scalar, vector, position, edge_index, edge_attr):
        if self.B: 
            row, col = edge_index
            deg = B_degree(col, scalar.size(0), edge_attr, dtype=scalar.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0     
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_attr
        else:
            edge_index, _ = add_self_loops(edge_index, num_nodes=scalar.size(0))   
            row, col = edge_index
            deg = degree(col, scalar.size(0), dtype=scalar.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, scalar=scalar, vector=vector, position=position, norm=norm)

    def message(self, scalar_i, scalar_j, vector_i, vector_j, position_i, position_j, norm):
        s_to_s = (torch.cat([scalar_i, scalar_j], dim=1))
        v_to_s = torch.cat([vector_i, vector_j], dim=-1)
        position = position_i - position_j

        s_to_s = self.activation(self.s_to_s(s_to_s))
        v_to_s = self.v_to_s(v_to_s)
        # position = position.float()
        position = position[:, :, None]
        position = position.expand(-1, -1, self.out_channels)
        v_to_s = self.activation(torch.sum(torch.mul(v_to_s, position), dim=-2))
        scalar_feature = torch.cat([(s_to_s).float(), (v_to_s).float()], dim=1)
        return norm.view(-1, 1) * self.scalar_linear(scalar_feature)


class Vector_Conv(MessagePassing):
    def __init__(self, in_channels, out_channels, B=False):
        super(Vector_Conv, self).__init__(aggr="add", node_dim=0)
        self.s_to_v = nn.Linear(2 * in_channels, out_channels)
        self.v_to_v = nn.Linear(2 * in_channels, out_channels)
        self.vector_linear = nn.Linear(2 * out_channels, out_channels)
        self.activation = nn.Tanh()
        self.B = B
        self.out_channels = out_channels

    def forward(self, scalar, vector, position, edge_index, edge_attr):
        if self.B: 
            row, col = edge_index
            deg = B_degree(col, scalar.size(0), edge_attr, dtype=scalar.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_attr
        else:
            edge_index, _ = add_self_loops(edge_index, num_nodes=scalar.size(0))   
            row, col = edge_index
            deg = degree(col, scalar.size(0), dtype=scalar.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] 
        return self.propagate(edge_index, scalar=scalar, vector=vector, position=position, norm=norm)

    def message(self, scalar_i, scalar_j, vector_i, vector_j, position_i, position_j, norm):
        s_to_v = torch.cat([scalar_i, scalar_j], dim=1)
        v_to_v = torch.cat([vector_i, vector_j], dim=-1)
        position = position_i - position_j
        position = position[:, :, None]
        position = position.expand(-1, -1, self.out_channels)

        s_to_v = self.s_to_v(s_to_v)
        s_to_v = s_to_v[:, None, :]
        s_to_v = s_to_v.expand(-1, 3, -1)
        s_to_v = self.activation(torch.mul(s_to_v, position))

        v_to_v = self.v_to_v(v_to_v)

        vector_feature = torch.cat([(v_to_v).float(), (s_to_v).float()], dim=-1)
        vector_feature = self.vector_linear(vector_feature)
        norm = norm[:, None, None]
        norm = norm.expand(-1, 3, self.out_channels)
        return torch.mul(norm, vector_feature)


class MolNet_Layer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, B=False):
        super(MolNet_Layer, self).__init__()
        self.scalar_conv = Scalar_Conv(in_channels, out_channels, B)
        self.vector_conv = Vector_Conv(in_channels, out_channels, B)
        self.silu = torch.nn.SiLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, scalar, vector, position, edge_index, edge_attr):
        scalar_feature = self.silu(self.scalar_conv(scalar, vector, position, edge_index, edge_attr)) + scalar
        vector_feature = self.tanh(self.vector_conv(scalar, vector, position, edge_index, edge_attr)) + vector
        return scalar_feature, vector_feature

# class Test_Conv(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super(Test_Conv, super).__init__(aggr="add")
#         self.scalar_liner_1 = nn.Linear(in_channels, 128)
#         self.scalar_liner_2 = nn.Linear(128, 128)
#         self.radial_linear = nn.Linear(20, 384)
#         self.activation = nn.ReLU()

#     def forward(self, scalar, vector, position, edge_index):
#         pass
    
#     def RBF(self, position1, position2):
#         return None

#     def message(self, scalar_i, scalar_j, vector_i, vector_j, position_i, position_j, norm):
#         neighbor_scalar = self.scalar_liner_2(self.activation(self.scalar_liner_1(scalar_j)))
#         neighbor_direction = self.radial_linear(self.RBF(position_i, position_j))
#         weight = torch.dot(neighbor_scalar, neighbor_direction)
#         weight_1, weight_2, weight_3 = torch.split(weight, [128, 128, 128])

#     def update(self, inputs):
#         return inputs

class Scalar_Conv_eq(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Scalar_Conv_eq, self).__init__(aggr="add")
        self.s_to_s = nn.Linear(2 * in_channels, in_channels)
        self.v_to_s = nn.Linear(2 * in_channels, in_channels)
        self.scalar_linear = nn.Linear(2 * in_channels, out_channels)
        self.activation = nn.ReLU()

    def forward(self, scalar, vector, position, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=scalar.size(0))   
        row, col = edge_index
        deg = degree(col, scalar.size(0), dtype=scalar.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
        return self.propagate(edge_index, scalar=scalar, vector=vector, position=position, norm=norm)

    def message(self, scalar_i, scalar_j, vector_i, vector_j, position_i, position_j, norm):
        s_to_s = (torch.cat([scalar_i, scalar_j], dim=1))
        v_to_s = torch.cat([vector_i, vector_j], dim=1).permute((0, 2, 1))
        position = position_i - position_j

        s_to_s = self.activation(self.s_to_s(s_to_s))
        v_to_s = self.v_to_s(v_to_s).permute((0, 2, 1))
        position = position.float()
        v_to_s = self.activation(torch.einsum("abc,ac->ab ", v_to_s, position))
        scalar_feature = torch.cat([(s_to_s).float(), (v_to_s).float()], dim=1)
        return norm.view(-1, 1) * self.scalar_linear(scalar_feature)


class Vector_Conv_eq(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(Vector_Conv_eq, self).__init__(aggr="add")
        self.s_to_v = nn.Linear(2 * in_channels, in_channels)
        self.v_to_v = nn.Linear(2 * in_channels, in_channels)
        self.vector_linear = nn.Linear(2 * in_channels, out_channels)
        self.activation = nn.Tanh()

    def forward(self, scalar, vector, position, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=scalar.size(0))   
        row, col = edge_index
        deg = degree(col, scalar.size(0), dtype=scalar.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    
        return self.propagate(edge_index, scalar=scalar, vector=vector, position=position, norm=norm)

    def message(self, scalar_i, scalar_j, vector_i, vector_j, position_i, position_j, norm):
        s_to_v = torch.cat([scalar_i, scalar_j], dim=1)
        v_to_v = torch.cat([vector_i, vector_j], dim=1).permute((0, 2, 1))
        position = position_i - position_j

        s_to_v = self.s_to_v(s_to_v)
        s_to_v = self.activation(torch.einsum("ab,ac->abc", s_to_v, position))
        v_to_v = self.v_to_v(v_to_v).permute((0, 2, 1))

        vector_feature = torch.cat([(v_to_v).float(), (s_to_v).float()], dim=1).permute((0, 2, 1))
        vector_feature = self.vector_linear(vector_feature).permute((0, 2, 1))

        return torch.einsum("ab,acd->acd", norm.view(-1, 1), vector_feature)


class GCN3D_Layer_eq(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN3D_Layer_eq, self).__init__()
        self.scalar_conv = Scalar_Conv(in_channels, out_channels)
        self.vector_conv = Vector_Conv(in_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

    def forward(self, scalar, vector, position, edge_index):
        scalar_feature = self.relu(self.scalar_conv(scalar, vector, position, edge_index))
        vector_feature = self.tanh(self.vector_conv(scalar, vector, position, edge_index))
        return scalar_feature, vector_feature
