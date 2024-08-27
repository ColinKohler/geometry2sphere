from typing import Tuple
import math
import torch
import torch.nn as nn
import torch_geometric.utils
from torch_cluster import grid_cluster
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils.repeat import repeat as torch_geo_repeat

from e3nn import nn as enn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.util.jit import compile_mode

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
  irreps_in1 = o3.Irreps(irreps_in1).simplify()
  irreps_in2 = o3.Irreps(irreps_in2).simplify()
  ir_out = o3.Irrep(ir_out)

  for _, ir1 in irreps_in1:
    for _, ir2 in irreps_in2:
      if ir_out in ir1 * ir2:
        return True
  return False

def scatter(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
  # special case of torch_scatter.scatter with dim=0
  out = src.new_zeros(dim_size, src.shape[1])
  index = index.reshape(-1, 1).expand_as(src)
  return out.scatter_add_(0, index, src)

def radius_graph(pos, r_max, batch):
  # naive and inefficient version of torch_cluster.radius_graph
  r = torch.cdist(pos, pos)
  index = ((r < r_max) & (r > 0)).nonzero().T
  index = index[:, batch[index[0]] == batch[index[1]]]
  return index

def avg_pool(cluster, data):
  cluster, perm = consecutive_cluster(cluster)

  data.x = torch_geometric.utils.scatter(data.x, cluster, dim=0, reduce='mean')
  data.pos = torch_geometric.utils.scatter(data.pos, cluster, dim=0, reduce='mean')
  data.node_attr = torch_geometric.utils.scatter(data.node_attr, cluster, dim=0, reduce='mean')
  data.edge_index, data.edge_attr = pool_edge(cluster, data.edge_index, data.edge_attr)
  data.batch = data.batch[perm]

  return data

def topk(x, ratio, batch):
  num_nodes = torch_geometric.utils.scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
  k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

  x, x_perm = torch.sort(x.view(-1), descending=True)
  batch = batch[x_perm]
  batch, batch_perm = torch.sort(batch, descending=False, stable=True)

  arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
  ptr = num_nodes.new_zeros(num_nodes.numel() + 1)
  torch.cumsum(num_nodes, 0, out=ptr[1:])
  batched_arange = arange - ptr[batch]
  mask  = batched_arange < k[batch]

  return x_perm[batch_perm[mask]]

def pool_edge(cluster, edge_index, edge_attr=None):
  num_nodes = cluster.size(0)
  edge_index = cluster[edge_index.view(-1)].view(2, -1)
  edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
  if edge_index.numel() > 0:
    edge_index, edge_attr = torch_geometric.utils.coalesce(edge_index, edge_attr, num_nodes)
  return edge_index, edge_attr

def filter_adj(edge_index, edge_attr, node_index, cluster_index=None, num_nodes=None):
  #num_nodes = max(int(edge_src.max()) + 1, int(edge_dst.max()) + 1)

  if cluster_index is None:
    cluster_index = torch.arange(node_index.size(0), device=node_index.device)

  # Create batch mask
  mask = node_index.new_full((num_nodes, ), -1)
  mask[node_index] = cluster_index

  # Filter out edges that are no longer connected
  row, col = mask[edge_index[0]], mask[edge_index[1]]
  mask = (row >= 0) & (col >= 0)
  row, col = row[mask], col[mask]

  return torch.stack([row, col], dim=0)

def s2_near_identity_grid(max_beta: float = math.pi / 8, n_alpha: int = 8, n_beta: int = 3):
    """
    :return: rings around the north pole
    size of the kernel = n_alpha * n_beta
    """
    beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
    alpha = torch.linspace(0, 2 * math.pi, n_alpha + 1)[:-1]
    a, b = torch.meshgrid(alpha, beta, indexing="ij")
    b = b.flatten()
    a = a.flatten()
    return torch.stack((a, b))

def so3_near_identity_grid(max_beta=math.pi / 8, max_gamma=2 * math.pi, n_alpha=8, n_beta=3, n_gamma=None):
  """
  :return: rings of rotations around the identity, all points (rotations) in
  a ring are at the same distance from the identity
  size of the kernel = n_alpha * n_beta * n_gamma
  """
  if n_gamma is None:
      n_gamma = n_alpha  # similar to regular representations
  beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
  alpha = torch.linspace(0, 2 * math.pi, n_alpha)[:-1]
  pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
  A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
  C = preC - A
  A = A.flatten()
  B = B.flatten()
  C = C.flatten()
  return torch.stack((A, B, C))

# L^2 * S^2 -> natural decomposition for a signal over S^2
def s2_irreps(lmax):
  return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

# L^2 * SO(3) -> natural decomposition for a signal over SO(3)
def so3_irreps(lmax):
  return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])

def flat_wigner(lmax, alpha, beta, gamma):
  return torch.cat([(2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)], dim=-1)

# Model Blocks
class SO3Convolution(nn.Module):
  def __init__(self, f_in, f_out, lmax_in, kernel_grid, lmax_out=None):
    super().__init__()
    if lmax_out == None:
      lmax_out = lmax_in
    self.register_parameter(
      'w', nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
    ) # [f_in, f_out, n_so3_pts]
    self.register_parameter(
      'D', nn.Parameter(flat_wigner(lmax_in, *kernel_grid))
    ) # [n_so3_pts, psi]
    self.lin = o3.Linear(so3_irreps(lmax_in), so3_irreps(lmax_out), f_in=f_in, f_out=f_out, internal_weights=False)

  def forward(self, x):
    # S2 DFT (matrix mulitplication)
    psi = torch.einsum('ni,xyn->xyi', self.D, self.w) / self.D.shape[0] ** 0.5
    return self.lin(x, weight=psi)

#class SO3ToS2Convolution(nn.Module):
#  def __init__(self, f_in, f_out, lmax, kernel_grid):
#    super().__init__()
#    self.register_parameter(
#      'w', nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
#    ) # [f_in, f_out, n_so3_pts]
#    self.register_parameter(
#      #'D', nn.Parameter(flat_wigner(lmax, *kernel_grid))
#      'D', nn.Parameter(o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization='component'))
#    ) # [n_so3_pts, psi]
#    self.lin = o3.Linear(so3_irreps(lmax), s2_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)
#
#  def forward(self, x):
#    # B x F_in x psi
#    # S2 DFT (matrix mulitplication)
#    psi = torch.einsum('ni,xyn->xyi', self.D, self.w) / self.D.shape[0] ** 0.5
#    return self.lin(x, weight=psi)

class SO3ToS2Convolution(nn.Module):
  def __init__(self, f_in, f_out, lmax_in, lmax_out, kernel_grid):
    super().__init__()
    self.lin = o3.Linear(so3_irreps(lmax_in), s2_irreps(lmax_out), f_in=f_in, f_out=f_out)

  def forward(self, x):
    return self.lin(x)

@compile_mode("script")
class GraphConvBlock(torch.nn.Module):
  def __init__(self, gconv, act=None, pool=None) -> None:
    super().__init__()
    self.gconv = gconv
    self.act = act
    self.pool = pool

  def forward(self, data) -> torch.Tensor:
    node_features = self.gconv(data)

    if self.act is not None:
      node_features = self.act(node_features)
    data.x = node_features

    if self.pool is not None:
      return self.pool(data)
    else: # Return the non-modified graph if not pooling
      return data
