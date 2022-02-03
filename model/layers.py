import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import Tensor

class ChebGraphConv(nn.Module):
    def __init__(self, K, in_features, out_features, bias):
        super(ChebGraphConv, self).__init__()
        self.K = K
        self.weight = nn.Parameter(torch.FloatTensor(K, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, gso):
        # Chebyshev polynomials:
        # x_0 = x,
        # x_1 = gso * x,
        # x_k = 2 * gso * x_{k-1} - x_{k-2},
        # where gso = 2 * gso / eigv_max - id.

        cheb_poly_feat = []
        if self.K < 0:
            raise ValueError('ERROR: The order of Chebyshev polynomials shoule be non-negative!')
        elif self.K == 0:
            # x_0 = x
            cheb_poly_feat.append(x)
        elif self.K == 1:
            # x_0 = x
            cheb_poly_feat.append(x)
            if gso.is_sparse:
                # x_1 = gso * x
                cheb_poly_feat.append(torch.sparse.mm(gso, x))
            else:
                if x.is_sparse:
                    x = x.to_dense
                # x_1 = gso * x
                cheb_poly_feat.append(torch.mm(gso, x))
        else:
            # x_0 = x
            cheb_poly_feat.append(x)
            if gso.is_sparse:
                # x_1 = gso * x
                cheb_poly_feat.append(torch.sparse.mm(gso, x))
                # x_k = 2 * gso * x_{k-1} - x_{k-2}
                for k in range(2, self.K):
                    cheb_poly_feat.append(torch.sparse.mm(2 * gso, cheb_poly_feat[k - 1]) - cheb_poly_feat[k - 2])
            else:
                if x.is_sparse:
                    x = x.to_dense
                # x_1 = gso * x
                cheb_poly_feat.append(torch.mm(gso, x))
                # x_k = 2 * gso * x_{k-1} - x_{k-2}
                for k in range(2, self.K):
                    cheb_poly_feat.append(torch.mm(2 * gso, cheb_poly_feat[k - 1]) - cheb_poly_feat[k - 2])
        
        feature = torch.stack(cheb_poly_feat, dim=0)
        if feature.is_sparse:
            feature = feature.to_dense()
        cheb_graph_conv = torch.einsum('bij,bjk->ik', feature, self.weight)

        if self.bias is not None:
            cheb_graph_conv = torch.add(input=cheb_graph_conv, other=self.bias, alpha=1)
        else:
            cheb_graph_conv = cheb_graph_conv

        return cheb_graph_conv

    def extra_repr(self) -> str:
        return 'K={}, in_features={}, out_features={}, bias={}'.format(
            self.K, self.in_features, self.out_features, self.bias is not None
        )