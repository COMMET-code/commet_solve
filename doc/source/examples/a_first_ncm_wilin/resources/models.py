import torch
from torch import nn

from typing import List, Callable, Optional


class ConvexLinear(nn.Module):

    def __init__(self,
                 size_in: int,
                 size_out: int,
                 use_bias=True):
        super(ConvexLinear, self).__init__()
        self.size_in: int = size_in
        self.size_out: int = size_out
        weights: torch.Tensor = torch.Tensor(size_out, size_in)
        self.weights = torch.nn.Parameter(weights)

        self.use_bias = use_bias

        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(size_out))

        torch.nn.init.kaiming_uniform_(self.weights, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.mm(x, torch.nn.functional.softplus(self.weights.t()))
        if self.use_bias:
            return w_times_x + self.bias
        else:
            return w_times_x


class MICNN(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 hidden_architecture: List[int] = [],
                 activation_function: Callable = torch.nn.functional.softplus):
        super(MICNN, self).__init__()

        self.n_inputs: int = n_inputs
        self.n_outputs: int = n_outputs
        self.activation_function: Callable = activation_function

        self.architecture = [self.n_inputs] + \
            hidden_architecture + [self.n_outputs]

        self.n_hidden_layers = len(hidden_architecture) - 1

        self.first_layer = ConvexLinear(self.n_inputs, self.architecture[1])

        self.layers = torch.nn.ModuleList([ConvexLinear(hidden_architecture[i],
                                                        hidden_architecture[i+1])
                                           for i in range(self.n_hidden_layers)])
        self.last_layer = ConvexLinear(self.architecture[-2], self.n_outputs)

        self.skip_layers = torch.nn.ModuleList([ConvexLinear(self.n_inputs,
                                                             hidden_architecture[i+1],
                                                             use_bias=False)
                                                for i in range(self.n_hidden_layers)])
        self.last_skip = ConvexLinear(self.n_inputs, self.n_outputs)

    def forward(self, x: torch.Tensor):
        z = self.first_layer(x)

        for layer, skip_layer in zip(self.layers, self.skip_layers):
            z = self.activation_function(layer(z) + skip_layer(x))

        z = self.last_layer(z) + self.last_skip(x)

        return z

class ICNN(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 hidden_architecture: List[int] = [],
                 activation_function: Callable = torch.nn.functional.softplus):
        super().__init__()

        self.n_inputs: int = n_inputs
        self.n_outputs: int = n_outputs
        self.activation_function: Callable = activation_function

        self.architecture = [self.n_inputs] + \
            hidden_architecture + [self.n_outputs]

        self.n_hidden_layers = len(hidden_architecture) - 1


        self.layers = torch.nn.ModuleList([ConvexLinear(hidden_architecture[i],
                                                        hidden_architecture[i+1])
                                           for i in range(self.n_hidden_layers)])
        self.last_layer = ConvexLinear(self.architecture[-2], self.n_outputs)

        self.first_layer = torch.nn.Linear(self.n_inputs, self.architecture[1])
        self.skip_layers = torch.nn.ModuleList([torch.nn.Linear(self.n_inputs,
                                                                hidden_architecture[i+1])
                                                for i in range(self.n_hidden_layers)])
        self.last_skip = torch.nn.Linear(self.n_inputs, self.n_outputs)

    def forward(self, x: torch.Tensor):
        z = self.first_layer(x)

        for layer, skip_layer in zip(self.layers, self.skip_layers):
            z = self.activation_function(layer(z) + skip_layer(x))

        z = self.last_layer(z) + self.last_skip(x)

        return z


class InvariantBasedNCM(nn.Module):
    def __init__(self,
                 inner_network: nn.Module):
        super(InvariantBasedNCM, self).__init__()
        self.inner_network: nn.Module = inner_network

    def get_K_from_C(self, C: torch.Tensor) -> torch.Tensor:

        C = 0.5*(C + C.transpose(1, 2))

        I3 = torch.det(C)
        C = C*I3[:, None, None]**(-1/3)

        I1 = torch.sum(C[:, [0, 1, 2], [0, 1, 2]], dim=1)

        C2 = torch.bmm(C, C)
        I2 = 0.5*(I1**2 - torch.sum(C2[:, [0, 1, 2], [0, 1, 2]], dim=[1]))

        J = torch.sqrt(I3)

        return torch.stack([
            I1-3,
            I2 ** (3/2) - (3 ** (3/2)),
            (J-1) ** 2
        ],
            dim=1)

    def W_NN_from_C(self,
                    C: torch. Tensor,
                    structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.inner_network(self.get_K_from_C(C))

    def W_NN_from_F(self,
                    F: torch. Tensor,
                    structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        C = torch.bmm(F.transpose(1, 2), F)
        return self.W_NN_from_C(C, structural_vectors)

    def forward(self, F: torch.Tensor):
        return self.W_NN_from_F(F)


class SSVBasedNCM(nn.Module):
    # These permutations are used to enforce that the behaviour is in SO(3)
    # See:
    #     6
    #     Input convex neural networks: universal approximation theorem and
    #     implementation for isotropic polyconvex hyperelastic energies
    LAMBDA_PERMUTATIONS = [
        lambda x: x[:, [0, 1, 2]],
        lambda x: x[:, [0, 2, 1]],
        lambda x: x[:, [1, 0, 2]],
        lambda x: x[:, [2, 0, 1]],
        lambda x: x[:, [2, 1, 0]],
        lambda x: x[:, [1, 2, 0]],
    ]

    LAMBDA_SIGNS = [
        lambda x: x*torch.Tensor([1, 1, 1]).unsqueeze(0),
        lambda x: x*torch.Tensor([-1, -1, 1]).unsqueeze(0),
        lambda x: x*torch.Tensor([-1, 1, -1]).unsqueeze(0),
        lambda x: x*torch.Tensor([1, -1, -1]).unsqueeze(0),
    ]

    def __init__(self, inner_network: nn.Module):
        """I use a principal stretch based formulation because this generalizes better
        That is, the invariant formulations are always special cases of principal stretch formulations.

        Parameters
        ----------
        descriptor_size : int
            [TODO:description]
        micro_param_size : int
            [TODO:description]
        hidden_architecture : List[int]
            [TODO:description]
        activation : Callable
            [TODO:description]

        """
        super(SSVBasedNCM, self).__init__()
        self.dim = 3
        self.inner_network: nn.Module = inner_network

    @staticmethod
    def get_augmented_lambdas(lambdas: torch.Tensor):

        adjoint_terms = torch.stack([
            lambdas[:, 0] * lambdas[:, 1],
            lambdas[:, 1] * lambdas[:, 2],
            lambdas[:, 2] * lambdas[:, 0]],
            dim=1)

        return torch.concat(
            [lambdas, adjoint_terms, torch.prod(lambdas, dim=1).unsqueeze(1)],
            dim=1
        )

    def forward(self, lambdas: torch.Tensor):
        """Get energy -- I (Ben) don't really use this TBH. Instead I use `energy_NN_from_...()` and `get_P()`

        Parameters
        ----------
        descriptors : torch.Tensor
            [TODO:description]
        micro_params : torch.Tensor
            [TODO:description]
        F : torch.Tensor
            [TODO:description]
        """

        adjoint_terms = torch.stack([
            lambdas[:, 0] * lambdas[:, 1],
            lambdas[:, 1] * lambdas[:, 2],
            lambdas[:, 2] * lambdas[:, 0]],
            dim=1)

        lambda_input = torch.concat(
            [lambdas, adjoint_terms, torch.prod(lambdas, dim=1).unsqueeze(1)],
            dim=1
        )

        return self.inner_network(lambda_input)

    def W_NN_from_lambdas(self, lambdas: torch.Tensor) -> torch.Tensor:

        result = torch.zeros([lambdas.shape[0], 1])
        for permutation_function in self.LAMBDA_PERMUTATIONS:
            for sign_function in self.LAMBDA_SIGNS:
                result = result + \
                    self(permutation_function(sign_function(lambdas)))

        return result/24

    def W_NN_from_C(self,
                    C: torch.Tensor,
                    structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:

        C = 0.5 * (C + C.transpose(1, 2))

        lambdas = torch.linalg.eigvalsh(C)**0.5
        return self.W_NN_from_lambdas(lambdas)

    def W_NN_from_F(self, F: torch.Tensor,
                    structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:

        C = torch.bmm(F.transpose(1, 2), F)
        return self.W_NN_from_C(C, structural_vectors)

    def get_P(self,
              F: torch.Tensor,
              create_graph: bool = True):

        F.requires_grad_(True)
        C = torch.bmm(F.transpose(1, 2), F)
        C = 0.5 * (C + C.transpose(1, 2))

        lambdas = torch.linalg.eigvalsh(C)**0.5
        psi_nn = self.W_NN_from_lambdas(lambdas)

        # We calculate `c_s` to enforce P=0 at F=I
        lambdas_ones = torch.ones_like(lambdas)
        lambdas_ones.requires_grad_(True)
        psi_0 = self.W_NN_from_lambdas(lambdas_ones)

        c_s = -torch.autograd.grad(psi_0, lambdas_ones,
                                   torch.ones_like(psi_0))[0]

        psi = psi_nn + torch.sum(c_s*lambdas, dim=1)

        return torch.autograd.grad(psi, F, torch.ones_like(psi), create_graph=create_graph)[0]


if __name__ == "__main__":

    # micro_behave_size = 2
    # hidden_arch = [16, 8, 4]

    inner_network = ICNN(n_inputs=7,
                          n_outputs=1,
                          hidden_architecture=[16, 16, 16])

    mat = SSVBasedNCM(inner_network)

    n_batch = 100
    n_dim = 3

    F = torch.zeros([n_batch, n_dim, n_dim])
    for i in range(3):
        F[:, i, i] = 1

    F[:, 0, 0] = torch.linspace(.1, 2, n_batch)
    F.requires_grad_(True)


    P = mat.get_P(F)

    print(f"{F=}\n{P=}")


# def main():

#     inner_network = MICNN(n_inputs=3,
#                           n_outputs=1,
#                           hidden_architecture=[16, 16, 16])

#     ncm = InvariantBasedNCM(inner_network)

#     batch_size = 100
#     dim = 3

#     F = torch.zeros(batch_size, dim, dim)
#     for i in range(dim):
#         F[:, i, i] = 1

#     F.requires_grad_(True)
#     energy = ncm.W_NN_from_F(F)
#     P = torch.autograd.grad(energy,
#                             F,
#                             torch.ones_like(energy))
