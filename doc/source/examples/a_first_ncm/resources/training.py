import matplotlib.pyplot as plt
import torch
from torch import nn
from typing import Union, List, Callable, Optional


torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


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

        else:
            self.bias = None

        torch.nn.init.kaiming_uniform_(self.weights, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.mm(x, torch.nn.functional.softplus(self.weights.t()))
        if self.bias is not None:
            result = result + self.bias

        return result

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

    @torch.jit.export
    def W_NN_from_C(self,
                    C: torch. Tensor,
                    structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.inner_network(self.get_K_from_C(C))

    @torch.jit.export
    def W_NN_from_F(self,
                    F: torch. Tensor,
                    structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        C = torch.bmm(F.transpose(1, 2), F)
        return self.W_NN_from_C(C, structural_vectors)

    def forward(self, F: torch.Tensor):
        return self.W_NN_from_F(F)


class GentThomas(nn.Module):

    def __init__(self,
                 c1: float = 1,
                 c2: float = 1,
                 kappa: float = 1):

        super().__init__()

        self.c1: nn.Parameter = nn.Parameter(torch.Tensor([c1]))
        self.c2: nn.Parameter = nn.Parameter(torch.Tensor([c2]))
        self.kappa: nn.Parameter = nn.Parameter(torch.Tensor([kappa]))

    @torch.jit.export
    def W_NN_from_C(self,
                    C: torch. Tensor,
                    structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns the strain energy density for a given right Cauchy-Green tensor and structural vectors.

        Parameters
        ----------
        C : torch.Tensor
            **shape (batch size, 3, 3)** Right Cauchy-Green tensors.
        structural_vectors : torch.Tensor
            **shape (batch size, number of structural vectors, 3)**

        Returns
        -------
        torch.Tensor
            **shape (batch size)** strain energy density

        """
        I3 = torch.det(C)
        I1 = torch.sum(C[:, [0, 1, 2], [0, 1, 2]], dim=1)

        C2 = torch.bmm(C, C)
        I2 = 0.5*(I1 ** 2 - torch.sum(C2[:, [0, 1, 2], [0, 1, 2]], dim=1))

        I1 = I1 * I3 ** (-1./3.)
        I2 = I2 * I3 ** (-2./3.)

        return self.c1*(I1-3) + self.c2*torch.log(I2/3) + 0.5*self.kappa*(I3**(1/2)-1)**2

    @torch.jit.export
    def W_NN_from_F(self,
                    F: torch.Tensor,
                    structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns the strain energy density for a given deformation gradient and structural vectors.

        Parameters
        ----------
        F : torch.Tensor
            **shape (batch size, 3, 3)** Deformation gradients.
        structural_vectors : torch.Tensor
            **shape (batch size, number of structural vectors, 3)**

        Returns
        -------
        torch.Tensor
            **shape (batch size)** strain energy density

        """

        C = torch.bmm(F.transpose(1, 2), F)
        return self.W_NN_from_C(C, structural_vectors)

    def forward(self, F: torch.Tensor, structural_vectors: Optional[torch.Tensor] = None):
        return self.W_NN_from_F(F, structural_vectors)


def get_stress(ncm: Union[InvariantBasedNCM, GentThomas],
               F: torch.Tensor) -> torch.Tensor:
    F = F.detach().requires_grad_(True)
    energy = ncm.W_NN_from_F(F)
    P = torch.autograd.grad(energy,
                            F,
                            torch.ones_like(energy),
                            create_graph=True)[0]
    return P


gent_thomas_model = GentThomas()


dim = 3

F_uni = torch.zeros(100, dim, dim)
for i in range(dim):
    F_uni[:, i, i] = 1

F_uni[:, 0, 0] = torch.linspace(.8, 1.2, F_uni.shape[0])


noise_scale = 0.1
P_data = get_stress(gent_thomas_model, F_uni)
P_data_noisey = P_data + P_data*torch.randn(P_data.shape)*noise_scale


inner_network = MICNN(n_inputs=3,
                      n_outputs=1,
                      hidden_architecture=[16, 16])
ncm = InvariantBasedNCM(inner_network)



optim = torch.optim.Adam(ncm.parameters(), lr=0.5)

for i in range(4000):
    optim.zero_grad()
    P_pred = get_stress(ncm, F_uni)
    loss = torch.nn.functional.mse_loss(P_pred, P_data_noisey.detach())
    print(f"{i=}\t{loss.item()=}")
    loss.backward()
    optim.step()


traced = torch.jit.trace(ncm, (F_uni, ))
traced.save("ncm.torchscript")

P_pred_uni = get_stress(ncm, F_uni)


plt.plot(F_uni[:, 0, 0].detach().cpu(), P_data[:, 0, 0].detach().cpu(), label="Ground truth")
plt.plot(F_uni[:, 0, 0].detach().cpu(), P_data_noisey[:, 0, 0].detach().cpu(), label="Noisey training data", marker='s', linewidth=0)
plt.plot(F_uni[:, 0, 0].detach().cpu(), P_pred_uni[:, 0, 0].detach().cpu(), label="Prediction")

plt.xlabel("$F_{11}$ [mm/mm]")
plt.ylabel("$P_{11}$ [kPa]")
plt.grid()
plt.legend()
plt.savefig("result.png")
