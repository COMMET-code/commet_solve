import torch
from torch import nn
from typing import List, Callable, Optional
from abc import ABC

class TwoBranchMLP(nn.Module):
    def __init__(self,
                 branch_1_layers: List[int],
                 branch_2_layers: List[int],
                 head_layers: List[int]
                 ):
        """A generalized version of a two branch MLP

        Note BA to SC: In Rui's work we ultimately use this to predict the (M)ICNN weights from the microstructure
        descriptors and behaviour of microstructure constituents.
        We will need to change this for your case -- i.e. to match way you are able to describe the
        fibre networks (statistical distributions for fibre length, valency etc).

        Parameters
        ----------
        branch_1_layers : List[int]
            [TODO:description]
        branch_2_layers : List[int]
            [TODO:description]
        head_layers : List[int]
            [TODO:description]

        """
        super(TwoBranchMLP, self).__init__()

        self.branch_1_in_size = branch_1_layers[0]
        self.branch_2_in_size = branch_2_layers[0]
        self.head_inp_size = branch_1_layers[-1] + branch_2_layers[-1]
        self.head_out_size = head_layers[-1]

        head_layers = [self.head_inp_size] + head_layers
        hidden_head_layers = head_layers[:-1]

        self.branch_1 = nn.Sequential()
        for layer_in, layer_out in zip(branch_1_layers[:-1], branch_1_layers[1:]):
            self.branch_1.append(nn.Linear(layer_in, layer_out))
            self.branch_1.append(nn.ReLU())

        self.branch_2 = nn.Sequential()
        for layer_in, layer_out in zip(branch_2_layers[:-1], branch_2_layers[1:]):
            self.branch_2.append(nn.Linear(layer_in, layer_out))
            self.branch_2.append(nn.ReLU())


        self.head = nn.Sequential()
        for layer_in, layer_out in zip(hidden_head_layers[:-1], hidden_head_layers[1:]):
            self.head.append(nn.Linear(layer_in, layer_out))
            self.head.append(nn.ReLU())

        self.head.append(nn.Linear(hidden_head_layers[-1], self.head_out_size))

    def forward(self,
                branch_1_inp: torch.Tensor,
                branch_2_inp: Optional[torch.Tensor] = None):

        if branch_2_inp is None: # Assume both inputs are in `branch_1_inp` as a concatenated tensor
            b1 = self.branch_1(branch_1_inp[:, :self.branch_1_in_size])
            b2 = self.branch_2(branch_1_inp[:, self.branch_1_in_size:])
        else:
            b1 = self.branch_1(branch_1_inp)
            b2 = self.branch_2(branch_2_inp)


        return self.head(torch.cat([b1, b2], dim=1))

class ParameterPredictor(nn.Module, ABC):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 bias: bool = True):
        """Base class for an nn.Module that outputs matrix of weights (out_size x in_size) and biases (out_size)
        Inheriting classes must implement the `forward` method so that the weights are returned, and if `bias` is set to `true`, a bias vector is also returned


        Note BA to SC: In Rui's work we ultimately use this to predict the (M)ICNN weights from the microstructure
        descriptors and behaviour of microstructure constituents.
        We will need to change this for your case -- i.e. to match way you are able to describe the
        fibre networks (statistical distributions for fibre length, valency etc).


        Parameters
        ----------
        in_size : int
            [TODO:description]
        out_size : int
            [TODO:description]
        bias : bool
            [TODO:description]

        """
        super(ParameterPredictor, self).__init__()

        self.in_size: int = in_size
        self.out_size: int = out_size
        self.bias: bool = bias

class TwoHeadMLPParameterPredictor(ParameterPredictor):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 descriptor_size: int,
                 micro_param_size: int,
                 descriptor_branch_hidden_layers: List[int],
                 micro_param_branch_hidden_layers: List[int],
                 head_hidden_layers: List[int],
                 bias: bool = True):
        """A parameter predictor to be used in a hypernetwork.
        In this parameter predictor, a TwoHeadMLP is used to predict the parameters for a given layer.

        Parameters
        ----------
        in_size : int
            [TODO:description]
        out_size : int
            [TODO:description]
        descriptor_size : int
            [TODO:description]
        micro_param_size : int
            [TODO:description]
        descriptor_branch_hidden_layers : List[int]
            [TODO:description]
        micro_param_branch_hidden_layers : List[int]
            [TODO:description]
        head_hidden_layers : List[int]
            [TODO:description]
        bias : bool
            [TODO:description]

        """
        super(TwoHeadMLPParameterPredictor, self).__init__(in_size, out_size, bias)

        self.mlp_out_size = in_size*out_size
        if self.bias: 
            self.mlp_out_size += out_size

        self.two_branch_mlp = TwoBranchMLP([descriptor_size] + descriptor_branch_hidden_layers,
                                           [micro_param_size] + micro_param_branch_hidden_layers,
                                           head_hidden_layers + [self.mlp_out_size])

    def forward(self,
                descriptors: torch.Tensor,
                micro_params: torch.Tensor):
        """Method for obtaining the weights and bias for a given layer from the descriptors and micro behaviour

        Parameters
        ----------
        descriptors : torch.Tensor
            size: [n_batch, descriptor_size]
        micro_params : torch.Tensor
            size: [n_batch, micro_param_size]

        Returns
        -------
        [TODO:return]
            size: [n_batch, size_out, size_in]
        [TODO:return]
            size: [n_batch, size_out]

        """
        param_predition = self.two_branch_mlp(descriptors, micro_params)
        
        if self.bias:
            bias = param_predition[:, -self.out_size:]
            weights = param_predition[:, :-self.out_size].reshape([-1, self.out_size, self.in_size])
            return weights, bias 
        else:
            return param_predition.reshape([-1, 
                                            self.out_size,
                                            self.in_size])

class Hyperlayer(nn.Module):
    def __init__(self,
                 param_predictor: ParameterPredictor,
                 weight_operator: Callable = lambda x: x):
        """Uses the parameters obtained by an instance of a ParameterPredictor and applies them.
        Optionally, may also apply an activation function to the weights (this can be used to make them positive, which can be useful for enforcing convexity and/or monotonicity.)

        Parameters
        ----------
        param_predictor : ParameterPredictor
            [TODO:description]
        weight_operator : Callable
            [TODO:description]

        """
        super(Hyperlayer, self).__init__()
        self.param_predictor: ParameterPredictor = param_predictor
        self.weight_operator: Callable = weight_operator 

    def forward(self,
                descriptors: torch.Tensor,
                micro_params: torch.Tensor,
                x: torch.Tensor):

        if self.param_predictor.bias:
            weights, bias = self.param_predictor(descriptors, micro_params)
            w_times_x = torch.einsum("bij,bj->bi",
                                     self.weight_operator(weights),
                                     x)
            return w_times_x + bias 

        else:
            weights = self.param_predictor(descriptors, micro_params)

            w_times_x = torch.einsum("bij,bj->bi",
                                     self.weight_operator(weights),
                                     x)
            return w_times_x 

class ConvexHyperlayer(Hyperlayer):
    def __init__(self,
                 param_predictor: ParameterPredictor):

        super().__init__(param_predictor,
                         weight_operator=torch.nn.functional.softplus)

class HyperAnisotropicLaplaceMaterial(nn.Module):
    """[TODO:description]


    Note BA to SC: This is specifically for constitutive behaviour in the context of Laplace equation-like 
    problems (eg. heat/concentration diffusion).
    So this isn't really relevant to you.


    Attributes
    ----------
    dim : [TODO:attribute]
    activation : [TODO:attribute]
    first_hyperlayer : [TODO:attribute]
    hyperlayers : [TODO:attribute]
    skip_hyperlayers : [TODO:attribute]
    last_hyperlayer : [TODO:attribute]
    last_skip_hyperlayer : [TODO:attribute]

    """
    def __init__(self,
                 descriptor_size: int,
                 micro_param_size: int,
                 hidden_architecture: List[int],
                 activation: Callable = torch.nn.functional.softplus):
        super(HyperAnisotropicLaplaceMaterial, self).__init__()
        self.dim = 3 
        self.activation: Callable = activation


        create_param_predictor = lambda in_size, out_size: TwoHeadMLPParameterPredictor(in_size=in_size,
                                                                          out_size=out_size,
                                                                          descriptor_size=descriptor_size,
                                                                          micro_param_size=micro_param_size,
                                                                          descriptor_branch_hidden_layers=[64, 32],
                                                                          micro_param_branch_hidden_layers=[64, 32],
                                                                          head_hidden_layers=[64, 32]
                                                                          )

        self.first_hyperlayer = Hyperlayer(create_param_predictor(self.dim, hidden_architecture[0]))

        self.hyperlayers = nn.ModuleList()
        self.skip_hyperlayers = nn.ModuleList()
        for inp, out in zip(hidden_architecture[:-1], hidden_architecture[1:]):
            self.hyperlayers.append(ConvexHyperlayer(create_param_predictor(inp, out))) 
            self.skip_hyperlayers.append(Hyperlayer(create_param_predictor(hidden_architecture[0], out))) 

        self.last_hyperlayer = ConvexHyperlayer(create_param_predictor(hidden_architecture[-1], 1))
        self.last_skip_hyperlayer = ConvexHyperlayer(create_param_predictor(hidden_architecture[0], 1))

    def forward(self,
                descriptors: torch.Tensor,
                micro_params: torch.Tensor,
                grad_u: torch.Tensor):
        x = self.first_hyperlayer(descriptors, micro_params, grad_u)
        x1 = x.clone()

        for layer, skip in zip(self.hyperlayers, self.skip_hyperlayers):
            x = self.activation(layer(descriptors, micro_params, x) + skip(descriptors, micro_params, x1))

        return self.last_hyperlayer(descriptors, micro_params, x) + self.last_skip_hyperlayer(descriptors, micro_params, x1)

    def flux(self,
             descriptors: torch.Tensor,
             micro_params: torch.Tensor,
             grad_u: torch.Tensor): 
        ref_grad_u = torch.zeros_like(grad_u)
        ref_grad_u.requires_grad_(True)
        
        psi_0 = self(descriptors, micro_params, ref_grad_u)
        flux_0 = torch.autograd.grad(psi_0,
                                     ref_grad_u,
                                     torch.ones_like(psi_0))[0]

        psi_nn = self(descriptors, micro_params, grad_u) 
        flux_nn = torch.autograd.grad(psi_nn,
                                      grad_u,
                                      torch.ones_like(psi_nn))[0]


        return flux_nn - flux_0

class IsotropicHyperelasticityHypernetwork(nn.Module):
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

    def __init__(self,
                 descriptor_size: int,
                 micro_param_size: int,
                 hidden_architecture: List[int],
                 activation: Callable = torch.nn.functional.softplus):

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
        super(IsotropicHyperelasticityHypernetwork, self).__init__()
        self.dim = 3 
        self.activation: Callable = activation


        create_param_predictor = lambda in_size, out_size: TwoHeadMLPParameterPredictor(in_size=in_size,
                                                                          out_size=out_size,
                                                                          descriptor_size=descriptor_size,
                                                                          micro_param_size=micro_param_size,
                                                                          descriptor_branch_hidden_layers=[64, 32],
                                                                          micro_param_branch_hidden_layers=[64, 32],
                                                                          head_hidden_layers=[64, 32]
                                                                          )

        self.first_hyperlayer = Hyperlayer(create_param_predictor(self.dim*2+1,
                                                                  hidden_architecture[0]))

        self.hyperlayers = nn.ModuleList()
        self.skip_hyperlayers = nn.ModuleList()
        for inp, out in zip(hidden_architecture[:-1], hidden_architecture[1:]):
            self.hyperlayers.append(ConvexHyperlayer(create_param_predictor(inp,
                                                                            out))) 
            self.skip_hyperlayers.append(Hyperlayer(create_param_predictor(hidden_architecture[0],
                                                                           out))) 

        self.last_hyperlayer = ConvexHyperlayer(create_param_predictor(hidden_architecture[-1], 1))
        self.last_skip_hyperlayer = ConvexHyperlayer(create_param_predictor(hidden_architecture[0], 1))
        pass

    @staticmethod
    def get_augmented_lambdas(lambdas: torch.Tensor):

        adjoint_terms = torch.stack([
            lambdas[:, 0] * lambdas[:, 1],
            lambdas[:, 1] * lambdas[:, 2],
            lambdas[:, 2] * lambdas[:, 0] ],
                                    dim=1) 

        return torch.concat(
            [lambdas, adjoint_terms, torch.prod(lambdas, dim=1).unsqueeze(1)],
            dim=1
        )

    def forward(self,
                descriptors: torch.Tensor,
                micro_params: torch.Tensor,
                lambdas: torch.Tensor):
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
            lambdas[:, 2] * lambdas[:, 0] ],
                                    dim=1) 


        lambda_input = torch.concat(
            [lambdas, adjoint_terms, torch.prod(lambdas, dim=1).unsqueeze(1)],
            dim=1
        )

        x = self.first_hyperlayer(descriptors, micro_params, lambda_input)
        x1 = x.clone()

        for layer, skip in zip(self.hyperlayers, self.skip_hyperlayers):
            x = self.activation(layer(descriptors, micro_params, x) + skip(descriptors, micro_params, x1))

        return self.last_hyperlayer(descriptors, micro_params, x) + self.last_skip_hyperlayer(descriptors, micro_params, x1)

    def energy_NN_from_lambdas(self,
                  descriptors: torch.Tensor,
                  micro_params: torch.Tensor,
                  lambdas: torch.Tensor):

        result = torch.zeros([lambdas.shape[0], 1])
        for permutation_function in self.LAMBDA_PERMUTATIONS:
            for sign_function in self.LAMBDA_SIGNS:
                result = result + self(descriptors, micro_params, permutation_function(sign_function(lambdas)))

        return result/24

    def energy_NN_from_F(self,
                  descriptors: torch.Tensor,
                  micro_params: torch.Tensor,
                  F: torch.Tensor):

        C = torch.bmm(F.transpose(1,2), F)
        lambdas = torch.linalg.eigvalsh(C)**0.5
        return self.energy_NN_from_lambdas(descriptors, micro_params, lambdas)

    def get_P(self,
              descriptors: torch.Tensor,
              micro_params: torch.Tensor,
              F: torch.Tensor):

        C = torch.bmm(F.transpose(1,2), F)
        lambdas = torch.linalg.eigvalsh(C)**0.5
        psi_nn = self.energy_NN_from_lambdas(descriptors, micro_params, lambdas)

        # We calculate `c_s` to enforce P=0 at F=I
        lambdas_ones = torch.ones_like(lambdas)
        lambdas_ones.requires_grad_(True)
        psi_0 = self.energy_NN_from_lambdas(descriptors, micro_params, lambdas_ones)

        c_s = -torch.autograd.grad(psi_0, lambdas_ones, torch.ones_like(psi_0))[0]

        psi = psi_nn + torch.sum(c_s*lambdas, dim=1)

        return torch.autograd.grad(psi, F, torch.ones_like(psi))[0]

if __name__ == "__main__":

    des_size = 64
    micro_behave_size = 2 
    hidden_arch = [16, 8, 4]
    mat = IsotropicHyperelasticityHypernetwork(des_size,
                                               micro_behave_size,
                                               hidden_arch)

    n_batch = 100
    n_dim = 3

    F = torch.zeros([n_batch, n_dim, n_dim])
    for i in range(3):
        F[:, i, i] = 1

    F[:, 0, 0] = torch.linspace(.1, 2, n_batch)
    F.requires_grad_(True)

    desc = torch.zeros([n_batch, des_size])
    micro_behave = torch.zeros([n_batch, micro_behave_size])

    P = mat.get_P(desc, micro_behave, F)

    print(f"{F=}\n{P=}")
