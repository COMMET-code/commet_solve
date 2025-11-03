import torch
from torch import nn
from typing import Optional


class NH(nn.Module):
    def __init__(self,
                 mu: float,
                 lam: float):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.Tensor([mu]))
        self.lam = torch.nn.Parameter(torch.Tensor([lam]))

    @torch.jit.export
    def W_NN_from_C(self,
                    C: torch. Tensor,
                    structural_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns the strain energy density for a given right Cauchy-Green tensor and structural vectors.

        Parameters
        ----------
        C : torch.Tensor
            **shape (batch size, 3, 3)** Deformation gradients.
        structural_vectors : torch.Tensor
            **shape (batch size, number of structural vectors, 3)**

        Returns
        -------
        torch.Tensor
            **shape (batch size)** strain energy density

        """
        I1 = torch.sum(C[:, [0, 1, 2], [0, 1, 2]], dim=1)
        I3 = torch.det(C)
        return 0.5 * self.mu * (I1 - 3 - torch.log(I3)) + 0.25 * self.lam * (I3 - 1 - torch.log(I3))

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



if __name__ == "__main__":
    mu: float = 77
    lam: float = 115
    nh: NH = NH(mu, lam)

    dim = 3
    F_mat = torch.zeros(100, dim, dim)

    traced = torch.jit.trace(nh, (F_mat, ))
    traced.save(f"nh.torchscript")
