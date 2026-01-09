import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import h5py
import cv2
from typing import List, Tuple, Dict, Union, Optional
import models as m

#|%%--%%| <z9BeIrwrZl|6KvMMphUX4>

def read_uniaxial_data():
    df = pd.read_csv("./brain_data/cortex/uniaxial.csv")
    dim = 3

    size = len(df)

    F_uni = torch.zeros(size, dim, dim)
    F_uni[:, 0, 0] = torch.Tensor(df["F11"].to_numpy())
    F_uni[:, 1, 1] = 1/torch.sqrt(F_uni[:, 0, 0])
    F_uni[:, 2, 2] = F_uni[:, 1, 1]

    P_uni = torch.zeros(size, dim, dim)
    P_uni[:, 0, 0] = torch.Tensor(df["P11"].to_numpy())
    return F_uni, P_uni

def read_shear_data():
    df = pd.read_csv("./brain_data/cortex/shear.csv")
    dim = 3

    size = len(df)

    F_shear = torch.zeros(size, dim, dim)
    for i in range(dim):
        F_shear[:, i, i] = 1
    F_shear[:, 0, 1] = torch.Tensor(df["F12"].to_numpy())


    P_shear = torch.zeros(size, dim, dim)
    P_shear[:, 0, 1] = torch.Tensor(df["P12"].to_numpy())
    return F_shear, P_shear

def get_stress(ncm: m.InvariantBasedNCM,
               F: torch.Tensor):
    F.requires_grad_(True)
    energy = ncm.W_NN_from_F(F)
    P = torch.autograd.grad(energy,
                            F,
                            torch.ones_like(energy),
                            create_graph=True)[0]
    return P


#|%%--%%| <6KvMMphUX4|KSL7nbDM4b>

F_shear, P_shear = read_shear_data()
F_uni, P_uni = read_uniaxial_data()

F_data = torch.concat([F_uni, F_shear])
P_data = torch.concat([P_uni, P_shear])

#|%%--%%| <KSL7nbDM4b|WiTmPdcfOw>

# inner_network = m.MICNN(n_inputs=3,
#                         n_outputs=1,
#                         hidden_architecture=[3, 3],
#                         activation_function=lambda x: torch.square(torch.nn.functional.softplus(x))
#                         )
# ncm = m.InvariantBasedNCM(inner_network)

inner_network = m.ICNN(n_inputs=7,
                        n_outputs=1,
                        # hidden_architecture=[10, 10, 10],
                        hidden_architecture=[30, 30, 30],
                        # activation_function=lambda x: torch.square(torch.nn.functional.softplus(x))
                        )
ncm = m.SSVBasedNCM(inner_network)

#|%%--%%| <WiTmPdcfOw|3cDebt8kYn>

optim = torch.optim.Adam(ncm.parameters(),
                         lr=0.1,
                         weight_decay=0.0001,
                         )


#|%%--%%| <3cDebt8kYn|x2grBlCftl>

for i in range(1000):

    optim.zero_grad()
    # P_pred = get_stress(ncm, F_data)
    P_pred = ncm.get_P(F_data)
    loss = torch.nn.functional.mse_loss(P_pred, P_data)
    print(f"{i=}\t{loss.item()=}")
    loss.backward()
    optim.step()


#|%%--%%| <x2grBlCftl|OGnbIhbTLu>

# P_pred_uni = get_stress(ncm, F_uni)
# P_pred_shear = get_stress(ncm, F_shear)

P_pred_uni = ncm.get_P(F_uni)
P_pred_shear = ncm.get_P(F_shear)

plt.plot(F_uni[:, 0, 0].detach().cpu(), P_uni[:, 0, 0].detach().cpu(), label="Data")
plt.plot(F_uni[:, 0, 0].detach().cpu(), P_pred_uni[:, 0, 0].detach().cpu(), label="Prediction")
plt.legend()

plt.figure()
plt.plot(F_shear[:, 0, 1].detach().cpu(), P_shear[:, 0, 1].detach().cpu(), label="Data")
plt.plot(F_shear[:, 0, 1].detach().cpu(), P_pred_shear[:, 0, 1].detach().cpu(), label="Prediction")
plt.legend()
