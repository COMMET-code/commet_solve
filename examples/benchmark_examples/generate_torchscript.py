import torch
from commet import discover as di
device = "cuda" if torch.cuda.is_available() else "cpu"

# |%%--%%| <9vTFerwKFn|qparA6vHkA>

dim = 3
width = 16
depth = 3

n_structural_vectors = 0

F_for_unopt = torch.zeros(100, dim, dim)
for i in range(dim):
    F_for_unopt[:, i, i] = 1

F_for_opt = torch.zeros((10, 3, 3))
for i in range(3):
    F_for_opt[:, i, i] = 1


# |%%--%%| <qparA6vHkA|L4hWmG0PsO>

for width in [2, 4, 8, 16, 32, 64]:
    for depth in [2, 3, 4]:

        invariants_layer = di.PolyconvexInvariants(
            n_structural_tensors=n_structural_vectors).to(device)

        inner_network = di.MICNN(n_inputs=invariants_layer.n_Ks,
                                 hidden_architecture=[width]*depth, n_outputs=1).to(device)

        model = di.HyperelasticNN(invariants_layer, inner_network).to(device)
        n_params = sum(param.numel() for param in model.parameters())

        traced_model = torch.jit.trace(
            model, (F_for_unopt, torch.zeros(F_for_unopt.shape[0], 0, dim)))
        traced_model.save(
            f"./examples/benchmark_examples/torchscripts/unoptimized_depth_{depth}_width_{width}_nparam_{n_params}.torchscript")

        optimized_model: di.OptInvariantsLayer = model.get_opt()
        optimized_model = optimized_model.to(torch.float64)

        traced_optimized = torch.jit.trace(optimized_model, (F_for_opt,))
        traced_optimized.save(
            f"./examples/benchmark_examples/torchscripts/optimized_depth_{depth}_width_{width}_nparam_{n_params}.torchscript")
