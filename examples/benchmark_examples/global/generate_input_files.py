import os
import json


def write_global_inp_file(n_material_points: int,
                          pth_to_torchscript: str):

    if "optimized" not in pth_to_torchscript and "unoptimized" not in pth_to_torchscript:
        raise Exception("Torchscript file")

    optimized = "optimized" in pth_to_torchscript and "unoptimized" not in pth_to_torchscript

    filename = pth_to_torchscript.split("/")[-1].split(".")[0]
    filename = f"npts_{n_material_points}_{filename}"

    json.dump({"n_material_points": n_material_points,
               "ram_monitor_period_microseconds": 5,
               "evaluation_method": "optimized" if optimized else "using_C",
               "path_to_torchscript": pth_to_torchscript},
              open(
                  f"./examples/benchmark_examples/global/input/{filename}.json", 'w')
              )




torchscripts_path = "./examples/benchmark_examples/torchscripts"
torchscript_paths = list(map(lambda filename: f"{
                             torchscripts_path}/{filename}", next(os.walk(torchscripts_path))[2]))

number_of_material_ponits = [2**i for i in range(0, 16, 2)]
print(number_of_material_ponits)

for npts in number_of_material_ponits:
    for pth_to_torchscript in torchscript_paths:
        write_global_inp_file(npts, pth_to_torchscript)



