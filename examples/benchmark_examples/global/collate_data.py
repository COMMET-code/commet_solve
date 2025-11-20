import os
from tqdm import tqdm
import pandas as pd
import json

benchmark_dir = "./examples/benchmark_examples/global"
output_dir = f"{benchmark_dir}/output"


# |%%--%%| <vz7q9BkUKQ|NWiTp3QfAF>

def get_files_in_dir(dir_path: str):
    return next(os.walk(dir_path))[2]


def file_to_row(pth_to_file: str):
    # Expected file name schema:
    # 'npts_<number of poinst>_<unoptimized/optimized>_depth_<depth>_width_<width>_nparam_<number of parameters>.json'
    file_name = pth_to_file.split("/")[-1]
    file_name_no_extension = file_name.split(".")[0]
    name_data = file_name_no_extension.split("_")

    n_points = int(name_data[1])
    optimized = name_data[2] == "optimized"
    depth = int(name_data[4])
    width = int(name_data[6])
    n_params = int(name_data[8])

    data = json.load(
        open(f"{output_dir}/{existing_output_file_paths[0]}", 'r')
    )
    if len(data["benchmarks"]) != 1:
        raise Exception(f"There should only be one benchmark run but {
                        len(data['benchmarks'])=}")

    data = data["benchmarks"][0]

    return {
        "optimized": optimized,
        "n_points": n_points,
        "depth": depth,
        "width": width,
        "n_params": n_params,
        "time_ms": data['cpu_time'],
        "pre_setup_ram_usage_bytes": data['pre_setup_ram_usage'],
        "max_setup_ram_usage_bytes": data['max_setup_ram_usage'],
        "setup_ram_usage_delta_bytes": data['max_setup_ram_usage']-data['pre_setup_ram_usage'],
        "pre_compute_ram_usage_bytes": data['pre_compute_ram_usage'],
        "max_compute_ram_usage_bytes": data['max_compute_ram_usage'],
        "compute_ram_usage_delta_bytes": data['max_compute_ram_usage'] - data['pre_compute_ram_usage']
    }


existing_output_file_paths = get_files_in_dir(output_dir)


lines = [file_to_row(f"{output_dir}/{file}")
         for file in existing_output_file_paths]
pd.DataFrame(lines).to_csv(f"{benchmark_dir}/collated_data.csv")
