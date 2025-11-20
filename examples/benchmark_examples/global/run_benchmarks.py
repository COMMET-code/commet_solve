import os
from tqdm import tqdm

input_dir = "./examples/benchmark_examples/global/input"
output_dir = "./examples/benchmark_examples/global/output"


def get_files_in_dir(dir_path: str):
    return next(os.walk(dir_path))[2]


input_file_paths = get_files_in_dir(input_dir)
existing_output_file_paths = get_files_in_dir(output_dir)

# |%%--%%| <QCRSwevu2a|F8xUrZOBpm>

files_to_run = [input_file for input_file in input_file_paths
                if input_file not in existing_output_file_paths]



#|%%--%%| <F8xUrZOBpm|keMU5J1nbC>

for file_name in (prog_bar := tqdm(files_to_run)):
    prog_bar.set_description(f"{file_name=}")
    os.system(f"./build/benchmark/ncm_vectorization_benchmark {input_dir}/{file_name} --benchmark_out={output_dir}/{file_name} --benchmark_time_unit=ms --benchmark_filter=BM_globally_vectorized >/dev/null 2>&1")


