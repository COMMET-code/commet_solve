import pytest
from pathlib import Path
import os
import shutil

RUN_COMMET = os.environ['RUN_COMMET']


def remove_if_exists(file_or_dir_to_remove: str):
    path = Path(file_or_dir_to_remove)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(file_or_dir_to_remove)
        else:
            os.remove(path)


def test_run_using_torchscript():

    current_file_path = str(Path(__file__).resolve())
    file_dir = "/".join(current_file_path.split("/")[:-1])
    inp_dir = file_dir+"/resources"

    result = os.system(f"cd {inp_dir} && conda run -n default python torchscript_models.py")
    assert result == 0, "Script for generating torchscript files did not run successfully."

    result = os.system(f"cd {inp_dir} && {RUN_COMMET} using_torchscript.jsonc")

    remove_if_exists(f"{inp_dir}/field_data")
    remove_if_exists(f"{inp_dir}/mesh_data")
    remove_if_exists(f"{inp_dir}/commet_solver.log")
    remove_if_exists(f"{inp_dir}/compute_times.json")
    remove_if_exists(f"{inp_dir}/problem_size.json")

    assert result == 0


