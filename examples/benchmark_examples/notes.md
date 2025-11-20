# Material point benchmarks

We are going to run some material point benchmarks, i.e. repeatedly evaluating constitutive behaviour
for material (quadrature) points in the absence of an FE solver, to evaluate the effects of vectorization and batching.
We also want to see the effects of NCM model size (i.e. the number of parameters), the effect of CGO vs non-CGO,
and compare an NCM model to a traditional model.

## Generating torchscript files
To see the effects of model size, we generate torchscript files for various widths and depths using the `generate_torchscript.py` 
script. 
This script also generates CGO and non-CGO torchscript files.
All models make use of an MICNN as the inner network.
The files are written to the `torchscripts` directory with names using the following convention

`<unoptimized/optimized>_depth_<depth>_width_<width>_nparam_<number of parameters>.torchscipt`

## Globally vectorized benchmarks
To run the benchmarks, you need to set
`sudo sysctl kernel.perf_event_paranoid=0`
to allow for monitoring of CPU performance counters.
We can then run a benchmark using the command
` ncm_vectorization_benchmark <path to input file> --benchmark_out=<path to output> --benchmark_time_unit=ms --benchmark_filter=<benchmark to run>`
The acceptable values for `benchmark to run` are `BM_globally_vectorized` or `BM_batch_vectorized`.
The input file is a json file with the following structure
```json
{
    "n_material_points": <int>,
    "ram_monitor_period_microseconds": <int>,
    "evaluation_method": <enum: "optimized"|"using_F"|"using_C">,
    "path_to_torchscript": <str>
}
```

The output file is also a json file.

### Input files
We now want to run benchmarks for global vectorization while permuting values for the following parameters:

* Number of material points
* Model size
* CGO vs non-CGO

To do so, we generate input files using the `global/generate_input_files.py` script.
The resulting input files are placed in `globa/input` and have the following naming convention
`npts_<number of material points>_<unoptimized/optimized>_depth_<depth>_width_<width>_nparam_<number of parameters>.torchscipt`

### Running benchmarks


## Running batch-vectorized benchmarks
