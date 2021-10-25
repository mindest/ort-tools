# ort-tools

## nvvp file parser

This script parses kernel related info from two `.nvvp` (or `.sqlite`) files of ORT runs with `enable_nvtx_profile` enabled/disabled, respectively.

usage:

```bash
$ python prof_parser.py -h
usage: prof_parser.py [-h] --file1 FILE1 --file2 FILE2 [--nsys] [--onnx ONNX]

optional arguments:
  -h, --help     show this help message and exit
  --file1 FILE1  profiling file (.nvvp or .sqlite) with ORT NVTX enabled
  --file2 FILE2  profiling file (.nvvp or .sqlite) with ORT NVTX disabled
  --nsys         nsys (file .sqlite) or nvprof (file .nvvp)
  --onnx ONNX    onnx file of torch exported model, with call stack info
```

The script will generate three files by default:

- `*_kernel.csv`: with kernel level info in every training step
- `*_op.csv`: with op level info in every training step
- `*_avg.csv`: with op level info averaged over the steps

If using Nsight System, option `--export sqlite` or `--stats true` is needed when running the profiling, to export the profiling data into a `.sqlite` file.
