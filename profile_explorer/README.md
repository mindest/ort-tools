# profile_explorer

This file is based on ORT's [profile_explorer.py](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/profile_explorer/profile_explorer.py). It accepts a profiling result file from ORT session run (with session option `enable_profiling` set).

Main help message:

```bash
$ python profile_explorer.py -h
usage: profile_explorer.py [-h] [--demangler DEMANGLER] [--shape-sensitive] [--dimension-sensitive] [--filter FILTER [FILTER ...]] [--csv CSV] [-c COUNT] [--start START] [--end END] [--mapping] [--gather GATHER] input

onnxruntime bench tool

positional arguments:
  input                 Trace input file, formatted as JSON

optional arguments:
  -h, --help            show this help message and exit
  --demangler DEMANGLER
                        The command to use to demangle C++ identifiers
  --shape-sensitive     Perform a shape sensitive analysis of kernel execution times
  --dimension-sensitive
                        Perform a kernel launch dimension sensitive analysis of kernel execution times
  --filter FILTER [FILTER ...]
                        Restrict analysis to the specified identifiers, i.e., specify a filter list. Also supports UNIX-style wildcards.
  --csv CSV             Save data to csv
  -c COUNT, --count COUNT
                        List top N items
  --start START, -s START
                        Index of the first model run to process (starting from 0, supports negative indices). Defaults to 1 to skip the first run (run 0), which is often a warmup step.
  --end END, -e END     Index of the last model run to process (exclusive, supports negative indices). Defaults to None, which means all runs starting from --start will be included.
  --mapping, -m         Whether dump op-kernel correlation
  --gather GATHER, -g GATHER
                        Accepts a python expression that will be evaluated as a list, which will be used to gather model runs.
```

The grouping of the tracing events heavily relies on the `model_run` events. Here a new argument `--gather` is added for easy filtering of these events. It accepts python expressions like `[1, 3, 5]`, `range(10)`.
