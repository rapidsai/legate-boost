# Memory usage in Legateboost

Here we describe some basic guidelines for memory usage in Legateboost.

## Configuring legate

Programs launched with legate often need to have the default memory limit increased to support larger workloads and get the most efficient use of the available hardware. When more data can be packed onto a smaller number of processors, as opposed to a higher number of processors with less data per processor, communication overheads are reduced and the utliization of individual processors is higher.

For example when running on CPUs, the below command increases the host memory from 4GB to 16GB and restricts the eager allocation to 10% of the total memory (this is expected to be more than sufficient).

```bash
legate --cpus 20 --sysmem 16000 --eager-alloc-percentage 10 example.py
```

When running on GPUs, use the following to also increase the GPU memory.

```bash
legate --gpus 1 --sysmem 16000 --fbmem 16000 --eager-alloc-percentage 10 example.py
```

To track memory, use the `--mem-usage` flag.

```bash
legate --mem-usage example.py
```

## Memory usage for different model types
In general the intermediate boosting stages of running legateboost (gradient calculation, prediction etc.) are expected to use an amount of memory linear in the number of dataset rows.

### Tree models
Tree models in legateboost have a memory efficient C++ implementation and are not expected to use substantially more memory than the dataset itself.

### Linear models and Kernel Ridge Regression
Linear models and kernel ridge regression are implemented using cunumeric. Due to intermediate results (e.g. from series of matrix operations) or data type conversions these algorithms can use several times more memory than the input dataset.
