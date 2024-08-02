#!/bin/bash

set -e

dir="legateboost_vs_xgb_8x_v100"
nrows=10000000
ncols=100
repeats=3
mkdir -p ${dir}
for n_gpus in 1 2 4 8; do
    legate --fbmem 29000 --sysmem 30000 --eager-alloc-percentage 20 --gpus ${n_gpus} legateboost_scaling.py --nrows ${nrows} --ncols ${ncols} --output ${dir}/legate_${n_gpus}.csv --repeats ${repeats} --model_types tree
    python dask_xgb_scaling.py --gpus ${n_gpus} --nrows ${nrows} --ncols ${ncols} --output ${dir}/xgb_${n_gpus}.csv --repeats ${repeats} --model_types tree
done

python make_charts.py ${dir} -o ${dir}/legate_vs_xgb_8x_v100.png
