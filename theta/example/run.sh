#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV_PREFIX=/path/to/env/location/ssim
DRIVER=driver.py

module swap PrgEnv-intel PrgEnv-gnu
export CRAYPE_LINK_TYPE=dynamic

echo ppn $1
echo nodes $2
echo allprocs $3
echo dbnodes $4
echo simnodes $5
echo mlnodes $6
batch_size=$7
db_tensors_batch_size=$8
echo batch_size $batch_size
echo db_tensors_batch_size $db_tensors_batch_size

module load miniconda-3/2021-07-28
conda activate $CONDA_ENV_PREFIX

python $DRIVER $1 $2 $3 $4 $5 $6 $batch_size $db_tensors_batch_size
