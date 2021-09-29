# Smartsim on Theta at ALCF


## Installation


### Install Smartsim

Login into one of Theta's login nodes with `ssh theta.alcf.anl.gov` and run the script [`install_ssim_theta.sh`](install_ssim_theta.sh)

```bash
./install_ssim_theta.sh /path/to/env/location
```

This will create a conda environment called `ssim` in `/path/to/env/location/ssim` and install the following software versions from the channel `conda-forge`:

- python 3.8
- pytorch 1.7.1
- tensorflow 2.4.2
- smartsim 0.3.2
- smartredis 0.2.0

To activate the environment type 
`conda activate /path/to/env/location/ssim`. 

To test the Smartsim installation type 
`python -c 'import smartsim' and python -c 'from smartsim import Experiment'`

To test the Smartredis installation type 
`python -c 'from smartredis import Client'`



```{.bash caption="Script to install Smartsim on Theta."}
#!/bin/bash

PREFIX="$1"
ENVNAME=ssim

# set the environment
module swap PrgEnv-intel PrgEnv-gnu
export CRAYPE_LINK_TYPE=dynamic
module unload craype-mic-knl
module load miniconda-3/2021-07-28

# create the conda environment
conda create -c conda-forge --prefix $PREFIX/$ENVNAME python=3.8 pytorch=1.7.1 pip
conda activate $PREFIX/$ENVNAME
conda install -c conda-forge git-lfs
git lfs install

# install smartsim
git clone https://github.com/CrayLabs/SmartSim.git --depth=1 --branch v0.3.2 smartsim-0.3.2
cd smartsim-0.3.2
pip install -e .[dev,ml]
smart -v --device cpu
#TEST: python -c 'import smartsim' and python -c 'from smartsim import Experiment'

# install smartredis
git clone https://github.com/CrayLabs/SmartRedis.git --depth=1 --branch v0.2.0 smartredis-0.2.0
cd smartredis-0.2.0
pip install -e .[dev]
#TEST: python -c 'from smartredis import Client'

export CC=/opt/gcc/9.3.0/bin/gcc
export CXX=/opt/gcc/9.3.0/bin/g++
make deps
make test-deps
make lib
```


### Install Horovod

[Horovod](https://github.com/horovod/horovod) is commonly used for data parallel training of deep learning models. 
To install Horovod on Theta, first activate the conda environment just created with `conda activate /path/to/env/location/ssim`, then run the script [`install_horovod_theta.sh`](install_horovod_theta.sh), which is based on [this guideline](https://github.com/jtchilders/conda_install_scripts/blob/master/alcf_theta/install_miniconda.sh#L169). 


```{.bash caption="Script install_horovod_theta.sh to install Horovod on Theta."}
#!/bin/bash

### Install Horovod
# NOTE: - need to source the SSIM conda environment first

# set the environment
module swap PrgEnv-intel PrgEnv-gnu
module swap gcc gcc/8.3.0
export CRAY_CPU_TARGET=mic-knl

# Horovod source and version
HOROVOD_REPO_URL=https://github.com/uber/horovod.git
HOROVOD_REPO_TAG=v0.21.3

echo Clone Horovod $HOROVOD_REPO_TAG git repo
DIR_NAME=horovod-$HOROVOD_REPO_TAG
git clone --recursive $HOROVOD_REPO_URL $DIR_NAME
cd $DIR_NAME
git checkout $HOROVOD_REPO_TAG

HOROVOD_CMAKE=$(which cmake) HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 CC=$(which cc) CXX=$(which CC) python setup.py bdist_wheel
HVD_WHL=$(find dist/ -name "horovod*.whl" -type f)
cp $HVD_WHL .
HVD_WHEEL=$(find . -maxdepth 1 -name "horovod*.whl" -type f)
echo Install Horovod $HVD_WHEEL
pip install --force-reinstall $HVD_WHEEL
```


## Smartsim configuration


### `maxclients`

In Redis' configuration file `../smartsim-0.3.2/smartsim/database/redis6.conf`, increase 
`maxclients` to 25000.


### Use KeyDB instead of Redis

Redis is Smartsim's default in-memory databsase and the only one officially supported. 
However Smartsim can also work with other types of databases, such as KeyDB, which may be preferred to Redis in some cases (see for example the [scaling tests](https://github.com/CrayLabs/SmartSim-Scaling)).
Here below are the instructions on how to install KeyDB and use it with Smartsim. 

Set the environment

```bash
module swap PrgEnv-intel PrgEnv-gnu
export CRAYPE_LINK_TYPE=dynamic
module unload craype-mic-knl
```

Clone and build KeyDB 

```bash
git clone https://www.github.com/eq-alpha/keydb.git --branch v6.0.16
cd keydb/
CC=gcc CXX=g++ make -j 8
```

Copy the server and configuration files over to smartsim and replace them with the
Redis files. SmartSim will then be tricked into running KeyDB instead of Redis

```bash
# create backup files for the default Redis server and configuration files
mv ../smartsim-0.3.2/smartsim/bin/redis-server smartsim-0.3.2/smartsim/bin/redis-server.bkp
mv ../smartsim-0.3.2/smartsim/database/redis6.conf smartsim-0.3.2/smartsim/database/redis6.conf.bkp

# replace with KeyDB's server and configuration files
cp src/keydb-server ../smartsim-0.3.2/smartsim/bin/redis-server
cp src/keydb.conf ../smartsim-0.3.2/smartsim/database/redis6.conf
```

Change the configuration file 
`../smartsim-0.3.2/smartsim/database/redis6.conf` 
as follows

- `loglevel`: change to verbose if debugging, otherwise notice or warning as verbose writing slows performance
- `save`: comment out all save options and replace with empty string ""
- `maxclients`: 25000
- `appendfsync`: comment all options out
- `server-threads`: 8
- `bind`: comment out
- `protected-mode`: no


## Submit jobs


[`submit.sh`](example/submit.sh) and [`run.sh`](example/run.sh) are scripts to submit a job that launches an all-python [example SmartSim application](#example) ([`driver.py`](example/driver.py)).


### `submit.sh`

```bash
#!/bin/bash

# args:
dbnodes=4
batch_size=4
db_tensors_per_ml_rank=2

ppn=64
nodes=8
allprocs=$(($nodes * $ppn))
simnodes=2
mlnodes=$(($nodes - $dbnodes - $simnodes))

ChargeAccount=youraccount
runtime=00:30:00
queue=debug-cache-quad 

echo number of nodes $nodes 
echo time in minutes $runtime
echo number of processes $allprocs
echo ppn  N $ppn
echo queue $queue
 
qsub -q $queue -n $nodes -t $runtime -A $ChargeAccount run.sh $ppn $nodes $allprocs $dbnodes $simnodes $mlnodes $batch_size $db_tensors_per_ml_rank
```


### `run.sh`

```bash
#!/bin/bash

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
db_tensors_per_ml_rank=$8
echo batch_size $batch_size
echo db_tensors_per_ml_rank $db_tensors_per_ml_rank

module load miniconda-3/2021-07-28
conda activate $CONDA_ENV_PREFIX

python $DRIVER $1 $2 $3 $4 $5 $6 $batch_size $db_tensors_per_ml_rank
```


## Example



The code in [`example`](example) is a simple all-python SmartSim application of online training. 
The [`submit.sh`](example/submit.sh) script submits to Theta's `debug-cache-quad` queue a job with 8 nodes with 64 processes each: 4 nodes are used by SmartSim's database, 2 nodes are used by the data producer and 2 nodes are used by the data consumer.
[`driver.py`](example/driver.py) starts the database and launches the data producer and the data consumer jobs through SmartSim. 
The data producer ([`load_data.py`](example/load_data.py)) creates two random vectors per rank and uploads them to the database. 
The data consumer ([`train_model.py`](example/train_model.py)) retrieves the latest avaialble data from the database and trains a model over a fixed number of epochs. 



## Documentation


Smartsim's official repository is <https://github.com/CrayLabs/SmartSim> and Smartredis' official repository is <https://github.com/CrayLabs/SmartRedis>. 

The Smartsim repository contains examples and the link to the official [documentation](https://www.craylabs.org/docs/overview.html). 

Additional information about scaling of Smartsim on HPC is at <https://github.com/CrayLabs/SmartSim-Scaling>.

For a list of publications using Smartsim, including the [paper to cite](https://arxiv.org/abs/2104.09355), see <https://github.com/CrayLabs/SmartSim#publications>.



