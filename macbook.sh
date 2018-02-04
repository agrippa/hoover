#!/bin/bash -l

# set -e

ulimit -c unlimited

echo "Running with OpenSHMEM installation at $OPENSHMEM_INSTALL"

export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/system/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/openshmem/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/sos/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OFI_HOME/lib:$LD_LIBRARY_PATH

export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=info
export SHMEM_SYMMETRIC_SIZE=$((1024 * 1024 * 1024))

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
export CORES_PER_SOCKET=8
export SOCKETS_PER_NODE=1
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))

# export HVR_TRACE_DUMP=1

# export HVR_DEFAULT_SPARSE_VEC_VAL=0.0
# srun --ntasks=1 ./bin/sparse_vec_test

# srun --ntasks=1 ./bin/pe_neighbors_set_test

# srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 ./bin/init_test 120

export CELL_DIM=100.0
export ACTORS_PER_CELL=1000
export PE_ROWS=4
export PE_COLS=2

export N_PORTALS=0
export N_INFECTED=1
# export MAX_NUM_TIMESTEPS=1000
# export MAX_NUM_TIMESTEPS=300
export MAX_NUM_TIMESTEPS=3
export INFECTION_RADIUS=4.0
export MAX_DELTA_VELOCITY=0.01

oshrun -n $CORES_PER_NODE $HOME/hoover/bin/infectious_test \
    $CELL_DIM $N_PORTALS $ACTORS_PER_CELL $PE_ROWS $PE_COLS $N_INFECTED \
    $MAX_NUM_TIMESTEPS $INFECTION_RADIUS $MAX_DELTA_VELOCITY
