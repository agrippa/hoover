#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J hoover
#SBATCH --exclusive
#SBATCH --contiguous
#SBATCH --mail-type=ALL

# set -e

ulimit -c unlimited

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/system/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/openshmem/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/sos/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OFI_HOME/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=/opt/intel/compilers_and_libraries_2017.1.132/linux/tbb/lib/intel64/gcc4.7/libtbbmalloc.so.2
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=info

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
export CORES_PER_SOCKET=12
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))

# export HVR_DEFAULT_SPARSE_VEC_VAL=0.0
# srun --ntasks=1 ./bin/sparse_vec_test

# srun --ntasks=1 ./bin/pe_neighbors_set_test

# srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 ./bin/init_test 200

export CELL_DIM=100.0
export N_PORTALS=4
export ACTORS_PER_CELL=10
export PE_ROWS=4
export PE_COLS=6
export N_INFECTED=1
srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
    --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 ./bin/infectious_test \
    $CELL_DIM $N_PORTALS $ACTORS_PER_CELL $PE_ROWS $PE_COLS $N_INFECTED
