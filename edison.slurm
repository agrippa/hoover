#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 16
###SBATCH -C haswell
#SBATCH -t 00:05:00
#SBATCH -J hoover
#SBATCH --exclusive
####SBATCH --contiguous
#SBATCH --mail-type=ALL

# set -e

ulimit -c unlimited

echo "Running on:"
echo $SLURM_NODELIST
echo
echo "Running with OpenSHMEM installation at $OPENSHMEM_INSTALL"

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/system/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/openshmem/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HCLIB_HOME/modules/sos/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OFI_HOME/lib:$LD_LIBRARY_PATH

export LD_PRELOAD=/opt/intel/compilers_and_libraries_2018.2.199/linux/tbb/lib/intel64/gcc4.7/libtbbmalloc.so.2
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=info
export SHMEM_SYMMETRIC_SIZE=$((2 * 1024 * 1024 * 1024))

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
export CORES_PER_SOCKET=12
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))

# export HVR_TRACE_DUMP=1

# export HVR_DEFAULT_SPARSE_VEC_VAL=0.0
# srun --ntasks=1 ./bin/sparse_vec_test

# srun --ntasks=1 ./bin/pe_neighbors_set_test

# srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 ./bin/init_test 120

export BASE_ACTORS_PER_CELL=400 # big
# export BASE_ACTORS_PER_CELL=200
# export BASE_ACTORS_PER_CELL=100

if [[ $SLURM_NNODES -eq 1 ]]; then
    # For 1 node
    export CELL_DIM=4000.0
    export ACTORS_PER_CELL=$(($BASE_ACTORS_PER_CELL * 1024))
    export PE_ROWS=4
    export PE_COLS=6
elif [[ $SLURM_NNODES -eq 4 ]]; then
    # For 4 nodes
    export CELL_DIM=2000.0
    export ACTORS_PER_CELL=$(($BASE_ACTORS_PER_CELL * 256))
    export PE_ROWS=8
    export PE_COLS=12
elif [[ $SLURM_NNODES -eq 16 ]]; then
    # 16 nodes, big run
    export CELL_DIM=1000.0
    export ACTORS_PER_CELL=$(($BASE_ACTORS_PER_CELL * 64))
    export PE_ROWS=16
    export PE_COLS=24
elif [[ $SLURM_NNODES -eq 64 ]]; then
    # For 64 nodes, big run
    export CELL_DIM=500.0
    export ACTORS_PER_CELL=$(($BASE_ACTORS_PER_CELL * 16))
    export PE_ROWS=32
    export PE_COLS=48
elif [[ $SLURM_NNODES -eq 256 ]]; then
    # For 256 nodes, big run
    export CELL_DIM=250.0
    export ACTORS_PER_CELL=$(($BASE_ACTORS_PER_CELL * 4))
    export PE_ROWS=64
    export PE_COLS=96
elif [[ $SLURM_NNODES -eq 1024 ]]; then
    # For 256 nodes, big run
    export CELL_DIM=125.0
    export ACTORS_PER_CELL=$BASE_ACTORS_PER_CELL
    export PE_ROWS=128
    export PE_COLS=192
else
    echo "Unsupported number of nodes! $SLURM_NNODES"
    exit 1
fi

export N_PORTALS=0
export N_INFECTED=1
# export MAX_NUM_TIMESTEPS=30
export MAX_NUM_TIMESTEPS=5
export INFECTION_RADIUS=10.0
export MAX_DELTA_VELOCITY=0.01

export HVR_SYMM_POOL_SIZE=60000

cd $SCRATCH

echo "RUNNING INFECTIOUS MODEL"
echo ""
srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
    --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 \
    $HOME/hoover/bin/infectious_test $CELL_DIM $N_PORTALS $ACTORS_PER_CELL \
    $PE_ROWS $PE_COLS $N_INFECTED $MAX_NUM_TIMESTEPS $INFECTION_RADIUS \
    $MAX_DELTA_VELOCITY

