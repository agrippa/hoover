#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 4
#SBATCH -C haswell
#SBATCH -t 00:07:00
#SBATCH -J hoover-infectious
#SBATCH --exclusive
####SBATCH --contiguous
#SBATCH --mail-type=ALL

# set -e

ulimit -c unlimited

echo "Running on:"
echo $SLURM_NODELIST
echo
echo "Running with OpenSHMEM installation at $OPENSHMEM_INSTALL"

export UCX_TLS=ugni_rdma
export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OFI_HOME/lib:$LD_LIBRARY_PATH

export SHMEM_ABORT_ON_ERROR=1
export SMA_OFI_PROVIDER=gni
export SHMEM_OPTIMIZED_MEMCPY=1

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
export CORES_PER_SOCKET=16
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))

# export PE_PER_SOCKET=1
export PE_PER_SOCKET=0

if [[ $PE_PER_SOCKET -eq 0 ]]; then
    if [[ $SLURM_NNODES -eq 1 ]]; then
        # For 1 node
        export CELL_DIM=4000.0
        export PE_ROWS=4
        export PE_COLS=8
    elif [[ $SLURM_NNODES -eq 4 ]]; then
        # For 4 nodes
        export CELL_DIM=2000.0
        export PE_ROWS=8
        export PE_COLS=16
    elif [[ $SLURM_NNODES -eq 16 ]]; then
        # 16 nodes, big run
        export CELL_DIM=1000.0
        export PE_ROWS=16
        export PE_COLS=32
    elif [[ $SLURM_NNODES -eq 64 ]]; then
        # For 64 nodes, big run
        export CELL_DIM=500.0
        export PE_ROWS=32
        export PE_COLS=64
    elif [[ $SLURM_NNODES -eq 256 ]]; then
        # For 256 nodes, big run
        export CELL_DIM=250.0
        export PE_ROWS=64
        export PE_COLS=128
    elif [[ $SLURM_NNODES -eq 1024 ]]; then
        # For 256 nodes, big run
        export CELL_DIM=125.0
        export PE_ROWS=128
        export PE_COLS=256
    else
        echo "Unsupported number of nodes! $SLURM_NNODES"
        exit 1
    fi

    export EXE=infectious_test
    export SLURM_ARGS="--ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1"
    export SHMEM_SYMMETRIC_SIZE=$((768 * 1024 * 1024))
else
    if [[ $SLURM_NNODES -eq 1 ]]; then
        # For 1 node
        export CELL_DIM=4000.0
        export PE_ROWS=1
        export PE_COLS=2
    elif [[ $SLURM_NNODES -eq 4 ]]; then
        # For 4 nodes
        export CELL_DIM=2000.0
        export PE_ROWS=2
        export PE_COLS=4
    else
        echo "Unsupported number of nodes! $SLURM_NNODES"
        exit 1
    fi

    export EXE=infectious_test.multi
    export SLURM_ARGS="--ntasks=$(($SLURM_NNODES * $SOCKETS_PER_NODE)) --ntasks-per-socket=1 --cpus-per-task=$CORES_PER_SOCKET --mem-bind=local"
    export SHMEM_SYMMETRIC_SIZE=$((2 * 1024 * 1024 * 1024))
    export OMP_NUM_THREADS=$CORES_PER_SOCKET
    export SHMEM_MAX_NUM_THREADS=$OMP_NUM_THREADS
fi

# Small problem setup
# export MAX_NUM_TIMESTEPS=270
export MAX_NUM_TIMESTEPS=180
export INPUT_FILE=$SCRATCH/491520-vert.16000-y.32000-x.1-infected.bin
export INFECTION_RADIUS=100
export HVR_SYMM_POOL_SIZE=3000000
export HVR_VERT_CACHE_PREALLOCS=3000000
export HVR_EDGES_POOL_SIZE=$((768 * 1024 * 1024))
export HVR_MAX_ACTIVE_PARTITIONS=500000

# Larger problem setup
# export MAX_NUM_TIMESTEPS=140
# export INPUT_FILE=$SCRATCH/3000000-vert.16000-y.32000-x.1-infected.bin
# export INFECTION_RADIUS=8
# export HVR_SYMM_POOL_SIZE=5000000
# export HVR_VERT_CACHE_PREALLOCS=6000000
# export HVR_EDGES_POOL_SIZE=$((512 * 1024 * 1024))
# export HVR_MAX_ACTIVE_PARTITIONS=400000

export MAX_WALLTIME_SECONDS=300

export HVR_SYMM_POOL_NNODES=1024
export HVR_VERT_CACHE_SEGS=50000
export HVR_PRODUCER_INFO_SEGS=8192
export HVR_DIST_BITVEC_POOL_SIZE=$((128 * 1024 * 1024))
export HVR_SPARSE_ARR_SEGS=16384
export HVR_SPARSE_ARR_POOL=$((192 * 1024 * 1024));
export HVR_CACHED_INTERACTIONS_POOL_SIZE=$((128 * 1024 * 1024))
export HVR_BUFFERED_MSGS_POOL_SIZE=$((64 * 1024 * 1024))
export HVR_PARTITION_LIST_SEGS=16384

# export HVR_TRACE_DUMP=1
# export HVR_TRACE_DUMP_ONLY_LAST=1
# export HVR_CACHE_TRACE_DUMP=1
# export HVR_DISABLE_PROFILING_PRINTS=1
export HVR_DISABLE_DEAD_PE_PROCESSING=1
# export HVR_HANG_ABORT=360
# export HVR_HANG_ABORT_PE=17

cd $SCRATCH
mkdir -p job.$SLURM_JOB_ID
cd job.$SLURM_JOB_ID

export EXE_ARGS="$CELL_DIM $CELL_DIM $PE_ROWS $PE_COLS $MAX_NUM_TIMESTEPS \
    $INFECTION_RADIUS $MAX_WALLTIME_SECONDS $INPUT_FILE"

srun $SLURM_ARGS $HOME/hoover/bin/$EXE $EXE_ARGS
