#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 64
#SBATCH -C haswell
#SBATCH -t 00:10:00
#SBATCH -J hoover-community
#SBATCH --exclusive
###SBATCH --contiguous
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

export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=info

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
export CORES_PER_SOCKET=16
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))

cd $SCRATCH

# export PE_PER_SOCKET=1
export PE_PER_SOCKET=0

if [[ $PE_PER_SOCKET -eq 1 ]]; then
    export SLURM_ARGS="--ntasks=$(($SLURM_NNODES * $SOCKETS_PER_NODE)) --ntasks-per-socket=1 --cpus-per-task=$CORES_PER_SOCKET"

    if [[ $SLURM_NNODES -eq 1 ]]; then
        export PE_CHUNKS="2 1 1"

    elif [[ $SLURM_NNODES -eq 4 ]]; then
        export PE_CHUNKS="2 2 2"
    elif [[ $SLURM_NNODES -eq 32 ]]; then
        export PE_CHUNKS="4 4 4"
    else
        echo "Unsupported # of nodes $SLURM_NNODES"
        exit 1
    fi

    export OMP_NUM_THREADS=$CORES_PER_SOCKET
    export EXE=community_detection.multi
    export SHMEM_MAX_NUM_THREADS=$OMP_NUM_THREADS
else
    export SLURM_ARGS="--ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1"

    if [[ $SLURM_NNODES -eq 1 ]]; then
        export PE_CHUNKS="4 4 2"
    elif [[ $SLURM_NNODES -eq 8 ]]; then
        export PE_CHUNKS="8 8 4"
    elif [[ $SLURM_NNODES -eq 64 ]]; then
        export PE_CHUNKS="16 16 8"
    elif [[ $SLURM_NNODES -eq 512 ]]; then
        export PE_CHUNKS="32 32 16"

    elif [[ $SLURM_NNODES -eq 2 ]]; then
        export PE_CHUNKS="4 4 4"
    elif [[ $SLURM_NNODES -eq 16 ]]; then
        export PE_CHUNKS="8 8 8"
    elif [[ $SLURM_NNODES -eq 128 ]]; then
        export PE_CHUNKS="16 16 16"
    elif [[ $SLURM_NNODES -eq 1024 ]]; then
        export PE_CHUNKS="32 32 32"

    elif [[ $SLURM_NNODES -eq 4 ]]; then
        export PE_CHUNKS="8 4 4"
    elif [[ $SLURM_NNODES -eq 32 ]]; then
        export PE_CHUNKS="16 8 8"
    elif [[ $SLURM_NNODES -eq 256 ]]; then
        export PE_CHUNKS="32 16 16"
    else
        echo "Unsupported # of nodes $SLURM_NNODES"
        exit 1
    fi

    export OMP_NUM_THREADS=1
    export EXE=community_detection
fi

export PARTITIONS="100 100 100"
export MIN_N_VERTICES_TO_ADD=800
export MAX_N_VERTICES_TO_ADD=1000
if [[ $PE_PER_SOCKET -eq 1 ]]; then
    export MIN_N_VERTICES_TO_ADD=$(($MIN_N_VERTICES_TO_ADD * $CORES_PER_SOCKET))
    export MAX_N_VERTICES_TO_ADD=$(($MAX_N_VERTICES_TO_ADD * $CORES_PER_SOCKET))
fi

export TIME_LIMIT_IN_SECONDS=300
export EDGE_DISTANCE=15
export DOMAIN_SIZE="12000 12000 10000"

# export HVR_DISABLE_PROFILING_PRINTS=1
export HVR_DISABLE_DEAD_PE_PROCESSING=1

export SHMEM_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024))
export HVR_VERT_CACHE_PREALLOCS=$((2 * 1024 * 1024))
export HVR_VERT_CACHE_SEGS=40000
export HVR_SPARSE_ARR_POOL=$((2 * 1024 * 1024))
export HVR_SPARSE_ARR_SEGS=4096
export HVR_DIST_BITVEC_POOL_SIZE=$((64 * 1024 * 1024))
export HVR_EDGES_POOL_SIZE=$((8 * 1024 * 1024))
export HVR_PRODUCER_INFO_SEGS=10240
export HVR_PARTITION_LIST_SEGS=10240

cd $SCRATCH
mkdir -p job.$SLURM_JOB_ID
cd job.$SLURM_JOB_ID

# export TEST_FILE=$HOME/hoover/community_test_cases/4_vert_1_cliq.csv
# export TEST_FILE=$HOME/hoover/community_test_cases/8_vert_2_cliq.csv
export TEST_FILE=$HOME/hoover/community_test_cases/8_vert_2_cliq_overlapping.csv

srun $SLURM_ARGS \
    $HOME/hoover/bin/$EXE \
    $TIME_LIMIT_IN_SECONDS \
    $EDGE_DISTANCE \
    $DOMAIN_SIZE \
    $PE_CHUNKS \
    $PARTITIONS \
    $MIN_N_VERTICES_TO_ADD $MAX_N_VERTICES_TO_ADD
    # $TEST_FILE

