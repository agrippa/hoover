#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 4
#SBATCH -C haswell
#SBATCH -t 00:10:00
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

export UCX_TLS=ugni_rdma
export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OFI_HOME/lib:$LD_LIBRARY_PATH

export SHMEM_ABORT_ON_ERROR=1
export SHMEM_MEMINFO_DISPLAY=1
# export LD_PRELOAD=/opt/intel/compilers_and_libraries_2018.2.199/linux/tbb/lib/intel64/gcc4.7/libtbbmalloc.so.2
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=info
export SHMEM_SYMMETRIC_SIZE=$((1024 * 1024 * 1024))

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
export TASKS_PER_SOCKET=12
export CORES_PER_SOCKET=12
export SOCKETS_PER_NODE=2
export CORES_PER_TASK=$(( $CORES_PER_SOCKET / $TASKS_PER_SOCKET ))

if [[ $SLURM_NNODES -eq 1 ]]; then
    # For 1 node
    export CELL_DIM=4000.0
    export PE_ROWS=4
    export PE_COLS=6
elif [[ $SLURM_NNODES -eq 4 ]]; then
    # For 4 nodes
    export CELL_DIM=2000.0
    export PE_ROWS=8
    export PE_COLS=12
elif [[ $SLURM_NNODES -eq 16 ]]; then
    # 16 nodes, big run
    export CELL_DIM=1000.0
    export PE_ROWS=16
    export PE_COLS=24
elif [[ $SLURM_NNODES -eq 64 ]]; then
    # For 64 nodes, big run
    export CELL_DIM=500.0
    export PE_ROWS=32
    export PE_COLS=48
elif [[ $SLURM_NNODES -eq 256 ]]; then
    # For 256 nodes, big run
    export CELL_DIM=250.0
    export PE_ROWS=64
    export PE_COLS=96
elif [[ $SLURM_NNODES -eq 1024 ]]; then
    # For 256 nodes, big run
    export CELL_DIM=125.0
    export PE_ROWS=128
    export PE_COLS=192
else
    echo "Unsupported number of nodes! $SLURM_NNODES"
    exit 1
fi

export N_PORTALS=0
export N_INFECTED=1
export MAX_NUM_TIMESTEPS=20
export INFECTION_RADIUS=100

# export HVR_TRACE_SHMALLOC=1
export HVR_SYMM_POOL_SIZE=2084000
export HVR_SYMM_POOL_NNODES=1024
export HVR_VEC_CACHE_PREALLOCS=2000000
export HVR_VERT_CACHE_SEGS=30000
export HVR_DIST_BITVEC_POOL_SIZE=$((16 * 1024 * 1024))
export HVR_SPARSE_ARR_SEGS=12288
export HVR_SPARSE_ARR_POOL=$((32 * 1024 * 1024));
export HVR_EDGES_POOL_SIZE=$((1 * 1024 * 1024 * 1024))

export HVR_TRACE_DUMP=1
export HVR_TRACE_DUMP_ONLY_LAST=1
export HVR_CACHE_TRACE_DUMP=1
# export HVR_DISABLE_PROFILING_PRINTS=1
export HVR_DISABLE_DEAD_PE_PROCESSING=1
# export HVR_HANG_ABORT=400
# export HVR_HANG_ABORT_PE=71

cd $SCRATCH
mkdir -p job.$SLURM_JOB_ID
cd job.$SLURM_JOB_ID

# SHMEM_OPTIMIZED_MEMCPY specifies which version of memcpy to use:
#   0 - system (glibc) version of memcpy.
#   1 - optimized version of memcpy if one is available for the processor being
#       used (there is one available for Intel)
#   2 - highly optimized version of memcpy if one is available for the processor
#       being used. In this release, a highly optimized version of memcpy() is
#       available only for Intel Haswell, Broadwell and KNL processors.
#
# For some reason, SHMEM_OPTIMIZED_MEMCPY=0 seems to trigger bus errors on
# Edison/Cori for some workloads.
export SHMEM_OPTIMIZED_MEMCPY=1

export SLURM_ARGS="--ntasks-per-node=$(( $TASKS_PER_SOCKET * $SOCKETS_PER_NODE )) \
    --cpus-per-task=$CORES_PER_TASK \
    --cpu_bind=verbose,cores \
    --cpu-freq=High \
    --mem-bind=local"
export EXE_ARGS="$CELL_DIM $CELL_DIM $PE_ROWS $PE_COLS $MAX_NUM_TIMESTEPS \
    $INFECTION_RADIUS 180 $SCRATCH/491520-vert.16000-y.32000-x.1-infected.bin"

echo "RUNNING INFECTIOUS MODEL"
echo ""
srun $SLURM_ARGS $HOME/hoover/bin/infectious_test $EXE_ARGS
# srun $SLURM_ARGS $HOME/hoover/bin/coupled_test 240
# srun $SLURM_ARGS $HOME/hoover/bin/init_test 800
# srun $SLURM_ARGS $HOME/hoover/bin/hvr_dist_bitvec_test
# srun $SLURM_ARGS $HOME/hoover/bin/mailbox_test
