#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -C haswell
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
export TASKS_PER_SOCKET=16
export CORES_PER_SOCKET=16
export SOCKETS_PER_NODE=2
export CORES_PER_TASK=$(( $CORES_PER_SOCKET / $TASKS_PER_SOCKET ))

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

export N_PORTALS=0
export N_INFECTED=1
export MAX_NUM_TIMESTEPS=240
export INFECTION_RADIUS=100

export HVR_SYMM_POOL_SIZE=2084000
export HVR_SYMM_POOL_NNODES=1024
export HVR_VEC_CACHE_PREALLOCS=3000000
export HVR_VERT_CACHE_SEGS=40000
export HVR_DIST_BITVEC_POOL_SIZE=$((64 * 1024 * 1024))
export HVR_SPARSE_ARR_SEGS=8192
export HVR_SPARSE_ARR_POOL=$((192 * 1024 * 1024));
export HVR_EDGES_POOL_SIZE=$((1 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))

export HVR_TRACE_DUMP=1
export HVR_TRACE_DUMP_ONLY_LAST=1
# export HVR_CACHE_TRACE_DUMP=1
# export HVR_DISABLE_PROFILING_PRINTS=1
export HVR_DISABLE_DEAD_PE_PROCESSING=1
# export HVR_HANG_ABORT=100
# export HVR_HANG_ABORT_PE=17

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
export SHMEM_OPTIMIZED_MEMCPY=1

export SLURM_ARGS="--ntasks-per-node=$(( $TASKS_PER_SOCKET * $SOCKETS_PER_NODE )) \
    --cpus-per-task=$CORES_PER_TASK \
    --cpu_bind=verbose,cores \
    --cpu-freq=High \
    --mem-bind=local"
export EXE_ARGS="$CELL_DIM $CELL_DIM $PE_ROWS $PE_COLS $MAX_NUM_TIMESTEPS \
    $INFECTION_RADIUS 600 $SCRATCH/491520-vert.16000-y.32000-x.1-infected.bin"

echo "RUNNING INFECTIOUS MODEL"
echo ""
# oshrun --npersocket $TASKS_PER_SOCKET $HOME/hoover/bin/init_test 30
# oshrun --npersocket $TASKS_PER_SOCKET $HOME/hoover/bin/hvr_set_msg_test
# oshrun --npersocket $TASKS_PER_SOCKET $HOME/hoover/bin/mailbox_test
# oshrun --npersocket $TASKS_PER_SOCKET $HOME/hoover/bin/infectious_test $EXE_ARGS

# srun $SLURM_ARGS $HOME/hoover/bin/init_test 500

export SIMULATION_TIMESTEPS=20
export MAX_SECONDS=120
srun $SLURM_ARGS $HOME/hoover/bin/simple_nbody_test $SIMULATION_TIMESTEPS $MAX_SECONDS

# srun $SLURM_ARGS $HOME/hoover/bin/pas $EXE_ARGS
# srun $SLURM_ARGS $HOME/hoover/bin/infectious_test $EXE_ARGS
# srun -n 2 $HOME/hoover/bin/complex_interact2
# srun -n 2 $HOME/hoover/bin/stale_state
# srun -n 2 $HOME/hoover/bin/complex_interact
# srun $SLURM_ARGS $HOME/hoover/bin/coupled_test 480

# Edge microbenchmarks
# srun -n 1 $HOME/hoover/bin/hvr_map_microbenchmark
# srun -n 1 $HOME/hoover/bin/hvr_2d_set_microbenchmark
# srun -n 1 $HOME/hoover/bin/hvr_irregular_matrix_microbenchmark
