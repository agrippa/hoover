#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:02:00
#SBATCH -J hoover-count-triangles
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
export LD_LIBRARY_PATH=$HOME/boost_1_58_0-install/lib:$LD_LIBRARY_PATH

export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=info

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
export CORES_PER_SOCKET=16
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))

cd $SCRATCH

export TIME_LIMIT_IN_SECONDS=30
export NVERTICES=10000000
export NEDGES_PER_STEP=1000

# export HVR_DISABLE_PROFILING_PRINTS=1
export HVR_DISABLE_DEAD_PE_PROCESSING=1
# export HVR_TRACE_SHMALLOC=1

export SHMEM_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
export HVR_VERT_CACHE_PREALLOCS=$((6 * 1024 * 1024))
export HVR_VERT_CACHE_SEGS=50000
export HVR_SPARSE_ARR_POOL=$((160 * 1024 * 1024))
export HVR_SPARSE_ARR_SEGS=4096
export HVR_DIST_BITVEC_POOL_SIZE=$((64 * 1024 * 1024))
export HVR_EDGES_POOL_SIZE=$((256 * 1024 * 1024))
export HVR_PRODUCER_INFO_SEGS=10240
export HVR_PARTITION_LIST_SEGS=10240
export HVR_BUFFERED_CHANGES_ALLOCATED=20000
export HVR_NEIGHBORS_LIST_POOL_SIZE=131072

cd $SCRATCH
mkdir -p job.$SLURM_JOB_ID
cd job.$SLURM_JOB_ID

export SLURM_ARGS="--ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1"
srun $SLURM_ARGS \
    $HOME/hoover/bin/update_rates $HOME/hoover/in-2004.graph

