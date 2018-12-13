#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:02:00
#SBATCH -J hoover
#SBATCH --exclusive
#SBATCH --mail-type=ALL

ulimit -c unlimited

echo "Running on:"
echo $SLURM_NODELIST
echo
echo "Running with OpenSHMEM installation at $OPENSHMEM_INSTALL"

export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib64:$LD_LIBRARY_PATH
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

export HVR_SYMM_POOL_SIZE=2084000
export HVR_SYMM_POOL_NNODES=1024
export HVR_VEC_CACHE_PREALLOCS=2800000
export HVR_VERT_CACHE_SEGS=40000
export HVR_EDGE_SET_SEGS=40000
export HVR_EDGE_SET_VALS=$((1024 * 1024 * 1024))
export HVR_EDGE_SET_NODES=65536
export HVR_DIST_BITVEC_POOL_SIZE=$((16 * 1024 * 1024))
export HVR_SPARSE_ARR_SEGS=2048
export HVR_SPARSE_ARR_POOL=$((32 * 1024 * 1024));

# export HVR_TRACE_DUMP=1
# export HVR_TRACE_DUMP_ONLY_LAST=1
# export HVR_DISABLE_PROFILING_PRINTS=1
export HVR_DISABLE_DEAD_PE_PROCESSING=1
# export HVR_HANG_ABORT=240
# export HVR_HANG_ABORT_PE=17

cd $SCRATCH

srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
    --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 \
    $HOME/hoover/bin/coupled_test 60
