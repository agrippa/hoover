#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 16
#SBATCH -C haswell
#SBATCH -t 00:02:00
#SBATCH -J hoover
#SBATCH --exclusive
#SBATCH --contiguous
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

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
export CORES_PER_SOCKET=12
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))

# export HVR_TRACE_DUMP=1

cd $SCRATCH

export SHMEM_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
export TIME_LIMIT_IN_SECONDS=50
export EDGE_DISTANCE=10
export DOMAIN_SIZE="12000 10000 10000"

# if [[ $SLURM_NNODES -eq 4 ]]; then
#     export PE_CHUNKS="6 4 4"
# elif [[ $SLURM_NNODES -eq 32 ]]; then
#     export PE_CHUNKS="12 8 8"
# elif [[ $SLURM_NNODES -eq 256 ]]; then
#     export PE_CHUNKS="24 16 16"
# else
#     echo "Unsupported # of nodes $SLURM_NNODES"
#     exit 1
# fi

if [[ $SLURM_NNODES -eq 2 ]]; then
    export PE_CHUNKS="3 4 4"
elif [[ $SLURM_NNODES -eq 16 ]]; then
    export PE_CHUNKS="6 8 8"
elif [[ $SLURM_NNODES -eq 128 ]]; then
    export PE_CHUNKS="12 16 16"
elif [[ $SLURM_NNODES -eq 1024 ]]; then
    export PE_CHUNKS="24 32 32"
fi

export PARTITIONS="40 40 40"
export MIN_N_VERTICES_TO_ADD=50
export MAX_N_VERTICES_TO_ADD=100
# export HVR_HANG_ABORT=300
export HVR_SYMM_POOL_SIZE=45000
export HVR_VEC_CACHE_PREALLOCS=60000
# export HVR_DISABLE_PROFILING_PRINTS=1
srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
    --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 \
    $HOME/hoover/bin/intrusion_detection \
    $TIME_LIMIT_IN_SECONDS \
    $EDGE_DISTANCE \
    $DOMAIN_SIZE \
    $PE_CHUNKS \
    $PARTITIONS \
    $MIN_N_VERTICES_TO_ADD $MAX_N_VERTICES_TO_ADD

# export SHMEM_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024 + 512 * 1024 * 1024))
# export TIME_LIMIT_IN_SECONDS=50
# export EDGE_DISTANCE=10
# export DOMAIN_SIZE="4 3 2"
# export PE_CHUNKS="4 3 2"
# export PARTITIONS="4 3 2"
# export MIN_N_VERTICES_TO_ADD=1
# export MAX_N_VERTICES_TO_ADD=2
# # export HVR_HANG_ABORT=300
# export HVR_SYMM_POOL_SIZE=45000
# export HVR_VEC_CACHE_PREALLOCS=60000
# # export HVR_DISABLE_PROFILING_PRINTS=1
# srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 \
#     $HOME/hoover/bin/intrusion_detection \
#     $TIME_LIMIT_IN_SECONDS \
#     $EDGE_DISTANCE \
#     $DOMAIN_SIZE \
#     $PE_CHUNKS \
#     $PARTITIONS \
#     $MIN_N_VERTICES_TO_ADD $MAX_N_VERTICES_TO_ADD
