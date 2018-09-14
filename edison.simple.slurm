#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -t 00:02:00
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
export SHMEM_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024))

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
export CORES_PER_SOCKET=12
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))

# export HVR_TRACE_DUMP=1

export MAX_NUM_TIMESTEPS=25

cd $SCRATCH

# echo ""
# echo "RUNNING GRID TEST"
# echo ""
# export HVR_SYMM_POOL_SIZE=40000
# export HVR_VEC_CACHE_PREALLOCS=60000
# export HVR_DISABLE_PROFILING_PRINTS=1
# srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 \
#     $HOME/hoover/bin/init_test 50

# echo ""
# echo "RUNNING ADD VERTICES TEST"
# echo ""
# export HVR_SYMM_POOL_SIZE=40000
# srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 \
#     $HOME/hoover/bin/add_vertices_test 40 $MAX_NUM_TIMESTEPS

# echo ""
# echo "RUNNING MULTI GRAPH TEST"
# echo ""
# export HVR_SYMM_POOL_SIZE=40000
# srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 \
#     $HOME/hoover/bin/multi_graph_test 40 $MAX_NUM_TIMESTEPS

# echo ""
# echo "RUNNING REMOVE VERTICES TEST"
# echo ""
# export HVR_SYMM_POOL_SIZE=40000
# srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 \
#     $HOME/hoover/bin/remove_vertices_test 40 $MAX_NUM_TIMESTEPS

# echo ""
# echo "RUNNING LOCK STRESS TEST"
# echo ""
# srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 \
#     $HOME/hoover/bin/write_lock_stress

# export HVR_SYMM_POOL_SIZE=40000
# export HVR_VEC_CACHE_PREALLOCS=60000
# export HVR_DISABLE_PROFILING_PRINTS=1
# srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
#     --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 \
#     $HOME/hoover/bin/const_attr_test 20

# srun --ntasks=4 $HOME/hoover/bin/neighbor_test_cross_pe

srun --ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) \
    --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1 \
    $HOME/hoover/bin/hvr_dist_bitvec_test

