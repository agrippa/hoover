#!/bin/bash -l

#SBATCH -p regular
#SBATCH -N 128
#SBATCH -C haswell
#SBATCH -t 00:20:00
#SBATCH -J hoover-update-rates
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
export LD_LIBRARY_PATH=$HOME/boost_1_58_0-install/lib:$LD_LIBRARY_PATH

export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=info

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
export CORES_PER_SOCKET=16
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))

cd $SCRATCH

# export HVR_HANG_ABORT=360
# export HVR_DISABLE_PROFILING_PRINTS=1
export HVR_DISABLE_DEAD_PE_PROCESSING=1
# export HVR_TRACE_SHMALLOC=1

export SHMEM_SYMMETRIC_SIZE=$((896 * 1024 * 1024))

export HVR_VERT_CACHE_PREALLOCS=$((4 * 1024 * 1024))
export HVR_VERT_CACHE_SEGS=50000
export HVR_SPARSE_ARR_POOL=$((5 * 1024 * 1024))
export HVR_SPARSE_ARR_SEGS=4096
export HVR_DIST_BITVEC_POOL_SIZE=$((64 * 1024 * 1024))
export HVR_EDGES_POOL_SIZE=$((48 * 1024 * 1024))
export HVR_PRODUCER_INFO_SEGS=10240
export HVR_PARTITION_LIST_SEGS=10240
export HVR_BUFFERED_CHANGES_ALLOCATED=200000
export HVR_NEIGHBORS_LIST_POOL_SIZE=$((512 * 1024))
export HVR_MAX_BUFFERED_MSGS=100

cd $SCRATCH
mkdir -p job.$SLURM_JOB_ID
cd job.$SLURM_JOB_ID

export SLURM_ARGS="--ntasks=$(($SLURM_NNODES * $CORES_PER_NODE)) --ntasks-per-socket=$CORES_PER_SOCKET --cpus-per-task=1"

# srun $SLURM_ARGS $HOME/hoover/bin/update_rates $HOME/hoover/in-2004.bin 65536
# srun $SLURM_ARGS $HOME/hoover/bin/update_rates $SCRATCH/soc-LiveJournal1.bin $((65536))
# srun $SLURM_ARGS $HOME/hoover/bin/update_rates "$SCRATCH/kronecker_graphs/scale=21_edgefactor=16_edges=33554432_vertices=2097152.bin" $((65536)) 120
# srun $SLURM_ARGS $HOME/hoover/bin/update_rates "$SCRATCH/kronecker_graphs/scale=22_edgefactor=16_edges=67108864_vertices=4194304.bin" $((65536)) 240
# srun $SLURM_ARGS $HOME/hoover/bin/update_rates "$SCRATCH/kronecker_graphs/scale=23_edgefactor=16_edges=134217728_vertices=8388608.bin" $((65536)) 240
srun $SLURM_ARGS $HOME/hoover/bin/update_rates "$SCRATCH/kronecker_graphs/scale=26_edgefactor=16_edges=1073741824_vertices=67108864.bin" $((65536)) 240
# srun $SLURM_ARGS $HOME/hoover/bin/update_rates "$SCRATCH/kronecker_graphs/scale=29_edgefactor=16_edges=8589934592_vertices=536870912.bin" $((65536)) 240
