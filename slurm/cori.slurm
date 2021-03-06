#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 00:20:00
#SBATCH -J hoover
#SBATCH --exclusive
####SBATCH --contiguous
#SBATCH --mail-type=ALL
#SBATCH -A m3225

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
# export SHMEM_MEMINFO_DISPLAY=1
# export LD_PRELOAD=/opt/intel/compilers_and_libraries_2018.2.199/linux/tbb/lib/intel64/gcc4.7/libtbbmalloc.so.2
export SMA_OFI_PROVIDER=gni
# export FI_LOG_LEVEL=info

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
export TASKS_PER_SOCKET=16
export CORES_PER_SOCKET=16
export SOCKETS_PER_NODE=2
export CORES_PER_TASK=$(( $CORES_PER_SOCKET / $TASKS_PER_SOCKET ))

export SHMEM_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024))
export HVR_VERT_CACHE_PREALLOCS=30000
export HVR_VERT_CACHE_SEGS=40000
export HVR_DIST_BITVEC_POOL_SIZE=$((64 * 1024 * 1024))
export HVR_SPARSE_ARR_SEGS=4096
export HVR_SPARSE_ARR_POOL=$((4 * 1024 * 1024))
export HVR_EDGES_POOL_SIZE=$((50 * 1024 * 1024))
export HVR_PRODUCER_INFO_SEGS=20480
export HVR_PARTITION_LIST_SEGS=16384
export HVR_NEIGHBORS_LIST_POOL_SIZE=16384

# export HVR_TRACE_DUMP=1
# export HVR_TRACE_DUMP_ONLY_LAST=1
# export HVR_CACHE_TRACE_DUMP=1
# export HVR_DISABLE_PROFILING_PRINTS=1
export HVR_DISABLE_DEAD_PE_PROCESSING=1
# export HVR_HANG_ABORT=100
# export HVR_HANG_ABORT_PE=17

cd $SCRATCH
mkdir -p job.$SLURM_JOB_ID
cd job.$SLURM_JOB_ID

export SLURM_ARGS="--ntasks-per-node=$(( $TASKS_PER_SOCKET * $SOCKETS_PER_NODE )) \
    --cpus-per-task=$CORES_PER_TASK \
    --cpu_bind=cores \
    --cpu-freq=High \
    --mem-bind=local"

##### Applications #####
# export SIMULATION_TIMESTEPS=2000
# export MAX_SECONDS=120
# srun $SLURM_ARGS $HOME/hoover/bin/simple_nbody_test $SIMULATION_TIMESTEPS $MAX_SECONDS

# make bin/explicit_edge_test bin/init_test bin/complex_interact bin/complex_interact2 bin/stale_state bin/coupled_test bin/connected_components

##### Functional tests #####
# echo 'mailbox_test'
# srun -n 2 $HOME/hoover/bin/mailbox_test
# echo
# echo 'mailbox_buffer_test'
# srun $SLURM_ARGS $HOME/hoover/bin/hvr_mailbox_buffer_test
# echo

echo 'explicit_edge_test'
srun -n 2 $HOME/hoover/bin/explicit_edge_test
echo
echo 'init_test'
srun $SLURM_ARGS $HOME/hoover/bin/init_test 100
echo
echo 'complex_interact'
srun -n 2 $HOME/hoover/bin/complex_interact
echo
echo 'complex_interact2'
srun -n 2 $HOME/hoover/bin/complex_interact2 300
echo
echo 'stale_state'
srun -n 2 $HOME/hoover/bin/stale_state
echo
echo 'coupled_test'
srun $SLURM_ARGS $HOME/hoover/bin/coupled_test 240
echo
echo 'connected_components'
srun $SLURM_ARGS $HOME/hoover/bin/connected_components 10 100000 100

