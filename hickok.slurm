#!/bin/bash -l

#SBATCH -p connectx5
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J hoover
#SBATCH --exclusive
#SBATCH --contiguous
#SBATCH --mail-type=ALL
#SBATCH --exclude=r01n07

# set -e

ulimit -c unlimited

echo "Running on:"
echo $SLURM_NODELIST
echo
echo "Using OpenSHMEM @ $OPENSHMEM_INSTALL"
echo

export SMA_SYMMETRIC_SIZE=$((1 * 1024 * 1024 * 1024))

export PSM2_SHAREDCONTEXTS=0
export PMI_MAX_KVS_ENTRIES=$((1000 * $SLURM_NNODES))
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OPENSHMEM_INSTALL/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$OFI_HOME/lib:$LD_LIBRARY_PATH

# export LD_PRELOAD=/opt/intel/compilers_and_libraries_2017.1.132/linux/tbb/lib/intel64/gcc4.7/libtbbmalloc.so.2
export SMA_OFI_PROVIDER=psm2
# export FI_LOG_LEVEL=info

# 2 sockets x 16 cores per socket for Cori Haswell
# 2 sockets x 12 cores per socket for Edison
# 2 sockets x 8 cores per socket for Hickok (2 hardware threads per core)
export CORES_PER_SOCKET=8
export SOCKETS_PER_NODE=2
export CORES_PER_NODE=$(($SOCKETS_PER_NODE * $CORES_PER_SOCKET))

# export HVR_TRACE_DUMP=1

# export HVR_DEFAULT_SPARSE_VEC_VAL=0.0
# srun --ntasks=1 ./bin/sparse_vec_test

# srun --ntasks=1 ./bin/pe_neighbors_set_test

COPY_TO=$(pwd)
echo "Clearing out old core files"
srun --ntasks-per-node=1 bash -c "rm -f /var/tmp/core.*"
echo
echo
echo

export SCALING=strong
if [[ $SLURM_NNODES -eq 1 ]]; then
    # For 2 nodes
    if [[ $SCALING == strong ]]; then
        export CELL_DIM=400.0
        export ACTORS_PER_CELL=3200
        export PE_ROWS=4
        export PE_COLS=4
    else
        export CELL_DIM=400.0
        export ACTORS_PER_CELL=2800
        export PE_ROWS=4
        export PE_COLS=4
    fi
elif [[ $SLURM_NNODES -eq 4 ]]; then
    # For 4 nodes
    if [[ $SCALING == strong ]]; then
        export CELL_DIM=200.0
        export ACTORS_PER_CELL=800
        export PE_ROWS=8
        export PE_COLS=8
    else
        export CELL_DIM=400.0
        export ACTORS_PER_CELL=2800
        export PE_ROWS=8
        export PE_COLS=8
    fi
elif [[ $SLURM_NNODES -eq 16 ]]; then
    # For 16 nodes
    if [[ $SCALING == strong ]]; then
        # export CELL_DIM=100.0 # for illustrating coupled execution
        export CELL_DIM=1000.0 # for illustrating uncoupled execution
        export ACTORS_PER_CELL=200
        export PE_ROWS=16
        export PE_COLS=16
    else
        export CELL_DIM=400.0
        export ACTORS_PER_CELL=2800
        export PE_ROWS=16
        export PE_COLS=16
    fi
else
    echo "Unsupported number of nodes! $SLURM_NNODES"
    exit 1
fi

export N_PORTALS=0
export N_INFECTED=1
export MAX_DELTA_VELOCITY=0.01

if [[ "$(basename $OPENSHMEM_INSTALL)" == "SOS-install" ]]; then
    # For SoS
    export PROCESS_CONFIG="-n $(($CORES_PER_NODE * $SLURM_NNODES))"
elif [[ "$(basename $OPENSHMEM_INSTALL)" == "osss-ucx" ]]; then
    # For OSSS
    export PROCESS_CONFIG="-npernode $CORES_PER_NODE"
else 
    echo "Unexpected OPENSHMEM_INSTALL when auto-configuring PROCESS_CONFIG? $OPENSHMEM_INSTALL"
    exit 1
fi

# for MAX_NUM_TIMESTEPS in 10 100 200; do
for MAX_NUM_TIMESTEPS in 100; do
    # for INFECTION_RADIUS in 4.0 10.0; do
    for INFECTION_RADIUS in 10.0; do

        $OPENSHMEM_INSTALL/bin/oshrun $PROCESS_CONFIG \
                $HOME/hoover/bin/infectious_test \
                $CELL_DIM $N_PORTALS $ACTORS_PER_CELL $PE_ROWS $PE_COLS $N_INFECTED \
                $MAX_NUM_TIMESTEPS $INFECTION_RADIUS $MAX_DELTA_VELOCITY
    done
done

echo
echo
echo
echo "Listing core files"
srun --ntasks-per-node=1 bash -c "ls -l /var/tmp/"
echo "Copying out old core files"
srun --ntasks-per-node=1 bash -c "cp /var/tmp/core.* $COPY_TO/"
