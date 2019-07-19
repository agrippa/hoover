#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 00:20:00
#SBATCH -J flink
#SBATCH -o flink.out
#SBATCH -e flink.err
#SBATCH --exclusive
####SBATCH --contiguous
#SBATCH --mail-type=ALL

# set -e

ulimit -c unlimited

echo "Running on:"
echo $SLURM_NODELIST

export FLINK_HOME=$HOME/hoover/bench/flink-gelly/flink-1.8.1

# module load spark/2.3.0
# module load scala/2.11.8
module load java/jdk1.8.0_51

echo
echo

mkdir -p $SCRATCH/job.$SLURM_JOB_ID
pushd $SCRATCH/job.$SLURM_JOB_ID

pushd /global/homes/j/jmg3/hoover/bench/flink-gelly/community-detection

NHOSTS=$(scontrol show hostname | wc -w)
if [[ $NHOSTS -lt 2 ]]; then
    echo Require at least 2 hosts
    exit 1
fi

MASTER_NODE=$(scontrol show hostname | head -n 1)

# Extracted from flink-1.8.1-bin-scala_2.11.tgz
rm $FLINK_HOME/log/*

rm -f *.class *.jar
javac -cp $FLINK_HOME/lib/flink-dist_2.11-1.8.1.jar:$FLINK_HOME/lib/flink-gelly_2.11-1.8.1.jar:$FLINK_HOME/lib/flink-gelly-scala_2.11-1.8.1.jar CommunityDetection.java
jar cvfm CommunityDetection.jar Manifest.txt *.class

FLINK_CONF=$FLINK_HOME/conf/flink-conf.yaml

cat $FLINK_CONF | grep -v "^jobmanager.rpc.address" > ${FLINK_CONF}.tmp
mv ${FLINK_CONF}.tmp $FLINK_CONF
cat $FLINK_CONF | grep -v "^jobmanager.heap.size" > ${FLINK_CONF}.tmp
mv ${FLINK_CONF}.tmp $FLINK_CONF
cat $FLINK_CONF | grep -v "^taskmanager.heap.size" > ${FLINK_CONF}.tmp
mv ${FLINK_CONF}.tmp $FLINK_CONF
cat $FLINK_CONF | grep -v "^env.java.home" > ${FLINK_CONF}.tmp
mv ${FLINK_CONF}.tmp $FLINK_CONF
cat $FLINK_CONF | grep -v "^taskmanager.numberOfTaskSlots" > ${FLINK_CONF}.tmp
mv ${FLINK_CONF}.tmp $FLINK_CONF
cat $FLINK_CONF | grep -v "^io.tmp.dirs" > ${FLINK_CONF}.tmp
mv ${FLINK_CONF}.tmp $FLINK_CONF
cat $FLINK_CONF | grep -v "^taskmanager.debug.memory.log:" > ${FLINK_CONF}.tmp
mv ${FLINK_CONF}.tmp $FLINK_CONF
cat $FLINK_CONF | grep -v "^taskmanager.debug.memory.log-interval:" > ${FLINK_CONF}.tmp
mv ${FLINK_CONF}.tmp $FLINK_CONF
cat $FLINK_CONF | grep -v "^resourcemanager.taskmanager-timeout:" > ${FLINK_CONF}.tmp
mv ${FLINK_CONF}.tmp $FLINK_CONF

echo "jobmanager.rpc.address: $MASTER_NODE" >> $FLINK_CONF
echo "jobmanager.heap.size: 4096m" >> $FLINK_CONF
echo "taskmanager.heap.size: 16g" >> $FLINK_CONF
echo "env.java.home: $JAVA_HOME" >> $FLINK_CONF
echo "taskmanager.numberOfTaskSlots: 32" >> $FLINK_CONF
echo "io.tmp.dirs: /tmp/" >> $FLINK_CONF
echo "taskmanager.debug.memory.log: false" >> $FLINK_CONF
echo "taskmanager.debug.memory.log-interval: 500" >> $FLINK_CONF
echo "resourcemanager.taskmanager-timeout: 60000" >> $FLINK_CONF

scontrol show hostname | tail -n +2 > $FLINK_HOME/conf/slaves

$FLINK_HOME/bin/start-cluster.sh

# ./flink-1.8.1/bin/flink run ./flink-1.8.1/examples/gelly/flink-gelly-examples_*.jar --algorithm PageRank --input HypercubeGraph --dimensions 3 --output Print
$FLINK_HOME/bin/flink run CommunityDetection.jar 27

# ./flink-1.8.1/bin/stop-cluster.sh

popd
