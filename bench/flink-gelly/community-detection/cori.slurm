#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 64
#SBATCH -C haswell
#SBATCH -t 00:30:00
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
export GELLY_STREAMING_HOME=$HOME/hoover/bench/flink-gelly/gelly-streaming

# module load spark/2.3.0
# module load scala/2.11.8
module load jdk/1.8.0_202

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
NWORKERS=$(($NHOSTS - 1))

MASTER_NODE=$(scontrol show hostname | head -n 1)

# Extracted from flink-1.8.1-bin-scala_2.11.tgz
rm $FLINK_HOME/log/*

rm -f *.jar

export CUSTOM_CLASSPATH=$FLINK_HOME/lib/flink-dist_2.11-1.8.1.jar
export CUSTOM_CLASSPATH=$CUSTOM_CLASSPATH:$FLINK_HOME/lib/flink-gelly_2.11-1.8.1.jar
export CUSTOM_CLASSPATH=$CUSTOM_CLASSPATH:$FLINK_HOME/lib/flink-gelly-scala_2.11-1.8.1.jar
export CUSTOM_CLASSPATH=$CUSTOM_CLASSPATH:${GELLY_STREAMING_HOME}/target/flink-gelly-streaming-0.1.0.jar

# rm -f *.class
# javac -Xlint:deprecation -cp $CUSTOM_CLASSPATH CommunityDetection.java
# jar cvfm CommunityDetection.jar Manifest.txt *.class

rm -f *.class
javac -Xlint:deprecation -cp $CUSTOM_CLASSPATH MyConnectedComponentsExample.java
jar cvfm MyConnectedComponentsExample.jar CCManifest.txt *.class

# rm -f *.class
# javac -Xlint:deprecation -cp $CUSTOM_CLASSPATH WindowTriangles.java
# jar cvfm WindowTriangles.jar WTManifest.txt *.class
 
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
cat $FLINK_CONF | grep -v "^taskmanager.network.memory.max:" > ${FLINK_CONF}.tmp
mv ${FLINK_CONF}.tmp $FLINK_CONF
cat $FLINK_CONF | grep -v "^taskmanager.network.memory.min:" > ${FLINK_CONF}.tmp
mv ${FLINK_CONF}.tmp $FLINK_CONF
cat $FLINK_CONF | grep -v "^taskmanager.network.memory.fraction:" > ${FLINK_CONF}.tmp
mv ${FLINK_CONF}.tmp $FLINK_CONF
cat $FLINK_CONF | grep -v "^slot.request.timeout:" > ${FLINK_CONF}.tmp
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
echo "taskmanager.network.memory.max: 6gb" >> $FLINK_CONF
echo "taskmanager.network.memory.min: 6gb" >> $FLINK_CONF
echo "taskmanager.network.memory.fraction: 0.2" >> $FLINK_CONF
echo "slot.request.timeout: 600000" >> $FLINK_CONF

scontrol show hostname | tail -n +2 > $FLINK_HOME/conf/slaves

$FLINK_HOME/bin/start-cluster.sh

echo "Running with parallelism $(($NWORKERS * 32))"

export MERGE_WINDOW_TIME=1000
export PRINT_WINDOW_TIME=2000
# export EXECUTION_TIME=30000
export EXECUTION_TIME=60000
export NVERTICES=10000000

# ./flink-1.8.1/bin/flink run ./flink-1.8.1/examples/gelly/flink-gelly-examples_*.jar --algorithm PageRank --input HypercubeGraph --dimensions 3 --output Print
# $FLINK_HOME/bin/flink run CommunityDetection.jar 27
$FLINK_HOME/bin/flink run --parallelism $(($NWORKERS * 32)) \
    MyConnectedComponentsExample.jar $MERGE_WINDOW_TIME $PRINT_WINDOW_TIME \
    $EXECUTION_TIME $NVERTICES
# $FLINK_HOME/bin/flink run WindowTriangles.jar

$FLINK_HOME/bin/stop-cluster.sh

popd
