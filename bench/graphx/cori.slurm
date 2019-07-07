#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 00:02:00
#SBATCH -J graphx
#SBATCH --exclusive
####SBATCH --contiguous
#SBATCH --mail-type=ALL
#SBATCH --image=nersc/spark-2.3.0:v1
#SBATCH --volume="/global/cscratch1/sd/<user_name>/tmpfiles:/tmp:perNodeCache=size=200G"

# set -e

ulimit -c unlimited

echo "Running on:"
echo $SLURM_NODELIST

export EXEC_CLASSPATH=
module load spark/2.3.0
module load scala/2.11.8
# module load java/jdk1.8.0_51

echo
echo

mkdir -p $SCRATCH/job.$SLURM_JOB_ID
pushd $SCRATCH/job.$SLURM_JOB_ID

export SPARK_HOME=/usr/local/bin/spark-2.3.0
export CLASSPATH=$SPARK_HOME/jars/spark-core_2.11-2.3.0.jar:$CLASSPATH
export CLASSPATH=$SPARK_HOME/jars/spark-sql_2.11-2.3.0.jar:$CLASSPATH
export CLASSPATH=$SPARK_HOME/jars/scala-library-2.11.8.jar:$CLASSPATH
export CLASSPATH=$SPARK_HOME/jars/hadoop-common-2.7.3.jar:$CLASSPATH
export CLASSPATH=$SPARK_HOME/jars/hadoop-mapreduce-client-core-2.7.3.jar:$CLASSPATH
export CLASSPATH=$SPARK_HOME/jars/apache-log4j-extras-1.2.17.jar:$CLASSPATH
export CLASSPATH=$SPARK_HOME/jars/guava-14.0.1.jar:$CLASSPATH

start-all.sh

pushd /global/homes/j/jmg3/hoover/bench/giraph
# for F in $(shifter ls /usr/local/bin/spark-2.3.0/jars/); do
#     echo $F
#     shifter jar tf /usr/local/bin/spark-2.3.0/jars/$F | grep ImmutableSet
# done

which scalac
shifter which scalac
shifter scalac PageRankExample.scala
shifter jar -cvf PageRankExample.jar \
    PageRankExample.class
echo 'Done compiling!'

# shifter spark-submit \
#     --class PageRankExample \
#     PageRankExample.jar

popd

stop-all.sh
