#!/bin/bash

set -e

export FLINK_HOME=$HOME/hoover/bench/flink-gelly/flink-1.8.1
export GELLY_STREAMING_HOME=$HOME/hoover/bench/flink-gelly/gelly-streaming

export CUSTOM_CLASSPATH=$FLINK_HOME/lib/flink-dist_2.11-1.8.1.jar
export CUSTOM_CLASSPATH=$CUSTOM_CLASSPATH:$FLINK_HOME/lib/flink-gelly_2.11-1.8.1.jar
export CUSTOM_CLASSPATH=$CUSTOM_CLASSPATH:$FLINK_HOME/lib/flink-gelly-scala_2.11-1.8.1.jar
export CUSTOM_CLASSPATH=$CUSTOM_CLASSPATH:${GELLY_STREAMING_HOME}/target/flink-gelly-streaming-0.1.0.jar


rm -f *.class
javac -Xlint:deprecation -cp $CUSTOM_CLASSPATH MyConnectedComponentsExample.java
jar cvfm MyConnectedComponentsExample.jar CCManifest.txt *.class

rm -f *.class
javac -Xlint:deprecation -cp $CUSTOM_CLASSPATH WindowTriangles.java
jar cvfm WindowTriangles.jar WTManifest.txt *.class
