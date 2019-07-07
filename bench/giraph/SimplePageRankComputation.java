/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// package org.apache.giraph.examples;

import org.apache.giraph.job.GiraphJob;
import org.apache.giraph.aggregators.DoubleMaxAggregator;
import org.apache.giraph.aggregators.DoubleMinAggregator;
import org.apache.giraph.aggregators.LongSumAggregator;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.giraph.io.VertexReader;
import org.apache.giraph.io.formats.GeneratedVertexInputFormat;
import org.apache.giraph.io.formats.TextVertexOutputFormat;
import org.apache.giraph.master.DefaultMasterCompute;
import org.apache.giraph.worker.WorkerContext;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.log4j.Logger;
import org.apache.hadoop.util.ToolRunner;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.conf.Configuration;

import com.google.common.collect.Lists;

import java.io.IOException;
import java.util.List;

/**
 * Demonstrates the basic Pregel PageRank implementation.
 */
@Algorithm(
    name = "Page rank"
)

public class SimplePageRankComputation implements Tool {

    private Configuration conf;
    @Override
    public Configuration getConf() {
        return conf;
    }
    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }


    @Override
    public int run(String[] args) throws Exception {
        int nworkers = 1;

        GiraphJob job = new GiraphJob(getConf(), getClass().getName());
        job.getConfiguration().setComputationClass(SimplePageRankComputationHelper.class);
        job.getConfiguration().setVertexInputFormatClass(
                SimplePageRankVertexInputFormat.class);
        job.getConfiguration().setWorkerContextClass(
                SimplePageRankWorkerContext.class);
        job.getConfiguration().setWorkerConfiguration(nworkers, nworkers, 100.0f);
        if (job.run(true)) {
            return 0;
        } else {
            return -1;
        }
    }

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new SimplePageRankComputation(), args));
    }
}
