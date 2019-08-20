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


import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import org.apache.flink.api.common.ProgramDescription;
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.FoldFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.tuple.Tuple;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.graph.Edge;
import org.apache.flink.graph.EdgeDirection;
import org.apache.flink.graph.streaming.EdgesApply;
import org.apache.flink.graph.streaming.SimpleEdgeStream;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.AscendingTimestampExtractor;
import org.apache.flink.streaming.api.functions.windowing.WindowFunction;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.streaming.api.windowing.windows.TimeWindow;
import org.apache.flink.types.NullValue;
import org.apache.flink.util.Collector;
import org.apache.flink.graph.streaming.GraphStream;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.functions.sink.RichSinkFunction;
import org.apache.flink.streaming.api.operators.StreamingRuntimeContext;
import org.apache.flink.streaming.api.functions.source.RichSourceFunction;
import org.apache.flink.streaming.api.functions.source.RichParallelSourceFunction;
import java.util.concurrent.ThreadLocalRandom;


/**
 * Counts exact number of triangles in a graph slice.
 */
public class WindowTriangles implements ProgramDescription {

	public static void main(String[] args) throws Exception {

		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        SimpleEdgeStream<Long, NullValue> edges = getGraphStream(env);
		// GraphStream<Long, NullValue, NullValue> edges = getGraphStream(env);

        DataStream<Tuple2<Integer, Long>> triangleCount = 
        	edges.slice(windowTime, EdgeDirection.ALL)
        	.applyOnNeighbors(new GenerateCandidateEdges())
        	.keyBy(0, 1).timeWindow(windowTime)
			.apply(new CountTriangles())
			.timeWindowAll(windowTime).sum(0);

        triangleCount.print();

        env.execute("Naive window triangle count");
    }


    // *************************************************************************
    //     UTIL METHODS
    // *************************************************************************

	@SuppressWarnings("serial")
	public static final class GenerateCandidateEdges implements
			EdgesApply<Long, NullValue, Tuple3<Long, Long, Boolean>> {

		@Override
		public void applyOnEdges(Long vertexID,
				Iterable<Tuple2<Long, NullValue>> neighbors,
				Collector<Tuple3<Long, Long, Boolean>> out) throws Exception {

			Tuple3<Long, Long, Boolean> outT = new Tuple3<>();
			outT.setField(vertexID, 0);
			outT.setField(false, 2); //isCandidate=false

			Set<Long> neighborIdsSet = new HashSet<Long>();
			for (Tuple2<Long, NullValue> t: neighbors) {
				outT.setField(t.f0, 1);
				out.collect(outT);
				neighborIdsSet.add(t.f0);
			}
			Object[] neighborIds = neighborIdsSet.toArray();
			neighborIdsSet.clear();
			outT.setField(true, 2); //isCandidate=true
			for (int i=0; i<neighborIds.length-1; i++) {
				for (int j=i; j<neighborIds.length; j++) {
					// only emit the candidates
					// with IDs larger than the vertex ID
					if (((long)neighborIds[i] > vertexID) && ((long)neighborIds[j] > vertexID)) {
						outT.setField((long)neighborIds[i], 0);
						outT.setField((long)neighborIds[j], 1);
						out.collect(outT);
					}
				}
			}
		}
	}

	@SuppressWarnings("serial")
	public static final class CountTriangles implements 
			WindowFunction<Tuple3<Long, Long, Boolean>, Tuple2<Integer, Long>, Tuple, TimeWindow>{

		@Override
		public void apply(Tuple key, TimeWindow window,
				Iterable<Tuple3<Long, Long, Boolean>> values,
				Collector<Tuple2<Integer, Long>> out) throws Exception {
			int candidates = 0;
			int edges = 0;
			for (Tuple3<Long, Long, Boolean> t: values) {
				if (t.f2) { // candidate
					candidates++;							
				}
				else {
					edges++;
				}
			}
			if (edges > 0) {
				out.collect(new Tuple2<Integer, Long>(candidates, window.maxTimestamp()));
			}
		}
	}

	private static Time windowTime = Time.of(300, TimeUnit.MILLISECONDS);
    private static long nedges = 200000;
    private static long nvertices = 10000000;

    public static class RandomEdgeSource extends RichParallelSourceFunction<Edge<Long, NullValue>> {
        private volatile boolean isRunning = true;
        private long privateCount = 0;

        @Override
        public void run(SourceContext<Edge<Long, NullValue>> ctx) {
            while (isRunning && privateCount < nedges) {
                ctx.collect(new Edge<>(Math.abs(ThreadLocalRandom.current().nextLong()) % nvertices,
                            Math.abs(ThreadLocalRandom.current().nextLong()) % nvertices,
                            NullValue.getInstance()));
                privateCount++;
            }
            int task = ((StreamingRuntimeContext) getRuntimeContext()).getIndexOfThisSubtask();
            System.out.println(task + "> Source count = " + privateCount);
        }

        @Override
        public void cancel() {
            isRunning = false;
        }
    }


    @SuppressWarnings("serial")
	private static SimpleEdgeStream<Long, NullValue> getGraphStream(StreamExecutionEnvironment env) {
        DataStream<Edge<Long, NullValue>> src = env.addSource(new RandomEdgeSource());
        return new SimpleEdgeStream<Long, NullValue>(src, env);
	}

    @SuppressWarnings("serial")
	public static final class EdgeValueTimestampExtractor extends AscendingTimestampExtractor<Edge<Long, Long>> {
		@Override
		public long extractAscendingTimestamp(Edge<Long, Long> element) {
			return element.getValue();
		}
	}

    @SuppressWarnings("serial")
	public static final class RemoveEdgeValue implements MapFunction<Edge<Long,Long>, NullValue> {
		@Override
		public NullValue map(Edge<Long, Long> edge) {
			return NullValue.getInstance();
		}
	}

    @Override
    public String getDescription() {
        return "Streaming Connected Components on Global Aggregation";
    }
}
