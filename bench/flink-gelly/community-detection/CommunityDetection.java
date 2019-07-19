import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.graph.Vertex;
import org.apache.flink.graph.Edge;
import org.apache.flink.graph.Graph;
import org.apache.flink.graph.Graph;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.common.functions.MapFunction;

import org.apache.flink.graph.gsa.ApplyFunction;
import org.apache.flink.graph.gsa.GatherFunction;
import org.apache.flink.graph.gsa.Neighbor;
import org.apache.flink.graph.gsa.SumFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.graph.gsa.GSAConfiguration;

import java.util.List;
import java.util.ArrayList;



public class CommunityDetection {

    public static final class ComputeMinVertexValGather extends GatherFunction<Tuple2<Double, Double>, String, Double> {
        public Double gather(Neighbor<Tuple2<Double, Double>, String> neighbor) {
            return neighbor.getNeighborValue().f1;
        }
    }

    public static final class ComputeMinVertexValSum extends SumFunction<Tuple2<Double, Double>, String, Double> {
        public Double sum(Double newValue, Double currentValue) {
            return Math.min(newValue, currentValue);
        }
    }

    public static final class ComputeMinVertexValApply extends ApplyFunction<Long, Tuple2<Double, Double>, Double> {
        public void apply(Double newVal, Tuple2<Double, Double> currentState) {
            setResult(new Tuple2<Double, Double>(currentState.f0, newVal));
        }
    }

    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        int maxIterations = 100;
        int nvertices = 100;

        if (args.length > 0) {
            maxIterations = Integer.parseInt(args[0]);
        }
        if (args.length > 1) {
            nvertices = Integer.parseInt(args[1]);
        }

        List<Vertex<Long, Tuple2<Double, Double>>> vertexList =
            new ArrayList<Vertex<Long, Tuple2<Double, Double>>>();
        for (int i = 0; i < nvertices; i++) {
            vertexList.add(new Vertex<Long, Tuple2<Double, Double>>((long)i,
                        new Tuple2<Double, Double>((double)i, (double)i)));
        }

        List<Edge<Long, String>> edgeList = new ArrayList<Edge<Long, String>>();
        for (int i = 0; i < nvertices-1; i++) {
            edgeList.add(new Edge<Long, String>((long)i, (long)(i+1), "foo"));
        }

        Graph<Long, Tuple2<Double, Double>, String> graph = Graph.fromCollection(
            vertexList, edgeList, env);

        GSAConfiguration config = new GSAConfiguration();

        graph = graph.runGatherSumApplyIteration(new ComputeMinVertexValGather(),
                new ComputeMinVertexValSum(), new ComputeMinVertexValApply(), maxIterations, config);

        // for (int iter = 0; iter < maxIterations; iter++) {
        //     long start = System.currentTimeMillis();
        //     graph = graph.runGatherSumApplyIteration(new ComputeMinVertexValGather(),
        //             new ComputeMinVertexValSum(), new ComputeMinVertexValApply(), 1, config);
        //     long newVertId = (long)nvertices + iter + 1;
        //     Tuple2<Double, Double> newVertVal = new Tuple2<Double, Double>(
        //             (double)newVertId, (double)newVertId);
        //     Vertex<Long, Tuple2<Double, Double>> newVert =
        //         new Vertex<Long, Tuple2<Double, Double>>(newVertId, newVertVal);

        //     // graph = graph.addVertex(newVert);
        //     long elapsed = System.currentTimeMillis() - start;
        //     System.out.println("Iteration " + iter + " took " + elapsed + " ms");
        // }

        long start_post = System.currentTimeMillis();

        DataSet<Vertex<Long, Tuple2<Double, Double>>> updatedVertices =
            graph.getVertices();

        long mid_post1 = System.currentTimeMillis();

        List<Vertex<Long, Tuple2<Double, Double>>> lupdatedVertices =
            updatedVertices.collect();

        long mid_post2 = System.currentTimeMillis();

        for (Vertex<Long, Tuple2<Double, Double>> v : lupdatedVertices) {
            Long id = v.getId();
            Double myVal = v.getValue().f0;
            Double minVal = v.getValue().f1;
            System.out.println(id + " : " + myVal + " : " + minVal);
        }

        long end_post = System.currentTimeMillis();

        System.out.println("Post took " + (end_post -start_post) + " ms, " +
                (mid_post1 - start_post) + " " + (mid_post2 - mid_post1) + " " +
                (end_post - mid_post2));
    }
}
