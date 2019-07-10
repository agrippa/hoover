import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.graph.Vertex;
import org.apache.flink.graph.Edge;
import org.apache.flink.graph.Graph;
import org.apache.flink.graph.Graph;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.common.functions.MapFunction;

import org.apache.flink.graph.spargel.ScatterFunction;
import org.apache.flink.graph.spargel.GatherFunction;
import org.apache.flink.graph.spargel.MessageIterator;
import org.apache.flink.api.java.tuple.Tuple2;

import java.util.List;
import java.util.ArrayList;



public class ScatterGather {
    public static final class ComputeMinVertexValScatter extends ScatterFunction<Long, Tuple2<Double, Double>, Double, String> {
        public void sendMessages(Vertex<Long, Tuple2<Double, Double>> vertex) {
            for (Edge<Long, String> edge : getEdges()) {
                sendMessageTo(edge.getTarget(), vertex.getValue().f1);
            }
        }
    }

    public static final class ComputeMinVertexValGather extends GatherFunction<Long, Tuple2<Double, Double>, Double> {
        public void updateVertex(Vertex<Long, Tuple2<Double, Double>> vertex, MessageIterator<Double> messages) {
            double minVal = vertex.getValue().f1;

            for (Double msg : messages) {
                if (msg < minVal) {
                    minVal = msg;
                }
            }

            setNewVertexValue(new Tuple2<Double, Double>(vertex.getValue().f0,
                        minVal));
        }
    }

    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        int maxIterations = 1000;
        int nvertices = 100;

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

        Graph<Long, Tuple2<Double, Double>, String> result =
            graph.runScatterGatherIteration(new ComputeMinVertexValScatter(), new ComputeMinVertexValGather(),
                    maxIterations);

        DataSet<Vertex<Long, Tuple2<Double, Double>>> updatedVertices =
            result.getVertices();

        List<Vertex<Long, Tuple2<Double, Double>>> lupdatedVertices =
            updatedVertices.collect();

        for (Vertex<Long, Tuple2<Double, Double>> v : lupdatedVertices) {
            Long id = v.getId();
            Double myVal = v.getValue().f0;
            Double minVal = v.getValue().f1;
            System.out.println(id + " : " + myVal + " : " + minVal);
        }
    }
}
