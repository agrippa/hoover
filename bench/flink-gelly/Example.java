import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.graph.Vertex;
import org.apache.flink.graph.Edge;
import org.apache.flink.graph.Graph;
import org.apache.flink.graph.Graph;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.common.functions.MapFunction;

import java.util.List;
import java.util.ArrayList;

public class Example {
    public static void main(String[] args) throws Exception {
        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        List<Vertex<Long, Long>> vertexList = new ArrayList<Vertex<Long, Long>>();
        vertexList.add(new Vertex<Long, Long>(3L, 3L));
        vertexList.add(new Vertex<Long, Long>(4L, 4L));
        vertexList.add(new Vertex<Long, Long>(5L, 5L));

        List<Edge<Long, String>> edgeList = new ArrayList<Edge<Long, String>>();
        edgeList.add(new Edge<Long, String>(3L, 4L, "foo"));
        edgeList.add(new Edge<Long, String>(4L, 5L, "bar"));

        Graph<Long, Long, String> graph = Graph.fromCollection(vertexList,
                edgeList, env);

        Graph<Long, Long, String> updatedGraph = graph.mapVertices(
                                new MapFunction<Vertex<Long, Long>, Long>() {
                                    public Long map(Vertex<Long, Long> value) {
                                        return value.getValue() + 1;
                                    }
                                });

        DataSet<Vertex<Long, Long>> updatedVertices = updatedGraph.getVertices();

        List<Vertex<Long, Long>> lupdatedVertices = updatedVertices.collect();

        for (Vertex<Long, Long> v : lupdatedVertices) {
            Long id = v.getId();
            Long val = v.getValue();
            System.out.println(id + " : " + val);
        }
    }
}
