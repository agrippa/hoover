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
   * Worker context used with {@link SimplePageRankComputation}.
   */
  public class SimplePageRankWorkerContext extends
      WorkerContext {
    /** Final max value for verification for local jobs */
    private static double FINAL_MAX;
    /** Final min value for verification for local jobs */
    private static double FINAL_MIN;
    /** Final sum value for verification for local jobs */
    private static long FINAL_SUM;

    public static double getFinalMax() {
      return FINAL_MAX;
    }

    public static double getFinalMin() {
      return FINAL_MIN;
    }

    public static long getFinalSum() {
      return FINAL_SUM;
    }

    @Override
    public void preApplication()
      throws InstantiationException, IllegalAccessException {
    }

    @Override
    public void postApplication() {
      FINAL_SUM = this.<LongWritable>getAggregatedValue(SimplePageRankComputationHelper.SUM_AGG).get();
      FINAL_MAX = this.<DoubleWritable>getAggregatedValue(SimplePageRankComputationHelper.MAX_AGG).get();
      FINAL_MIN = this.<DoubleWritable>getAggregatedValue(SimplePageRankComputationHelper.MIN_AGG).get();

      SimplePageRankComputationHelper.LOG.info("aggregatedNumVertices=" + FINAL_SUM);
      SimplePageRankComputationHelper.LOG.info("aggregatedMaxPageRank=" + FINAL_MAX);
      SimplePageRankComputationHelper.LOG.info("aggregatedMinPageRank=" + FINAL_MIN);
    }

    @Override
    public void preSuperstep() {
      if (getSuperstep() >= 3) {
        SimplePageRankComputationHelper.LOG.info("aggregatedNumVertices=" +
            getAggregatedValue(SimplePageRankComputationHelper.SUM_AGG) +
            " NumVertices=" + getTotalNumVertices());
        if (this.<LongWritable>getAggregatedValue(SimplePageRankComputationHelper.SUM_AGG).get() !=
            getTotalNumVertices()) {
          throw new RuntimeException("wrong value of SumAggreg: " +
              getAggregatedValue(SimplePageRankComputationHelper.SUM_AGG) + ", should be: " +
              getTotalNumVertices());
        }
        DoubleWritable maxPagerank = getAggregatedValue(SimplePageRankComputationHelper.MAX_AGG);
        SimplePageRankComputationHelper.LOG.info("aggregatedMaxPageRank=" + maxPagerank.get());
        DoubleWritable minPagerank = getAggregatedValue(SimplePageRankComputationHelper.MIN_AGG);
        SimplePageRankComputationHelper.LOG.info("aggregatedMinPageRank=" + minPagerank.get());
      }
    }

    @Override
    public void postSuperstep() { }
  }

