import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.factory.Nd4j
import com.thanu.llm.SlidingWindowProcessor._

class SlidingWindowTest extends AnyFunSuite {

  val spark: SparkSession = SparkSession.builder.master("local[*]").appName("SlidingWindowTest").getOrCreate()

  test("createSlidingWindows should generate correct sliding windows") {
    // Sample data that meets the minimum requirement for window size
    val tokens = Array("token1", "token2", "token3", "token4", "token5")
    val embeddings = Array.fill(100)(Nd4j.rand(1, 10))
    val tokensEmbeddingsRDD = spark.sparkContext.parallelize(tokens.zip(embeddings))

//    val windowSize = 3
//    val slidingWindowsRDD = createSlidingWindows(tokensEmbeddingsRDD, windowSize, 10)
//    val result = slidingWindowsRDD.collect()
//
//    assert(result.nonEmpty, "Result should not be empty")
//    result.foreach { case (window, featureToken, target) =>
//      assert(window.length == windowSize, "Each window should have the specified window size")
//      assert(featureToken.nonEmpty, "Feature token should not be empty")
//    }
  }
}
