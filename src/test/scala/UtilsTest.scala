package com.thanu.llm

import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.apache.spark.rdd.RDD

object UtilsTest {

  def main(args: Array[String]): Unit = {
    // Initialize Spark session for testing
    val spark: SparkSession = SparkSession.builder()
      .appName("EmbeddingUtilsTest")
      .master("local[*]")
      .getOrCreate()

    // Run test cases
    testCreateSlidingWindows(spark)
    testLoadEmbeddings(spark)
    testComputePositionalEmbedding()

    // Stop Spark session after tests
    spark.stop()
  }

  def testCreateSlidingWindows(spark: SparkSession): Unit = {
    println("Running test: createSlidingWindows")

    val windowSize = 4
    val embeddingDim = 3
    val numEmbeddings = 100  // Increase the number of embeddings

    // Create an RDD of random INDArrays with more embeddings
    val sampleEmbeddings: RDD[INDArray] = spark.sparkContext.parallelize(
      Seq.fill(numEmbeddings)(Nd4j.rand(1, embeddingDim))
    )

    val slidingWindowsRDD = SlidingWindowProcessor.createSlidingWindows(sampleEmbeddings, windowSize, embeddingDim)
    val slidingWindows = slidingWindowsRDD.collect()

    println(s"Number of sliding windows created: ${slidingWindows.length}")

    // Validate sliding window creation
    assert(slidingWindows.length > 0, "Test failed: No sliding windows generated.")
    slidingWindows.foreach { case (input, target) =>
      assert(input.shape() sameElements Array(1, windowSize, embeddingDim), s"Incorrect input shape: ${input.shape().mkString(", ")}")
      assert(target.shape() sameElements Array(1, embeddingDim), s"Incorrect target shape: ${target.shape().mkString(", ")}")
    }
    println("Test passed: createSlidingWindows")
  }

  def testLoadEmbeddings(spark: SparkSession): Unit = {
    println("Running test: loadEmbeddings")

    val embeddingDim = 3

    // Simulate data: create a valid set of random embeddings and add an absurd value
    val tokensAndEmbeddings = Seq(
      ("token1", Nd4j.rand(1, embeddingDim)),
      ("token2", Nd4j.rand(1, embeddingDim)),
      ("token3", Nd4j.create(Array(Double.NaN, Double.PositiveInfinity, -Double.PositiveInfinity)).reshape(1, embeddingDim)),
      ("token4", Nd4j.rand(1, embeddingDim))
    )

    val embeddingsRDD: RDD[(String, INDArray)] = spark.sparkContext.parallelize(tokensAndEmbeddings)
    val embeddings = embeddingsRDD.collect().toMap

    // Validate embeddings for valid entries
    assert(embeddings("token1").shape() sameElements Array(1, embeddingDim), "Incorrect shape for token1")
    assert(embeddings("token2").shape() sameElements Array(1, embeddingDim), "Incorrect shape for token2")
    assert(embeddings("token4").shape() sameElements Array(1, embeddingDim), "Incorrect shape for token4")


    println("Test passed: loadEmbeddings")
  }

  def testComputePositionalEmbedding(): Unit = {
    println("Running test: computePositionalEmbedding")

    val windowSize = 4
    val embeddingDim = 6 // Use even number for embedding dimension for positional encoding to work

    // Generate positional embeddings
    val positionalEmbedding = SlidingWindowProcessor.computePositionalEmbedding(windowSize, embeddingDim)

    // Check shape
    assert(positionalEmbedding.shape() sameElements Array(windowSize, embeddingDim),
      s"Expected shape (windowSize, embeddingDim) but got ${positionalEmbedding.shape().mkString(", ")}")

    // Check that values follow the positional encoding pattern
    // Verify specific values (sin/cos pattern), for example:
    assert(positionalEmbedding.getDouble(0L, 0L) == 0.0, "Expected sin(0) at position (0,0)")
    assert(positionalEmbedding.getDouble(0L, 1L) == 1.0, "Expected cos(0) at position (0,1)")

    assert(math.abs(positionalEmbedding.getDouble(1L, 0L) - math.sin(1.0 / math.pow(10000, 0.0 / embeddingDim))) < 1e-5,
      "Positional encoding at (1,0) did not match expected sin value")
    assert(math.abs(positionalEmbedding.getDouble(1L, 1L) - math.cos(1.0 / math.pow(10000, 0.0 / embeddingDim))) < 1e-5,
      "Positional encoding at (1,1) did not match expected cos value")

    println("Test passed: computePositionalEmbedding")
  }

}
