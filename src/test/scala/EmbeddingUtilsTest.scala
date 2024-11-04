package com.thanu.llm

import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.apache.spark.rdd.RDD

object DatasetCreation {

  def main(args: Array[String]): Unit = {
    // Initialize Spark session for testing
    val spark: SparkSession = SparkSession.builder()
      .appName("EmbeddingUtilsTest")
      .master("local[*]")
      .getOrCreate()

    // Test Case 1: Testing `SlidingWindowProcessor.createSlidingWindows`
    testCreateSlidingWindows(spark)

    // Test Case 2: Testing `EmbeddingUtils.loadEmbeddings` with random and absurd values
    testLoadEmbeddings(spark)

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
}
