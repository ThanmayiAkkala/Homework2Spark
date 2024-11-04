package com.thanu.llm

import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.apache.spark.rdd.RDD

object SlidingWindowTest {

  def main(args: Array[String]): Unit = {
    // Initialize Spark session for testing
    val spark: SparkSession = SparkSession.builder()
      .appName("SlidingWindowProcessorTest")
      .master("local[*]")
      .getOrCreate()

    // Test parameters
    val windowSize = 4
    val embeddingDim = 300
    val numEmbeddings = 100  // Increase the number of embeddings

    // Create an RDD of random INDArrays with more embeddings
    val sampleEmbeddings: RDD[INDArray] = spark.sparkContext.parallelize(
      Seq.fill(numEmbeddings)(Nd4j.rand(1, embeddingDim))
    )

    // Check the size of the input data
    println(s"Number of embeddings: ${sampleEmbeddings.count()}")  // Expecting 10

    // Run sliding window creation
    val slidingWindowsRDD = SlidingWindowProcessor.createSlidingWindows(sampleEmbeddings, windowSize, embeddingDim)
    val slidingWindows = slidingWindowsRDD.collect()

    // Debug output for sliding windows
    println(s"Number of sliding windows created: ${slidingWindows.length}")
    slidingWindows.foreach { case (input, target) =>
      println(s"Input shape: ${input.shape().mkString(", ")}, Target shape: ${target.shape().mkString(", ")}")
    }

    // Validate sliding window creation
    if (slidingWindows.length > 0) {
      println(s"Test passed: Sliding windows generated. Number of sliding windows: ${slidingWindows.length}")
    } else {
      println("Test failed: No sliding windows generated.")
    }

    slidingWindows.foreach { case (input, target) =>
      if (input.shape().deep == Array(1, windowSize, embeddingDim).deep && target.shape().deep == Array(1, embeddingDim).deep) {
        println("Test passed: Sliding window shapes are correct.")
      } else {
        println("Test failed: Incorrect sliding window shapes.")
        println(s"Input shape: ${input.shape().mkString(", ")}")
        println(s"Target shape: ${target.shape().mkString(", ")}")
      }
    }

    // Stop Spark session after tests
    spark.stop()
  }
}
