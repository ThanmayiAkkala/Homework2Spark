package com.thanu.llm

import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.activations.Activation
import org.apache.spark.rdd.RDD

object EmbeddingUtilsTest {

  def main(args: Array[String]): Unit = {
    // Initialize Spark session for testing
    val spark: SparkSession = SparkSession.builder()
      .appName("EmbeddingUtilsTest")
      .master("local[*]")
      .getOrCreate()

    testDataSetCreationAndFiltering(spark)

    // Stop Spark session after tests
    spark.stop()
  }

  // Test Cases 1-4 as previously provided...

  // Test Case 5: Validate DataSet Creation and Filtering Logic
  def testDataSetCreationAndFiltering(spark: SparkSession): Unit = {
    println("Running test: DataSet creation and filtering")

    val embeddingDim = 3
    val windowSize = 4

    // Generate sample sliding window data
    val sampleWindows: Seq[(INDArray, INDArray)] = Seq(
      (Nd4j.rand(1, windowSize, embeddingDim), Nd4j.rand(1, embeddingDim)),
      (Nd4j.rand(1, windowSize, embeddingDim), Nd4j.rand(1, embeddingDim)),
      (Nd4j.rand(1, windowSize, embeddingDim), Nd4j.rand(1, embeddingDim)),
      (Nd4j.rand(1, windowSize, embeddingDim), Nd4j.rand(1, embeddingDim)),
      (Nd4j.rand(1, windowSize, embeddingDim), Nd4j.rand(1, embeddingDim))
    )

    // Parallelize and create DataSet RDD
    val dataSetRDD: RDD[DataSet] = spark.sparkContext.parallelize(sampleWindows)
      .filter { case (input, target) =>
        input.shape()(1) == windowSize && target.shape()(1) == embeddingDim
      }
      .map { case (input, target) =>
        new DataSet(input.reshape(1, windowSize, embeddingDim), target.reshape(1, embeddingDim))
      }

    // Validate DataSet creation and split
    val Array(trainingData, testData) = dataSetRDD.randomSplit(Array(0.8, 0.2), seed = 12345)

    // Check that DataSet objects have correct shapes
    trainingData.collect().foreach { dataSet =>
      assert(dataSet.getFeatures.shape() sameElements Array(1, windowSize, embeddingDim),
        s"Incorrect feature shape in training data: ${dataSet.getFeatures.shape().mkString(", ")}")
      assert(dataSet.getLabels.shape() sameElements Array(1, embeddingDim),
        s"Incorrect label shape in training data: ${dataSet.getLabels.shape().mkString(", ")}")
    }

    testData.collect().foreach { dataSet =>
      assert(dataSet.getFeatures.shape() sameElements Array(1, windowSize, embeddingDim),
        s"Incorrect feature shape in test data: ${dataSet.getFeatures.shape().mkString(", ")}")
      assert(dataSet.getLabels.shape() sameElements Array(1, embeddingDim),
        s"Incorrect label shape in test data: ${dataSet.getLabels.shape().mkString(", ")}")
    }

    // Confirm the split proportions roughly match 80-20
    assert(trainingData.count() >= 3, "Training set size should be roughly 80% of total")
    assert(testData.count() >= 1, "Test set size should be roughly 20% of total")

    println("Test passed: DataSet creation and filtering")
  }
}
