package com.thanu.llm

import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.activations.Activation
import org.apache.spark.rdd.RDD

object ModelTest {

  def main(args: Array[String]): Unit = {
    // Initialize Spark session for testing
    val spark: SparkSession = SparkSession.builder()
      .appName("EmbeddingUtilsTest")
      .master("local[*]")
      .getOrCreate()

    // Run test cases
    testModelConfiguration()

    // Stop Spark session after tests
    spark.stop()
  }

  // Test Case 4: Validate Model Configuration
  def testModelConfiguration(): Unit = {
    println("Running test: model configuration")

    val embeddingDim = 3
    val lstmLayerSize = 200
    val outputLayerSize = embeddingDim

    // Model Configuration
    val modelConf = new NeuralNetConfiguration.Builder()
      .updater(new Adam(0.005))
      .list()
      .layer(0, new LSTM.Builder()
        .nIn(embeddingDim)
        .nOut(lstmLayerSize)
        .activation(Activation.TANH)
        .build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .nIn(lstmLayerSize)
        .nOut(outputLayerSize)
        .activation(Activation.SOFTMAX)
        .build())
      .build()

    val model = new MultiLayerNetwork(modelConf)
    model.init()

    // Validate the model configuration
    val lstmLayer = model.getLayer(0).conf().getLayer.asInstanceOf[LSTM]
    val outputLayer = model.getLayer(1).conf().getLayer.asInstanceOf[RnnOutputLayer]

    // Assert LSTM layer configuration
    assert(lstmLayer.getNIn == embeddingDim, s"Expected LSTM input size $embeddingDim but got ${lstmLayer.getNIn}")
    assert(lstmLayer.getNOut == lstmLayerSize, s"Expected LSTM output size $lstmLayerSize but got ${lstmLayer.getNOut}")
    assert(lstmLayer.getActivationFn == Activation.TANH.getActivationFunction, "Expected LSTM activation TANH")

    // Assert Output layer configuration
    assert(outputLayer.getNIn == lstmLayerSize, s"Expected Output layer input size $lstmLayerSize but got ${outputLayer.getNIn}")
    assert(outputLayer.getNOut == outputLayerSize, s"Expected Output layer output size $outputLayerSize but got ${outputLayer.getNOut}")
    assert(outputLayer.getActivationFn == Activation.SOFTMAX.getActivationFunction, "Expected Output layer activation SOFTMAX")
    println("Test passed: model configuration")
  }
}
