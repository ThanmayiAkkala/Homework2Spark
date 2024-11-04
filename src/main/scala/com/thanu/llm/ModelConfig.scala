package com.thanu.llm

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.learning.config.Adam

object ModelConfig {

  def createModel(embeddingDim: Int): MultiLayerNetwork = {
    val modelConf = new NeuralNetConfiguration.Builder()
      .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER)
      .updater(new Adam(0.005))
      .list()
      .layer(0, new LSTM.Builder()
        .nIn(embeddingDim)
        .nOut(200)
        .activation(Activation.TANH)
        .build())
      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
        .activation(Activation.SOFTMAX)
        .nIn(200)
        .nOut(embeddingDim)
        .build())
      .build()

    val model = new MultiLayerNetwork(modelConf)
    model.init()
    model
  }
}
