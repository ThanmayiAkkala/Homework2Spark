package com.thanu.llm

import org.apache.spark.rdd.RDD
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import scala.collection.mutable.ListBuffer


object SlidingWindowProcessor {

  // Spark-compatible function to create sliding windows
//  def createSlidingWindows(embeddingsRDD: RDD[INDArray], windowSize: Int, embeddingDim: Int): RDD[(INDArray, INDArray)] = {
//    embeddingsRDD
//      .mapPartitions(iter => {
//        val embeddings = iter.toArray  // Convert iterator to array for sliding operation
//        val dataSetList = new ListBuffer[(INDArray, INDArray)]()
//
//        for (i <- 0 until embeddings.length - windowSize) {
//          // Extract input embeddings (window of size windowSize)
//          val inputWindow = embeddings.slice(i, i + windowSize)
//          val target = embeddings(i + windowSize)
//
//          // Stack and reshape to 3D format [1, windowSize, embeddingDim]
////          val inputEmbeddings = Nd4j.vstack(inputWindow: _*).reshape(1, windowSize, embeddingDim)
////          val inputEmbeddings = Nd4j.vstack(inputWindow: _*).reshape(1, windowSize, embeddingDim)
////
////          val target3D = target.reshape(1, 1, embeddingDim)
//// Adjusting the sliding window shape to be in [sequence_length, embedding_dim]
//          val inputEmbeddings = Nd4j.vstack(inputWindow: _*).reshape(windowSize, embeddingDim)
//          val target3D = target.reshape(embeddingDim) // Ensures it’s a single 1D array for each sequence end token
//
//
//          dataSetList += ((inputEmbeddings, target3D))
//        }
//        dataSetList.iterator
//      })
//  }
// In SlidingWindowProcessor.scala
def createSlidingWindows(embeddingsRDD: RDD[INDArray], windowSize: Int, embeddingDim: Int): RDD[(INDArray, INDArray)] = {
  embeddingsRDD
    .mapPartitions(iter => {
      val embeddings = iter.toArray
      val dataSetList = new ListBuffer[(INDArray, INDArray)]()

      for (i <- 0 until embeddings.length - windowSize) {
        // Input window
        val inputWindow = embeddings.slice(i, i + windowSize)
        val input3D = Nd4j.vstack(inputWindow: _*).reshape(1, windowSize, embeddingDim) // 3D input [1, windowSize, embeddingDim]

        // Target
        val target = embeddings(i + windowSize).reshape(1, embeddingDim) // 2D target [1, embeddingDim]

        dataSetList += ((input3D, target))
      }
      dataSetList.iterator
    })
}

  // Function to compute positional embeddings (same as original)
  def computePositionalEmbedding(windowSize: Int, embeddingDim: Int): INDArray = {
    val positionalEncoding = Nd4j.zeros(windowSize, embeddingDim)

    for (pos <- 0 until windowSize) {
      for (i <- 0 until embeddingDim by 2) {
        val angle = pos / math.pow(10000, (2.0 * i) / embeddingDim)
        positionalEncoding.putScalar(Array(pos, i), math.sin(angle))
        positionalEncoding.putScalar(Array(pos, i + 1), math.cos(angle))
      }
    }
    positionalEncoding
  }
}
//package com.thanu.llm
//
//import org.apache.spark.rdd.RDD
//import org.apache.spark.sql.SparkSession
//import org.nd4j.linalg.api.ndarray.INDArray
//import org.nd4j.linalg.factory.Nd4j
//import scala.collection.mutable.ListBuffer
//
//object SlidingWindowProcessor {
//
//  // Function to create sliding windows with tokens, window embeddings, and target embeddings
//  def createSlidingWindows(tokensEmbeddingsRDD: RDD[(String, INDArray)], windowSize: Int, embeddingDim: Int): RDD[(Array[INDArray], String, INDArray)] = {
//    tokensEmbeddingsRDD
//      .mapPartitions(iter => {
//        val tokensEmbeddings = iter.toArray  // Convert iterator to array for sliding operation
//        val dataSetList = new ListBuffer[(Array[INDArray], String, INDArray)]()
//
//        for (i <- 0 until tokensEmbeddings.length - windowSize) {
//          // Extract window embeddings and the feature token with target
//          val windowEmbeddings = tokensEmbeddings.slice(i, i + windowSize).map(_._2) // embeddings in window
//          val featureToken = tokensEmbeddings(i)._1                                 // feature token
//          val targetEmbedding = tokensEmbeddings(i + windowSize)._2                 // target embedding
//
//          dataSetList += ((windowEmbeddings, featureToken, targetEmbedding))
//        }
//        dataSetList.iterator
//      })
//  }
// Function to create sliding windows with tokens, window embeddings, and target embeddings
//def createSlidingWindows(tokensEmbeddingsRDD: RDD[(String, INDArray)], windowSize: Int, embeddingDim: Int): RDD[(Array[INDArray], String, INDArray)] = {
//  tokensEmbeddingsRDD
//    .mapPartitions(iter => {
//      val tokensEmbeddings = iter.toArray // Convert iterator to array for sliding operation
//      val dataSetList = new ListBuffer[(Array[INDArray], String, INDArray)]()
//
//      // Only proceed if the tokensEmbeddings array has enough data
//      if (tokensEmbeddings.length > windowSize) {
//        for (i <- 0 until tokensEmbeddings.length - windowSize) {
//          val windowEmbeddings = tokensEmbeddings.slice(i, i + windowSize).map(_._2) // embeddings in window
//          val featureToken = tokensEmbeddings(i)._1 // feature token
//          val targetEmbedding = tokensEmbeddings(i + windowSize)._2 // target embedding
//
//          dataSetList += ((windowEmbeddings, featureToken, targetEmbedding))
//        }
//      } else {
//        println("Warning: Not enough data to create a sliding window for this partition.")
//      }
//      dataSetList.iterator
//    })
//}

  // Function to compute positional embeddings
//  def computePositionalEmbedding(windowSize: Int, embeddingDim: Int): INDArray = {
//    val positionalEncoding = Nd4j.zeros(windowSize, embeddingDim)
//    for (pos <- 0 until windowSize) {
//      for (i <- 0 until embeddingDim by 2) {
//        val angle = pos / math.pow(10000, (2.0 * i) / embeddingDim)
//        positionalEncoding.putScalar(Array(pos, i), math.sin(angle))
//        positionalEncoding.putScalar(Array(pos, i + 1), math.cos(angle))
//      }
//    }
//    positionalEncoding
//  }
//}