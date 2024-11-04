package com.thanu.llm

import java.util.logging.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import scala.collection.mutable.ListBuffer

object SlidingWindowProcessor {
  val logger: Logger = Logger.getLogger(this.getClass.getName)

  // Function to create sliding windows
  def createSlidingWindows(embeddingsRDD: RDD[INDArray], windowSize: Int, embeddingDim: Int): RDD[(INDArray, INDArray)] = {
    logger.info(s"Creating sliding windows with windowSize=$windowSize and embeddingDim=$embeddingDim.")

    embeddingsRDD
      .mapPartitions(iter => {
        val embeddings = iter.toArray
        val dataSetList = new ListBuffer[(INDArray, INDArray)]()

        logger.info(s"Processing ${embeddings.length} embeddings in partition.")

        for (i <- 0 until embeddings.length - windowSize) {
          // Input window
          val inputWindow = embeddings.slice(i, i + windowSize)
          val input3D = Nd4j.vstack(inputWindow: _*).reshape(1, windowSize, embeddingDim)

          // Target
          val target = embeddings(i + windowSize).reshape(1, embeddingDim)

          // Log shapes for debugging
          logger.fine(s"Generated input3D shape: ${input3D.shape().mkString(",")}, target shape: ${target.shape().mkString(",")}")

          dataSetList += ((input3D, target))
        }

        if (dataSetList.isEmpty) {
          logger.warning("No sliding windows created for this partition.")
        } else {
          logger.info(s"Generated ${dataSetList.size} sliding windows for this partition.")
        }

        dataSetList.iterator
      })
  }

  // Function to compute positional embeddings
  def computePositionalEmbedding(windowSize: Int, embeddingDim: Int): INDArray = {
    logger.info(s"Computing positional embeddings with windowSize=$windowSize and embeddingDim=$embeddingDim.")
    val positionalEncoding = Nd4j.zeros(windowSize, embeddingDim)

    for (pos <- 0 until windowSize) {
      for (i <- 0 until embeddingDim by 2) {
        val angle = pos / math.pow(10000, (2.0 * i) / embeddingDim)
        positionalEncoding.putScalar(Array(pos, i), math.sin(angle))
        positionalEncoding.putScalar(Array(pos, i + 1), math.cos(angle))
      }
    }
    logger.info("Positional embeddings computed successfully.")
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