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
