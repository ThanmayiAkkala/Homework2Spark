//package com.thanu.llm
//
//import scala.util.Random
//import org.apache.spark.rdd.RDD
//import org.apache.spark.sql.SparkSession
//import org.nd4j.linalg.api.ndarray.INDArray
//import org.nd4j.linalg.factory.Nd4j
//
//object EmbeddingUtils {
//  // Function to generate a random embedding vector of a given dimension
//  def generateRandomEmbedding(dim: Int): INDArray = {
//    Nd4j.rand(1, dim)
//  }
//
//  // Function to load embeddings
//  // This is used in this assignment
//  def loadEmbeddings(spark: SparkSession, path: String, embeddingDim: Int): RDD[(String, INDArray)] = {
//    val embeddingsDF = spark.read
//      .option("header", "true")
//      .csv(path)
//
//    embeddingsDF.rdd.map(row => {
//      val token = row.getString(0)
//      val embedding = Nd4j.create(row.toSeq.tail.map(_.toString.toDouble).toArray).reshape(1, embeddingDim)
//      (token, embedding)
//    })
//  }
//}
package com.thanu.llm

import java.util.logging.{Level, Logger}
import scala.util.Random
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object EmbeddingUtils {
  val logger: Logger = Logger.getLogger(this.getClass.getName)

  // Function to generate a random embedding vector of a given dimension
  def generateRandomEmbedding(dim: Int): INDArray = {

    Nd4j.rand(1, dim)
  }

  // Function to load embeddings from a CSV file
  def loadEmbeddings(spark: SparkSession, path: String, embeddingDim: Int): RDD[(String, INDArray)] = {
    logger.info(s"Loading embeddings from path: $path with embedding dimension $embeddingDim.")

    try {
      val embeddingsDF = spark.read
        .option("header", "true")
        .csv(path)

      embeddingsDF.rdd.map(row => {
        val token = row.getString(0)
        val embedding = try {
          Nd4j.create(row.toSeq.tail.map(_.toString.toDouble).toArray).reshape(1, embeddingDim)
        } catch {
          case e: NumberFormatException =>
            logger.log(Level.SEVERE, s"Error parsing embedding values for token: $token", e)
            Nd4j.zeros(1, embeddingDim)  // Return a zero embedding in case of error
        }
        (token, embedding)
      }).filter { case (token, embedding) =>
        val isValid = embedding.shape()(1) == embeddingDim
        if (!isValid) {
          logger.warning(s"Invalid embedding dimension for token '$token'. Expected $embeddingDim but got ${embedding.shape()(1)}.")
        }
        isValid
      }
    } catch {
      case e: Exception =>
        logger.log(Level.SEVERE, s"Failed to load embeddings from path: $path", e)
        spark.sparkContext.emptyRDD[(String, INDArray)]
    }
  }
}
