package com.thanu.llm

import scala.util.Random
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object EmbeddingUtils {
  // Function to generate a random embedding vector of a given dimension
  def generateRandomEmbedding(dim: Int): INDArray = {
    Nd4j.rand(1, dim)
  }

  // Function to load embeddings from a file if provided as a CSV
  def loadEmbeddings(spark: SparkSession, path: String, embeddingDim: Int): RDD[(String, INDArray)] = {
    val embeddingsDF = spark.read
      .option("header", "true")
      .csv(path)

    embeddingsDF.rdd.map(row => {
      val token = row.getString(0)
      val embedding = Nd4j.create(row.toSeq.tail.map(_.toString.toDouble).toArray).reshape(1, embeddingDim)
      (token, embedding)
    })
  }
}
