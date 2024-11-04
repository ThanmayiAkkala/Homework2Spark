package com.thanu.llm
import java.util.logging.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.api.TrainingMaster
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.evaluation.classification.Evaluation
import org.deeplearning4j.util.ModelSerializer

object MainDriver {
  val logger: Logger = Logger.getLogger(this.getClass.getName)

  def main(args: Array[String]): Unit = {
    var spark: SparkSession = null

    try {
      if (args.length != 3) {
        logger.severe("Usage: LLMEncoderDriver <input_path> <output_path> <embedding_dim>")
        System.exit(0)
      }

      val inputPath = args(0)
      val outputPath = args(1)
      val embeddingDim = args(2).toInt
      val numEpochs = 5 // Number of epochs for training

      // Initialize Spark session
      spark = SparkSession.builder
        .appName("LLM Encoder with Spark")
        //.master("local[*]")
        .getOrCreate()
      spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.outputdir.overwrite", "true")

      val sc = spark.sparkContext

      // Load tokens and create embeddings
      logger.info("Loading data and creating embeddings.")
      val tokensRDD = sc.textFile(inputPath)
      val embeddingsRDD = tokensRDD.map { token =>
        (token, EmbeddingUtils.generateRandomEmbedding(embeddingDim))
      }

      val embeddingValuesRDD = embeddingsRDD.map { case (_, embedding) => embedding }
      val slidingWindowsRDD = SlidingWindowProcessor.createSlidingWindows(embeddingValuesRDD, windowSize = 4, embeddingDim)

      // Save embeddings and sliding windows for inspection
      logger.info("Saving embeddings and sliding windows for inspection.")
      embeddingsRDD.saveAsTextFile(outputPath + "/embeddings")
      slidingWindowsRDD.saveAsTextFile(outputPath + "/sliding_windows")

      val dataSetRDD: RDD[DataSet] = slidingWindowsRDD
        .filter { case (input, target) =>
          input.shape()(0) == 4 && target.shape()(0) == 4
        }
        .map { case (input, target) =>
          new DataSet(input.reshape(1, 4, embeddingDim), target.reshape(1, 4, embeddingDim))
        }

      val nonEmptyDataSetRDD = dataSetRDD.filter(_.numExamples() > 0)
      val Array(trainingData, testData) = nonEmptyDataSetRDD.randomSplit(Array(1, 0.2), seed = 12345)

      // Import the model configuration from ModelConfig and initialize the model
      val model = ModelConfig.createModel(embeddingDim)

      // Training Listeners
      model.setListeners(new ScoreIterationListener(10)) // Log every 10 iterations

      // Distributed Training Setup
      logger.info("Setting up distributed training.")
      val trainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
        .averagingFrequency(5)
        .batchSizePerWorker(32)
        .workerPrefetchNumBatches(2)
        .build()

      val sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster)

      // Training over multiple epochs
      logger.info("Starting epoch-based training.")
      val globalStartTime = System.currentTimeMillis()

      for (epoch <- 1 to numEpochs) {
        logger.info(s"Starting epoch $epoch")
        val epochStartTime = System.currentTimeMillis()

        // Train the model
        try {
          sparkModel.fit(trainingData)
        } catch {
          case e: NullPointerException =>
            logger.log(Level.SEVERE, "Caught NullPointerException during training, likely due to empty partitions.", e)
        }

        val epochEndTime = System.currentTimeMillis()
        logger.info(s"Completed epoch $epoch in ${(epochEndTime - epochStartTime) / 1000.0} seconds")

        // Manual evaluation after each epoch
        logger.info(s"Evaluating model after epoch $epoch.")
        val evaluation = new Evaluation(embeddingDim)
        testData.collect().foreach { data =>
          val output = model.output(data.getFeatures)
          evaluation.eval(data.getLabels, output)
        }
        logger.info(s"Evaluation Metrics for Epoch $epoch:${evaluation.stats()}")
      }

      val globalEndTime = System.currentTimeMillis()
      logger.info(s"Total training time: ${(globalEndTime - globalStartTime) / 1000.0} seconds")

      // Save the trained model
      ModelSerializer.writeModel(model, outputPath + "/LLM_Spark_Model.zip", true)
      logger.info(s"Model saved at ${outputPath}/LLM_Spark_Model.zip")

      // Spark Training Metrics for Performance Insights
      val trainingStats = sparkModel.getSparkTrainingStats
      if (trainingStats != null) {
        logger.info("Training statistics:" + trainingStats.statsAsString())
      } else {
        logger.warning("No training statistics available.")
      }

    } catch {
      case e: Exception =>
        logger.log(Level.SEVERE, "An unexpected error occurred", e)

    } finally {
      if (spark != null) {
        spark.stop()
      }
      logger.info("Shutting down Spark session.")
      System.exit(0)
    }
  }
}

