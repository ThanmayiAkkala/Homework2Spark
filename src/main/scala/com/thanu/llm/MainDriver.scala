////package com.thanu.llm
////
////import org.apache.spark.sql.SparkSession
////import org.nd4j.linalg.api.ndarray.INDArray
////import org.nd4j.linalg.factory.Nd4j
////
////
////import org.apache.spark.rdd.RDD
////
////object LLMEncoderDriver {
////  def main(args: Array[String]): Unit = {
////    if (args.length != 3) {
////      System.err.println("Usage: LLMEncoderDriver <input_path> <output_path> <embedding_dim>")
////      System.exit(-1)
////    }
////
////    val inputPath = args(0)
////    val outputPath = args(1)
////    val embeddingDim = args(2).toInt
////
////    // Initialize Spark session
////    val spark = SparkSession.builder
////      .appName("LLM Encoder with Spark")
////      .master("local[*]")
////      .getOrCreate()
////
////    // Load tokens and create embeddings
////    val tokensRDD = spark.sparkContext.textFile(inputPath)
////    val embeddingsRDD = tokensRDD.map { token =>
////      (token, EmbeddingUtils.generateRandomEmbedding(embeddingDim))
////    }
////
////    // Extract only the embeddings (INDArray) for sliding window processing
////    val embeddingValuesRDD = embeddingsRDD.map { case (_, embedding) => embedding }
////
////    // Process sliding windows with the specified embedding dimension
////    val slidingWindowsRDD = SlidingWindowProcessor.createSlidingWindows(embeddingValuesRDD, windowSize = 4, embeddingDim)
////
////    // Save outputs for inspection
////    embeddingsRDD.saveAsTextFile(outputPath + "/embeddings")
////    slidingWindowsRDD.saveAsTextFile(outputPath + "/sliding_windows")
////
////    spark.stop()
////  }
////}
////object LLMEncoderDriver {
////
////  def main(args: Array[String]): Unit = {
////    val spark = SparkSession.builder.appName("LLM Encoder").getOrCreate()
////    val sc = spark.sparkContext
////
////    // Load tokens and embeddings as an RDD
////    val embeddingDim = 10
////    val tokens = Array("example", "of", "tokenized", "data") // Replace with actual tokens
////    val embeddings = Array.fill(tokens.length)(EmbeddingUtils.generateRandomEmbedding(embeddingDim)) // Replace with actual embeddings
////    val tokensEmbeddingsRDD: RDD[(String, INDArray)] = sc.parallelize(tokens.zip(embeddings))
////
////    // Set the sliding window size
////    val windowSize = 3
////
////    // Generate sliding windows with feature tokens, embeddings, and target embeddings
////    val slidingWindowsRDD = SlidingWindowProcessor.createSlidingWindows(tokensEmbeddingsRDD, windowSize, embeddingDim)
////
////    // Save sliding windows with tokens and embeddings for inspection
////    slidingWindowsRDD.map { case (window, featureToken, targetEmbedding) =>
////      // Updated line with specific toString method
////      s"Window Embeddings: ${window.map(_.data().asDouble().mkString(",")).mkString("; ")} | Feature Token: $featureToken | Target Embedding: ${targetEmbedding.data().asDouble().mkString(",")}"
////
////    }.saveAsTextFile("output/sliding_windows_output.txt")
////
////    // Prepare model input with 3D tensor of only embeddings
////    val modelInput = slidingWindowsRDD.map(_._1).collect() // Collect 3D array of embeddings for model input
////
////    println(s"Model input tensor dimensions: (${modelInput.length}, ${modelInput.head.length}, ${modelInput.head.head.columns()})")
////
////    spark.stop()
////  }
////}
//package com.thanu.llm
//
//import org.apache.spark.rdd.RDD
//import org.apache.spark.sql.SparkSession
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration
//import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
//import org.deeplearning4j.optimize.listeners.ScoreIterationListener
//import org.deeplearning4j.spark.api.TrainingMaster
//import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
//import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
//import org.nd4j.linalg.activations.Activation
//import org.nd4j.linalg.dataset.DataSet
//import org.nd4j.linalg.lossfunctions.LossFunctions
//import org.nd4j.linalg.learning.config.Adam
//import org.nd4j.evaluation.classification.Evaluation
//import org.deeplearning4j.util.ModelSerializer
//
//
//object LLMEncoderDriver {
//  def main(args: Array[String]): Unit = {
//    if (args.length != 3) {
//      System.err.println("Usage: LLMEncoderDriver <input_path> <output_path> <embedding_dim>")
//      System.exit(-1)
//    }
//
//    val inputPath = args(0)
//    val outputPath = args(1)
//    val embeddingDim = args(2).toInt
//
//    // Initialize Spark session
//    val spark = SparkSession.builder
//      .appName("LLM Encoder with Spark")
//      .master("local[*]")
//      .getOrCreate()
//    val sc = spark.sparkContext
//
//    // Load tokens and create embeddings
//    val tokensRDD = spark.sparkContext.textFile(inputPath)
//    val embeddingsRDD = tokensRDD.map { token =>
//      (token, EmbeddingUtils.generateRandomEmbedding(embeddingDim))
//    }
//
//    // Extract only the embeddings (INDArray) for sliding window processing
//    val embeddingValuesRDD = embeddingsRDD.map { case (_, embedding) => embedding }
//
//    // Process sliding windows with the specified embedding dimension
//    val slidingWindowsRDD = SlidingWindowProcessor.createSlidingWindows(embeddingValuesRDD, windowSize = 4, embeddingDim)
//
//    // Save embeddings and sliding windows for inspection
//    embeddingsRDD.saveAsTextFile(outputPath + "/embeddings")
//    slidingWindowsRDD.saveAsTextFile(outputPath + "/sliding_windows")
//
//    // Convert sliding windows to DataSet format for DL4J training
////    val dataSetRDD: RDD[DataSet] = slidingWindowsRDD.map { case (input, target) =>
////      new DataSet(input, target)
////    }this is 1
////val dataSetRDD: RDD[DataSet] = slidingWindowsRDD.map { case (input, target) =>
////  new DataSet(input.reshape(1, input.shape()(0), embeddingDim), target.reshape(1, 1, embeddingDim))
////}this is the 2
//val dataSetRDD: RDD[DataSet] = slidingWindowsRDD
//  .filter { case (input, target) =>
//    // Ensure both input and target have the correct shape
//    input.shape()(0) == 4 && target.shape()(0) == 4
//  }
//  .map { case (input, target) =>
//    new DataSet(input.reshape(1, 4, embeddingDim), target.reshape(1, 4, embeddingDim))
//  }
//
//
//
//    slidingWindowsRDD.take(5).foreach { case (input, target) =>
//      println(s"Input shape: ${input.shape().mkString(",")}")
//      println(s"Target shape: ${target.shape().mkString(",")}")
//    }
//
//    // Model Configuration
//    val modelConf = new NeuralNetConfiguration.Builder()
//      .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER)
//      .updater(new Adam(0.005))
//      .list()
//      .layer(0, new LSTM.Builder()
//        .nIn(300) // Set to embeddingDim only
//        .nOut(200)
//        .activation(Activation.TANH)
//        .build())
//      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//        .activation(Activation.SOFTMAX)
//        .nIn(200)
//        .nOut(300)
//        .build())
//      .build()
//
//    val model = new MultiLayerNetwork(modelConf)
//    model.init()
//    model.setListeners(new ScoreIterationListener(10))
//
//    // Distributed Training Setup
//    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
//      .averagingFrequency(5)
//      .batchSizePerWorker(32)
//      .workerPrefetchNumBatches(2)
//      .build()
//    // Filter out empty datasets to avoid NullPointerException
//    val nonEmptyDataSetRDD = dataSetRDD.filter(_.numExamples() > 0)
//
//    // Proceed with training using the non-empty dataset
//    println("Starting training with non-empty data...")
//
//
//
//
//    val sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster)
//    println(s"Number of partitions: ${dataSetRDD.getNumPartitions}")
//    dataSetRDD.foreachPartition(partition => {
//      if (!partition.hasNext) println("Empty partition found")
//    })
//
//    // Training and Metric Collection
//    println("Starting training...")
//    val startTime = System.currentTimeMillis()
//   // sparkModel.fit(nonEmptyDataSetRDD)
//    try {
//      sparkModel.fit(nonEmptyDataSetRDD)
//    } catch {
//      case e: NullPointerException =>
//        println("Caught NullPointerException during training, likely due to empty partitions. Ensure non-empty data partitions.")
//        e.printStackTrace()
//    }
//    val endTime = System.currentTimeMillis()
//    println(s"Training completed in ${(endTime - startTime) / 1000.0} seconds")
//
//    // Model Evaluation (Optional - if you have test data)
//    val evaluation = new Evaluation(embeddingDim)
//    dataSetRDD.collect().foreach { data =>
//      val output = model.output(data.getFeatures)
//      evaluation.eval(data.getLabels, output)
//    }
//    println(evaluation.stats())
//
//    // Save Model
//    ModelSerializer.writeModel(model, outputPath + "/LLM_Spark_Model.zip", true)
//    println(s"Model saved at ${outputPath}/LLM_Spark_Model.zip")
//
//    // Print Spark Training Metrics
//    val trainingStats = sparkModel.getSparkTrainingStats
//    println("Training statistics:")
//    println(trainingStats.statsAsString())
//
//    // Stop Spark Context
//    spark.stop()
//  }
//}
//package com.thanu.llm
//
//import org.apache.spark.rdd.RDD
//import org.apache.spark.sql.SparkSession
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration
//import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
//import org.deeplearning4j.optimize.listeners.ScoreIterationListener
//import org.deeplearning4j.spark.api.TrainingMaster
//import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
//import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
//import org.nd4j.linalg.activations.Activation
//import org.nd4j.linalg.dataset.DataSet
//import org.nd4j.linalg.lossfunctions.LossFunctions
//import org.nd4j.linalg.learning.config.Adam
//import org.nd4j.evaluation.classification.Evaluation
//import org.deeplearning4j.util.ModelSerializer
//
//object LLMEncoderDriver {
//  def main(args: Array[String]): Unit = {
//    if (args.length != 3) {
//      System.err.println("Usage: LLMEncoderDriver <input_path> <output_path> <embedding_dim>")
//      System.exit(-1)
//    }
//
//    val inputPath = args(0)
//    val outputPath = args(1)
//    val embeddingDim = args(2).toInt
//
//    // Initialize Spark session
//    val spark = SparkSession.builder
//      .appName("LLM Encoder with Spark")
//      .master("local[*]")
//      .config("spark.local.dir", "C:\\Bigdata")  // Adjust path if needed
//      .getOrCreate()
//    spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.outputdir.overwrite", "true")
//
//    val sc = spark.sparkContext
//
//    // Load tokens and create embeddings
//    val tokensRDD = spark.sparkContext.textFile(inputPath)
//    val embeddingsRDD = tokensRDD.map { token =>
//      (token, EmbeddingUtils.generateRandomEmbedding(embeddingDim))
//    }
//
//    // Extract only the embeddings (INDArray) for sliding window processing
//    val embeddingValuesRDD = embeddingsRDD.map { case (_, embedding) => embedding }
//
//    // Process sliding windows with the specified embedding dimension
//    val slidingWindowsRDD = SlidingWindowProcessor.createSlidingWindows(embeddingValuesRDD, windowSize = 4, embeddingDim)
//
//    // Save embeddings and sliding windows for inspection
//    embeddingsRDD.saveAsTextFile(outputPath + "/embeddings")
//    slidingWindowsRDD.saveAsTextFile(outputPath + "/sliding_windows")
//
//    // Convert sliding windows to DataSet format for DL4J training
//    val dataSetRDD: RDD[DataSet] = slidingWindowsRDD
//      .filter { case (input, target) =>
//        input.shape()(0) == 4 && target.shape()(0) == 4
//      }
//      .map { case (input, target) =>
//        new DataSet(input.reshape(1, 4, embeddingDim), target.reshape(1, 4, embeddingDim))
//      }
//
//    val nonEmptyDataSetRDD = dataSetRDD.filter(_.numExamples() > 0)
//
//    // Split dataset into 80% training and 20% testing
//    val Array(trainingData, testData) = nonEmptyDataSetRDD.randomSplit(Array(0.8, 0.2), seed = 12345)
//
//    // Model Configuration
//    val modelConf = new NeuralNetConfiguration.Builder()
//      .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER)
//      .updater(new Adam(0.005))
//      .list()
//      .layer(0, new LSTM.Builder()
//        .nIn(embeddingDim)
//        .nOut(200)
//        .activation(Activation.TANH)
//        .build())
//      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//        .activation(Activation.SOFTMAX)
//        .nIn(200)
//        .nOut(embeddingDim)
//        .build())
//      .build()
//
//    val model = new MultiLayerNetwork(modelConf)
//    model.init()
//    println("setlistners")
//    model.setListeners(new ScoreIterationListener(10))
//
//    // Distributed Training Setup
//    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
//      .averagingFrequency(5)
//      .batchSizePerWorker(32)
//      .workerPrefetchNumBatches(2)
//      .build()
//
//    val sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster)
//
//    // Training and Metric Collection
//    println("Starting training...")
//    val startTime = System.currentTimeMillis()
//    try {
//      sparkModel.fit(trainingData)
//    } catch {
//      case e: NullPointerException =>
//        println("Caught NullPointerException during training, likely due to empty partitions.")
//        e.printStackTrace()
//    }
//    val endTime = System.currentTimeMillis()
//    println(s"Training completed in ${(endTime - startTime) / 1000.0} seconds")
//
//    // Model Evaluation on Test Data
//    val evaluation = new Evaluation(embeddingDim)
//    try {
//      val evaluationDataSetRDD = testData.collect()
//      if (evaluationDataSetRDD.nonEmpty) {
//        evaluationDataSetRDD.foreach { data =>
//          val output = model.output(data.getFeatures)
//          evaluation.eval(data.getLabels, output)
//        }
//        println("Evaluation Metrics on Test Data:")
//        println(evaluation.stats())
//      } else {
//        println("No data available for evaluation.")
//      }
//    } catch {
//      case e: NullPointerException =>
//        println("Caught NullPointerException during evaluation, likely due to empty partitions.")
//        e.printStackTrace()
//    }
//
//    // Save Model
//    ModelSerializer.writeModel(model, outputPath + "/LLM_Spark_Model.zip", true)
//    println(s"Model saved at ${outputPath}/LLM_Spark_Model.zip")
//
//    // Print Spark Training Metrics
//    val trainingStats = sparkModel.getSparkTrainingStats
//    if (trainingStats != null) {
//      println("Training statistics:")
//      println(trainingStats.statsAsString())
//    } else {
//      println("No training statistics available.")
//    }
//
//    // Stop Spark Context
//    spark.stop()
//  }
//}
//package com.thanu.llm
//import java.util.logging.{Level, Logger}
//import org.apache.spark.rdd.RDD
//import org.apache.spark.sql.SparkSession
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration
//import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
//import org.deeplearning4j.optimize.listeners.ScoreIterationListener
//import org.deeplearning4j.spark.api.TrainingMaster
//import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
//import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
//import org.nd4j.linalg.activations.Activation
//import org.nd4j.linalg.dataset.DataSet
//import org.nd4j.linalg.lossfunctions.LossFunctions
//import org.nd4j.linalg.learning.config.Adam
//import org.nd4j.evaluation.classification.Evaluation
//import org.deeplearning4j.util.ModelSerializer
//
//object MainDriver {
//  val logger: Logger = Logger.getLogger(this.getClass.getName)
//
//  def main(args: Array[String]): Unit = {
//    if (args.length != 3) {
//      logger.severe("Usage: LLMEncoderDriver <input_path> <output_path> <embedding_dim>")
//      System.exit(-1)
//    }
//
//    val inputPath = args(0)
//    val outputPath = args(1)
//    val embeddingDim = args(2).toInt
//    val numEpochs = 5 // Number of epochs for training
//
//    // Initialize Spark session
//    val spark = SparkSession.builder
//      .appName("LLM Encoder with Spark")
//      .master("local[*]")
//      //.config("spark.local.dir", "C:\\Bigdata")  // Adjust path if needed
//      .getOrCreate()
//    spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.outputdir.overwrite", "true")
//
//    val sc = spark.sparkContext
//
//    // Load tokens and create embeddings
//    logger.info("Loading data and creating embeddings.")
//    val tokensRDD = spark.sparkContext.textFile(inputPath)
//    val embeddingsRDD = tokensRDD.map { token =>
//      (token, EmbeddingUtils.generateRandomEmbedding(embeddingDim))
//    }
//
//    val embeddingValuesRDD = embeddingsRDD.map { case (_, embedding) => embedding }
//    val slidingWindowsRDD = SlidingWindowProcessor.createSlidingWindows(embeddingValuesRDD, windowSize = 4, embeddingDim)
//
//    // Save embeddings and sliding windows for inspection
//    logger.info("Saving embeddings and sliding windows for inspection.")
//    embeddingsRDD.saveAsTextFile(outputPath + "/embeddings")
//    slidingWindowsRDD.saveAsTextFile(outputPath + "/sliding_windows")
//
//    val dataSetRDD: RDD[DataSet] = slidingWindowsRDD
//      .filter { case (input, target) =>
//        input.shape()(0) == 4 && target.shape()(0) == 4
//      }
//      .map { case (input, target) =>
//        new DataSet(input.reshape(1, 4, embeddingDim), target.reshape(1, 4, embeddingDim))
//      }
//
//    val nonEmptyDataSetRDD = dataSetRDD.filter(_.numExamples() > 0)
//    val Array(trainingData, testData) = nonEmptyDataSetRDD.randomSplit(Array(1, 0.2), seed = 12345)
//
//    // Model Configuration
//    logger.info("Configuring the model.")
//    val modelConf = new NeuralNetConfiguration.Builder()
//      .weightInit(org.deeplearning4j.nn.weights.WeightInit.XAVIER)
//      .updater(new Adam(0.005))
//      .list()
//      .layer(0, new LSTM.Builder()
//        .nIn(embeddingDim)
//        .nOut(200)
//        .activation(Activation.TANH)
//        .build())
//      .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
//        .activation(Activation.SOFTMAX)
//        .nIn(200)
//        .nOut(embeddingDim)
//        .build())
//      .build()
//
//    val model = new MultiLayerNetwork(modelConf)
//    model.init()
//    model.setListeners(new ScoreIterationListener(10))
//
//    // Distributed Training Setup
//    logger.info("Setting up distributed training.")
//    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
//      .averagingFrequency(5)
//      .batchSizePerWorker(32)
//      .workerPrefetchNumBatches(2)
//      .build()
//
//    val sparkModel = new SparkDl4jMultiLayer(sc, model, trainingMaster)
//
//    // Training over multiple epochs
//    logger.info("Starting epoch-based training.")
//    val globalStartTime = System.currentTimeMillis()
//
//    for (epoch <- 1 to numEpochs) {
//      logger.info(s"Starting epoch $epoch")
//      val epochStartTime = System.currentTimeMillis()
//
//      // Train the model
//      try {
//        sparkModel.fit(trainingData)
//      } catch {
//        case e: NullPointerException =>
//          logger.log(Level.SEVERE, "Caught NullPointerException during training, likely due to empty partitions.", e)
//      }
//
//      val epochEndTime = System.currentTimeMillis()
//      logger.info(s"Completed epoch $epoch in ${(epochEndTime - epochStartTime) / 1000.0} seconds")
//
//      // Evaluate the model after each epoch
//      val evaluation = new Evaluation(embeddingDim)
//      val evaluationDataSetRDD = testData.collect()
//      if (evaluationDataSetRDD.nonEmpty) {
//        evaluationDataSetRDD.foreach { data =>
//          val output = model.output(data.getFeatures)
//          evaluation.eval(data.getLabels, output)
//        }
//        logger.info(s"Evaluation Metrics for Epoch $epoch:\n${evaluation.stats()}")
//      } else {
//        logger.warning(s"No data available for evaluation in Epoch $epoch.")
//      }
//    }
//
//    val globalEndTime = System.currentTimeMillis()
//    logger.info(s"Total training time: ${(globalEndTime - globalStartTime) / 1000.0} seconds")
//
//    // Save the trained model
//    ModelSerializer.writeModel(model, outputPath + "/LLM_Spark_Model.zip", true)
//    logger.info(s"Model saved at ${outputPath}/LLM_Spark_Model.zip")
//
//    // Print Spark Training Metrics
//    val trainingStats = sparkModel.getSparkTrainingStats
//    if (trainingStats != null) {
//      logger.info("Training statistics:\n" + trainingStats.statsAsString())
//    } else {
//      logger.warning("No training statistics available.")
//    }
//
//    // Stop Spark Context
//    logger.info("Shutting down Spark session.")
//    spark.stop()
//  }
//}

package com.thanu.llm
import java.util.logging.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.api.TrainingMaster
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.evaluation.classification.Evaluation
import org.deeplearning4j.util.ModelSerializer

object MainDriver {
  val logger: Logger = Logger.getLogger(this.getClass.getName)

  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      logger.severe("Usage: LLMEncoderDriver <input_path> <output_path> <embedding_dim>")
      System.exit(-1)
    }

    val inputPath = args(0)
    val outputPath = args(1)
    val embeddingDim = args(2).toInt
    val numEpochs = 5 // Number of epochs for training

    // Initialize Spark session
    val spark = SparkSession.builder
      .appName("LLM Encoder with Spark")
      .master("local[*]")
      .getOrCreate()
    spark.conf.set("spark.hadoop.mapreduce.output.fileoutputformat.outputdir.overwrite", "true")

    val sc = spark.sparkContext

    // Load tokens and create embeddings
    logger.info("Loading data and creating embeddings.")
    val tokensRDD = spark.sparkContext.textFile(inputPath)
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

    // Model Configuration
    logger.info("Configuring the model.")
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

    // Stop Spark Context
    logger.info("Shutting down Spark session.")
    spark.stop()
  }
}
