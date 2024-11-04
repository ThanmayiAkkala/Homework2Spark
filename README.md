# CS441 Homework2

**Name**: Thanmayi Akkala  
**Email**: takkal2@uic.edu

**UIN**: 650556907

**Video Link**: 


## Overview

This project focuses on implementing a distributed pipeline to train a Large Language Model (LLM) encoder using Apache Spark and DeepLearning4J (DL4J), optimized for handling sequential data in a large-scale distributed environment. It begins by generating or loading embeddings for each token in the dataset, which are then processed into input sequences using a sliding window approach. This transformation enables the model to learn contextual patterns from overlapping windows, ideal for sequence-based neural networks like LSTMs. The pipeline is configured with an LSTM-based model, which is trained in parallel across Spark’s distributed architecture to maximize scalability and efficiency. Following each epoch, the model undergoes evaluation on test data to track performance metrics such as accuracy and loss, alongside key statistics like gradient norms, learning rates, and training durations, providing a comprehensive view of the training process. Finally, the trained model is saved for future applications, and Spark’s UI enables monitoring of job execution, memory usage, and shuffling statistics to optimize resource utilization and identify performance improvements. 

### Key Files:
- **EmbeddingUtils.scala**: This utility file includes essential functions for working with embeddings.
- **SlidingWindowProcessor.scala**:Handles the creation of sliding windows to transform token embeddings into a suitable format for training
- **ModelConfig.scala**: The Model is configered in this file.
- **MainDriver.scala**: The main driver file that orchestrates the Spark job, model training, and evaluation steps:
  - Epoch-based training loop for distributed training on Spark.
  - Model evaluation using test datasets, and saves metrics like training statistics, gradient norms, and evaluation results.

### Build Instructions
This project uses **SBT** (Scala Build Tool) to manage dependencies and compile the project.

**build.sbt** includes the following dependencies:

- `org.apache.spark`: Provides the Spark libraries used to set up and manage the distributed training infrastructure.
- `org.apache.hadoop`: For distributed processing and handling large datasets within the Spark job.
- `org.deeplearning4j`: For the LSTM model and neural network training, including functionalities for model configuration, training, and evaluation.
- `org.nd4j`: For handling and manipulating ND4J tensors, essential for creating embeddings and working with multidimensional data arrays.
- Logging libraries such as `logback` and `slf4j`: Used for detailed logging and debugging of the training, evaluation, and job execution processes.


## Running the Project

### Prerequisites:
- **Java Version**: Ensure that **Java 11** is installed and set as your default JDK.
- **Scala Version**: The project is built with **Scala 2.12.5**.
- **Spark Version**: Pre-built for Apache Hadoop 3.3 and later.
- **SBT Version**: Use **SBT 1.x** to build and package the project.
- **Deeplearning4j**: The project uses **Deeplearning4j 1.0.0-M2.1** for Word2Vec embeddings.

### How to Run Locally in IntelliJ:

1. **Clone the Project**: 
   - Open IntelliJ IDEA and clone the project into your workspace. Or create a new project in intellij and add the key files provided above under src/main/scala and the Build.sbt and Plugin.sbt under the project folder.

2. **Build the Project**:
   - Make sure that `build.sbt` is properly configured with all dependencies.
   - Run `sbt clean assembly` to compile and build the project into a JAR file.

3. **Running the Code in IntelliJ**:
   - Open the `MainDriver.scala` file.
   - Provide the required input arguments to the `main` method. Example:
     ```
     sbt run com.thanu.llm.MainDriver <input file path> <output directory> <embedding dimension>
     ```
   - Replace `<input file path>` with the path to your input data file, `<output directory>` with the directory where you want to save results, and `<embedding dimension>` with the desired dimension for the embeddings.

   - **Note:** For the input file, an open-source embedding file is used along with the previouos output files to train the model.

     Example:

![image](https://github.com/user-attachments/assets/2f0fca4a-3a19-4c59-987d-66d66d28dca2)

   - After running, to check if it has successfully implemented, please check the output directory for the files even if warnings are shown.
   - The output file typically include embeddings folder(for testing) and a sliding window folder that contains features and target embeddings and the finally when the model is trained it is saved as a zip file.

4. **Running with Spark Locally**:
   - Ensure that Spark is configured and running.
   - Also ensure you loaded the plugins.sbt for the assembly jar to work.
   - Run the following command:
     ```
      spark-submit \
       --class com.thanu.llm.MainDriver \
       --master <spark-master-url> \
       --deploy-mode <deploy-mode> \
       --num-executors <num-executors> \
       --executor-memory <executor-memory> \
       --executor-cores <executor-cores> \
       /path/to/project.jar <input file path> <output directory> <embedding dimension>
     ```
   - The `<input-path>` is the path to your input embedding file, and the `<output directory>` directories where output files will be written.

5. **Running on Amazon EMR**:
   - Upload the compiled JAR file and dataset files to **S3**.
   - Create an EMR cluster with **Spark** and **Scala** pre-installed.
   - Add a step to the EMR cluster and add the jar file from s3 and pass the spark submit options and the arguments respectively
   - Monitor the cluster for job completion and download the results from S3.
### Scala Unit/Integration Tests:
The tests are under in src/tests/scala. These can be run using sbt test at once or sbt.
It can be run using the scala test or by passing the files individually like: sbt "testOnly *SlidingWindowTest"
More detailed in this docs: https://docs.google.com/document/d/1CsSLDK4hZqzr5Y7--g8d4cAiiCtesisuCnXA9J8Bxn8/edit?usp=sharing
### Output Explanation :
The folders and the model is saved in this way in the output dierectory mentioned:

![image](https://github.com/user-attachments/assets/addf1332-8c41-4f35-b70e-ade8fa79cc04)

The embeddings and the sliding window folder contain embedding and feature and targets repectively

![image](https://github.com/user-attachments/assets/d9bed975-12fe-4529-b7a4-aff6d94c4b37)

Evaluation metrics calculated:

![image](https://github.com/user-attachments/assets/4f3fc0be-a8f1-4e30-ac1a-d10b064f10d3)
![image](https://github.com/user-attachments/assets/de995337-712c-424b-bbe4-af8e70ad0156)

![image](https://github.com/user-attachments/assets/93472386-4144-4d66-a30a-76536e4b48d0)


![image](https://github.com/user-attachments/assets/a19221cd-608f-462e-9d82-3944b2f1a2d4)

![image](https://github.com/user-attachments/assets/1659ee78-26bb-44ed-8f92-164f8fb1094f)


After deploying on emr when the status is complete:

![image](https://github.com/user-attachments/assets/3f01cf0d-7fac-474d-a262-bc53f3c46526)

the output folders that are passed as arguments for output_1 and output_2 are created and the corresponding output files:

![image](https://github.com/user-attachments/assets/ff30a583-39b1-49b3-85d7-6b1541d8078d)

![image](https://github.com/user-attachments/assets/a9882073-f6c0-4158-ae20-2547dda6a0da)





