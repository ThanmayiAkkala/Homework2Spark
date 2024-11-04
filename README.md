# CS441 Homework1

**Name**: Thanmayi Akkala  
**Email**: takkal2@uic.edu

**UIN**: 650556907

**Video Link**: https://youtu.be/wxtleucxmeU
(Would suggest to watch in 1.5x or even 2x :-) )

## Overview
The LLM Encoder Project is designed to process large-scale text corpora using a parallel distributed system architecture. This project utilizes Hadoop's MapReduce framework to handle data tokenization, word frequency calculation, and Word2Vec-based token embedding generation. The primary goal of the system is to generate token embeddings that capture the semantic meaning of words in the corpus and identify tokens that are semantically similar using cosine similarity.

The system processes text data efficiently by leveraging the MapReduce paradigm for distributed data handling. The project is designed to scale and handle massive datasets, making it particularly suited for cloud environments like Amazon EMR. By using a combination of tokenization and deep learning-based Word2Vec embeddings, this system provides meaningful word embeddings and relationships between tokens, which are valuable for various natural language processing tasks.


This project is implemented using **Java 11** to ensure compatibility with modern cloud environments and uses key technologies such as Hadoop 3.3.6 and Deeplearning4j for Word2Vec embeddings.


### Key Files:
- **TokenisationMapper.scala**: Tokenizes input text into individual tokens and emits.
- **TokenisationReducer.scala**: Aggregates the count of each token across the dataset and is responisible for the next step of generating vector embeddings.
- **Word2VecMapper.scala**: Trains the Word2Vec model on the input tokens and outputs the token embeddings.
- **Word2VecReducer.scala**: Handles the embeddings and processes any further aggregation or reduction tasks.
- **LLMEncoderDriver.scala**: The driver class orchestrating the MapReduce job flow.

### Build Instructions
This project uses **SBT** (Scala Build Tool) to manage dependencies and compile the project.

**build.sbt** includes the following dependencies:
- `org.apache.hadoop`: For MapReduce and Hadoop Common libraries.
- `org.deeplearning4j`: For the Word2Vec implementation.
- `jtokkit`: For tokenization.
- Logging libraries like `logback` and `slf4j`.

## Running the Project

### Prerequisites:
- **Java Version**: Ensure that **Java 11** is installed and set as your default JDK.
- **Scala Version**: The project is built with **Scala 2.13.8**.
- **Hadoop Version**: This project is tested on **Hadoop 3.3.6**.
- **SBT Version**: Use **SBT 1.x** to build and package the project.
- **Deeplearning4j**: The project uses **Deeplearning4j 1.0.0-M2.1** for Word2Vec embeddings.

### How to Run Locally in IntelliJ:

1. **Clone the Project**: 
   - Open IntelliJ IDEA and clone the project into your workspace. Or create a new project in intellij and add the key files provided above under src/main/scala and the Build.sbt and Plugin.sbt under the project folder.

2. **Build the Project**:
   - Make sure that `build.sbt` is properly configured with all dependencies.
   - Run `sbt clean assembly` to compile and build the project into a JAR file.

3. **Running the Code in IntelliJ**:
   - Open the `LLMEncoderDriver.scala` file.
   - Provide the required input arguments to the `main` method. Example:
     ```
     sbt run com.thanu.llm.LLMEncoderDriver <input file path> <output directory> <output_directory_2>
     ```
     Example:

![image](https://github.com/user-attachments/assets/d0376e38-8855-40b1-816e-fb8384d61e51)

   - Ensure the dataset is in the appropriate directory (e.g., Project Gutenberg texts).
   - Select `Run` or `Debug` from IntelliJ's menu to start the process.
   - After running, to check if it has successfully implemented, please check the output directory for the files even if warnings are shown.

4. **Running with Hadoop Locally**:
   - Ensure that Hadoop is configured and running.
   - Also ensure you loaded the plugins.sbt for the assembly jar to work.
   - Run the following command:
     ```
     hadoop jar target/scala-2.13/CloudLLMProject-assembly.jar com.thanu.llm.LLMEncoderDriver <input-path> <output-path-1> <output-path-2>
     ```
   - The `<input-path>` is the path to your input text file, and the `<output-path-1>` and `<output-path-2>` are the directories where output files will be written.

5. **Running on Amazon EMR**:
   - Upload the compiled JAR file and dataset files to **S3**.
   - Create an EMR cluster with **Hadoop** and **Scala** pre-installed.
   - Add a step to the EMR cluster using the following command format:
     ```
     hadoop jar s3://<your-bucket-name>/CloudLLMProject-assembly.jar com.thanu.llm.LLMEncoderDriver s3://<your-bucket-name>/<input-path> s3://<your-bucket-name>/<output-path-1> s3://<your-bucket-name>/<output-path-2>
     ```
   - Monitor the cluster for job completion and download the results from S3.
### Scala Unit/Integration Tests:
The tests are under in src/tests/scala. These can be run using sbt test at once or sbt.
It can be run using the scala test or by passing the files individually like: sbt "testOnly *Word2VecMapperTest"
More detailed in this docs: https://docs.google.com/document/d/1CsSLDK4hZqzr5Y7--g8d4cAiiCtesisuCnXA9J8Bxn8/edit?usp=sharing
### Output Explanation:
The first mapper reducer gives the tokens and the number of occurences.
![image](https://github.com/user-attachments/assets/77be1062-127d-4b9a-83df-e7dc667a091d)
![image](https://github.com/user-attachments/assets/dc9753e5-f7a5-45e8-86a9-e454904f6825)

The wordsvec Mapper and reducer gives the token and its corresponding embeddings and the similar tokens down in the output file

![image](https://github.com/user-attachments/assets/5a2ebc7b-40ec-49c8-979a-b0f9c3520dd5)
![image](https://github.com/user-attachments/assets/aba4da8e-37e6-4162-a3c0-3dce78397cbb)

After deploying on emr when the status is complete:

![image](https://github.com/user-attachments/assets/3f01cf0d-7fac-474d-a262-bc53f3c46526)

the output folders that are passed as arguments for output_1 and output_2 are created and the corresponding output files:

![image](https://github.com/user-attachments/assets/ff30a583-39b1-49b3-85d7-6b1541d8078d)

![image](https://github.com/user-attachments/assets/a9882073-f6c0-4158-ae20-2547dda6a0da)





