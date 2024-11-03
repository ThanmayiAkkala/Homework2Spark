ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.5"

libraryDependencies ++= Seq(
  // Hadoop dependencies (only necessary if interacting with HDFS)
  "org.apache.hadoop" % "hadoop-common" % "3.3.6",
  "org.apache.hadoop" % "hadoop-client" % "3.3.6",

  // Spark Core and SQL for distributed processing
  "org.apache.spark" %% "spark-core" % "3.3.1",
  "org.apache.spark" %% "spark-sql" % "3.3.1",

  // Deeplearning4j and Nd4j for machine learning and vector operations
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1",
  "org.deeplearning4j" % "dl4j-spark_2.12" % "1.0.0-M2.1",
  // Tokenizer dependency
  "com.knuddels" % "jtokkit" % "1.1.0",

  // Logging dependencies
  "ch.qos.logback" % "logback-classic" % "1.2.3",
  "org.slf4j" % "slf4j-api" % "1.7.30",

  // Configuration library
  "com.typesafe" % "config" % "1.4.3",

  // Testing libraries
  "org.scalatest" %% "scalatest" % "3.2.15" % Test,
  "org.scalamock" %% "scalamock" % "5.2.0" % Test,
  "org.mockito" %% "mockito-scala" % "1.17.7" % Test,
  "org.scalatestplus" %% "mockito-3-4" % "3.2.9.0" % Test
)

lazy val root = (project in file("."))
  .settings(
    name := "LLM-Spark-DL4J",

    // Set the name of the assembled JAR
    assembly / assemblyJarName := "LLM-Spark-DL4J-assembly.jar",

    // Handle merge conflicts when building an assembly JAR
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", xs@_*) =>
        xs match {
          case "MANIFEST.MF" :: Nil => MergeStrategy.discard
          case "services" :: _ => MergeStrategy.concat
          case _ => MergeStrategy.discard
        }
      case "reference.conf" => MergeStrategy.concat  // Merge configuration files
      case x if x.endsWith(".proto") => MergeStrategy.rename  // Handle protobuf conflicts
      case x if x.contains("hadoop") => MergeStrategy.first  // Use first Hadoop version
      case PathList("org", "slf4j", _*) => MergeStrategy.first  // Handle SLF4J conflicts
      case PathList("org", "nd4j", _*) => MergeStrategy.first  // Handle ND4J conflicts
      case PathList("org", "deeplearning4j", _*) => MergeStrategy.first  // Handle DL4J conflicts
      case _ => MergeStrategy.first  // Default strategy: take the first found
    }
  )
