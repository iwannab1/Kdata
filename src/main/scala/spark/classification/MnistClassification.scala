package spark.classification

import java.io.File

import org.apache.spark.sql.SparkSession
import raonbit.spark.ml.MnistDataset

object MnistClassification extends App{

  // http://yann.lecun.com/exdb/mnist/

  val location = "C:\\machinelearning\\IdeaProject\\Kdata\\src\\main\\resources\\mnist"
  val locationFile = new File(location)

  if (!locationFile.exists)
    locationFile.mkdirs

  val trainDataset = new MnistDataset(location, "train")
  val testDataset = new MnistDataset(location, "t10k")

  val trainExamples = trainDataset.examples.take(10000).toList
  val testExamples = trainDataset.examples.drop(trainExamples.size).take(1000).toList

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  val train = spark.createDataFrame(trainExamples)
  val test = spark.sparkContext.parallelize(testExamples)



}
