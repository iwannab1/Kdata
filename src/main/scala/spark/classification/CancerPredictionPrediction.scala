package spark.classification

import java.io.File
import java.net.URL

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import raonbit.spark.ml.DataSplitter

import scala.sys.process._

object CancerPredictionPrediction extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  val filename = "cancer.csv"

  val dfSchema = StructType(Array(
    StructField("scn", IntegerType, true),
    StructField("ct", IntegerType, true),
    StructField("ucs", IntegerType, true),
    StructField("ucs2", IntegerType, true),
    StructField("ma", IntegerType, true),
    StructField("secz", IntegerType, true),
    StructField("bn", IntegerType, true),
    StructField("bc", IntegerType, true),
    StructField("nn", IntegerType, true),
    StructField("mit", IntegerType, true),
    StructField("class", StringType, true)))

  val df = spark.read
              .schema(dfSchema)
              .option("header", "false")
              .csv(filename)

  val nadf = df.na.drop()
  val predictData = nadf.sample(false, 0.2).drop("class")

  val model = PipelineModel.load("./model/cancer-classification-model")

  val predictions = model.transform(predictData)
  predictions.select("prediction", "rawPrediction", "probability").show


}
