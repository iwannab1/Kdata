package spark.transformer

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler}
import org.apache.spark.ml.linalg.Vectors

object Normalizer extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  // For implicit conversions like converting RDDs to DataFrames
  import spark.implicits._

  val dataFrame = spark.createDataFrame(Seq(
    (0, Vectors.dense(1.0, 0.5, -1.0)),
    (1, Vectors.dense(2.0, 1.0, 1.0)),
    (2, Vectors.dense(4.0, 10.0, 2.0))
  )).toDF("id", "features")

  dataFrame.show()
//
//  // Normalize each Vector using $L^1$ norm.
//  val normalizer = new Normalizer()
//    .setInputCol("features")
//    .setOutputCol("normFeatures")
//    .setP(1.0)
//
//  val l1NormData = normalizer.transform(dataFrame)
//  println("Normalized using L^1 norm")
//  l1NormData.show()
//
//  // Normalize each Vector using $L^\infty$ norm.
//  val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
//  println("Normalized using L^inf norm")
//  lInfNormData.show()

//  val scaler = new StandardScaler()
//    .setInputCol("features")
//    .setOutputCol("scaledFeatures")
//
//  // Compute summary statistics by fitting the StandardScaler.
//  val scalerModel = scaler.fit(dataFrame)
//
//  // Normalize each feature to have unit standard deviation.
//  val scaledData = scalerModel.transform(dataFrame)
//  scaledData.show()

//  val scaler = new MinMaxScaler()
//    .setInputCol("features")
//    .setOutputCol("scaledFeatures")
//
//  // Compute summary statistics and generate MinMaxScalerModel
//  val scalerModel = scaler.fit(dataFrame)
//
//  // rescale each feature to range [min, max].
//  val scaledData = scalerModel.transform(dataFrame)
//  scaledData.select("features", "scaledFeatures").show(false)

  val scaler = new MaxAbsScaler()
    .setInputCol("features")
    .setOutputCol("scaledFeatures")

  // Compute summary statistics and generate MinMaxScalerModel
  val scalerModel = scaler.fit(dataFrame)

  // rescale each feature to range [min, max].
  val scaledData = scalerModel.transform(dataFrame)
  scaledData.select("features", "scaledFeatures").show(false)

}
