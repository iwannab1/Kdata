package spark.transformer

import org.apache.spark.ml.feature.{Binarizer, Bucketizer, PolynomialExpansion}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object bucket extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

//  val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)
//
//  val data = Array(-999.9, -0.5, -0.3, 0.0, 0.2, 999.9)
//  val dataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
//  dataFrame.show(truncate = false)
//
//  val bucketizer = new Bucketizer()
//    .setInputCol("features")
//    .setOutputCol("bucketedFeatures")
//    .setSplits(splits)
//
//  // Transform original data into its bucket index.
//  val bucketedData = bucketizer.transform(dataFrame)
//
//  println(s"Bucketizer output with ${bucketizer.getSplits.length-1} buckets")
//  bucketedData.show(truncate = false)
//
//  val splitsArray = Array(
//    Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity),
//    Array(Double.NegativeInfinity, -0.3, 0.0, 0.3, Double.PositiveInfinity))
//
//  val data2 = Array(
//    (-999.9, -999.9),
//    (-0.5, -0.2),
//    (-0.3, -0.1),
//    (0.0, 0.0),
//    (0.2, 0.4),
//    (999.9, 999.9))
//  val dataFrame2 = spark.createDataFrame(data2).toDF("features1", "features2")
//
//  val bucketizer2 = new Bucketizer()
//    .setInputCols(Array("features1", "features2"))
//    .setOutputCols(Array("bucketedFeatures1", "bucketedFeatures2"))
//    .setSplitsArray(splitsArray)
//
//  // Transform original data into its bucket index.
//  val bucketedData2 = bucketizer2.transform(dataFrame2)
//
//  println(s"Bucketizer output with [" +
//    s"${bucketizer2.getSplitsArray(0).length-1}, " +
//    s"${bucketizer2.getSplitsArray(1).length-1}] buckets for each input column")
//  bucketedData2.show()



//  val data = Array(
//    Vectors.dense(2.0, 1.0),
//    Vectors.dense(0.0, 0.0),
//    Vectors.dense(3.0, -1.0)
//  )
//  val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
//  df.show()
//
//  val polyExpansion = new PolynomialExpansion()
//    .setInputCol("features")
//    .setOutputCol("polyFeatures")
//    .setDegree(3)
//
//  val polyDF = polyExpansion.transform(df)
//  polyDF.show(false)

  import org.apache.spark.ml.feature.QuantileDiscretizer

  val data = Array((0, 18.0), (1, 19.0), (2, 8.0), (3, 5.0), (4, 2.2))
  val df = spark.createDataFrame(data).toDF("id", "hour")
  df.show()

  val discretizer = new QuantileDiscretizer()
    .setInputCol("hour")
    .setOutputCol("result")
    .setNumBuckets(4)

  val result = discretizer.fit(df).transform(df)
  result.show(false)


}
