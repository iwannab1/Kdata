package spark.transformer

import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.sql.SparkSession

object onehot extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  val df = spark.createDataFrame(Seq(
    (0.0, 1.0),
    (1.0, 5.0),
    (2.0, 1.0),
    (0.0, 2.0),
    (0.0, 1.0),
    (2.0, 0.0)
  )).toDF("categoryIndex1", "categoryIndex2")

  val encoder = new OneHotEncoderEstimator()
    .setInputCols(Array("categoryIndex1", "categoryIndex2"))
    .setOutputCols(Array("categoryVec1", "categoryVec2"))
  val model = encoder.fit(df)

  val encoded = model.transform(df)
  encoded.show()


}
