package spark.transformer

import org.apache.spark.ml.feature.Imputer
import org.apache.spark.sql.SparkSession

object imputer extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  val df = spark.createDataFrame(Seq(
    (1.0, Double.NaN),
    (2.0, Double.NaN),
    (Double.NaN, 3.0),
    (4.0, 4.0),
    (5.0, 5.0)
  )).toDF("a", "b")

//  val imputer = new Imputer()
//    .setInputCols(Array("a", "b"))
//    .setOutputCols(Array("out_a", "out_b"))
//

//  val imputer = new Imputer()
//    .setInputCols(Array("a", "b"))
//    .setOutputCols(Array("out_a", "out_b"))
//    .setStrategy("median") // mean, median

    val imputer = new Imputer()
      .setInputCols(Array("a", "b"))
      .setOutputCols(Array("out_a", "out_b"))
      .setMissingValue(1.0) // missing value : NaN -> 1.0

  val model = imputer.fit(df)
  model.transform(df).show()

}
