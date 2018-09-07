package spark.transformer

import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SparkSession

object string extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()


  val df = spark.createDataFrame(
    Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")) )
    .toDF("id", "category")

  val indexer = new StringIndexer()
    .setInputCol("category")
    .setOutputCol("categoryIndex")
    .setHandleInvalid("keep")
    .fit(df)

  val indexed = indexer.transform(df)
  indexed.show()

  val df2 = spark.createDataFrame(
    Seq((0, "a"), (1, "b"), (2, "e"), (3, "f")) )
    .toDF("id", "category")

  val df2indexed = indexer.transform(df2)
  df2indexed.show()

  val converter = new IndexToString()
    .setInputCol("categoryIndex")
    .setOutputCol("originalCategory")

  val converted = converter.transform(indexed)

  converted.show()


}
