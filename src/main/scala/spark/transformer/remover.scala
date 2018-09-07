package spark.transformer

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.SparkSession

object remover extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  val dataSet = spark.createDataFrame(Seq(
    (0, Seq("나는", "학교에", "간다")),
    (1, Seq("너는", "어디에", "가니?"))
  )).toDF("id", "raw")

  val remover = new StopWordsRemover()
    .setInputCol("raw")
    .setOutputCol("filtered")
    .setStopWords(Array("나는", "너는"))

  remover.transform(dataSet).show(false)




}