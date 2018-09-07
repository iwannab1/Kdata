package spark.transformer

import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.sql.SparkSession

object vindexer extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  val data = spark.read.format("libsvm").load("C:\\spark-2.1.0-bin-hadoop2.7\\data\\mllib\\sample_libsvm_data.txt")
  data.show(false)

  val indexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexed")
    .setMaxCategories(10)

  val indexerModel = indexer.fit(data)

  val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
  println(s"Chose ${categoricalFeatures.size} " +
    s"categorical features: ${categoricalFeatures.mkString(", ")}")

  // Create new column "indexed" with categorical values transformed to indices
  val indexedData = indexerModel.transform(data)
  indexedData.show()

}
