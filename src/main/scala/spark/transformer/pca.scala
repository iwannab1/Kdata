package spark.transformer

import org.apache.spark.ml.feature.{PCA, VectorIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object pca extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  val data = Array(
    Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
    Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
    Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0) )

  val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
  df.show(false)

  val pca = new PCA()
    .setInputCol("features")
    .setOutputCol("pcaFeatures")
    .setK(3)
    .fit(df)

  val result = pca.transform(df)
  result.show(false)

  val vectorindexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexed")
    .setMaxCategories(3)

  val indexerModel = vectorindexer.fit(df)
  val indexedData = indexerModel.transform(df)
  indexedData.show(false)



}
