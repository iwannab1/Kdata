package spark.clustering

import org.apache.spark.ml.clustering.{BisectingKMeans, BisectingKMeansModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object CreditCardClustering3 extends App{

  // Bisecting k-means : hierarchical clustering using a divisive (or “top-down”)

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  val df = spark.read.option("header","true").option("inferSchema","true").csv("./src/main/resources/CC GENERAL.csv")
  val completedf = df.na.drop

  val featureCols = df.columns.drop(1)

  val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val df2 = assembler.transform(completedf)

  val bkm = new BisectingKMeans().setK(2).setSeed(1)
  val model = bkm.fit(df2)

  // Evaluate clustering.
  val cost = model.computeCost(df2)
  println(s"Within Set Sum of Squared Errors = $cost")

}
