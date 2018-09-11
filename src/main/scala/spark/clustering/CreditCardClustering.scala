package spark.clustering

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession

object CreditCardClustering extends App{

  // https://www.kaggle.com/arjunbhasin2013/ccdata

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  val df = spark.read.option("header","true").option("inferSchema","true").csv("./src/main/resources/CC GENERAL.csv")
  df.show(false)
  df.printSchema

  val completedf = df.na.drop

  val featureCols = df.columns.drop(1)

  val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val df2 = assembler.transform(completedf)

  val kmeans = new KMeans()
    .setK(4).setSeed(1L)
  val model = kmeans.fit(df2)

  val predictions = model.transform(df2)
  predictions.select("CUST_ID", "prediction").show(false)

  import org.apache.spark.sql.functions._

  predictions.groupBy("prediction").agg(count("*").alias("Num of Records")).show()

  val evaluator = new ClusteringEvaluator()

  val silhouette = evaluator.evaluate(predictions)
  println(s"Silhouette with squared euclidean distance = $silhouette")

  // silhouette
  // 클러스터 안의 거리가 짧을 수록 좋고(cohesion), 다른 클러스터와의 거리는 멀수록 좋다(separation)
  // 실루엣은 -1 부터 1사이의 값을 가진다.
  // 높을 수록 좋다.




}
