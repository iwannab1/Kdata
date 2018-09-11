package spark.clustering

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.SparkSession

object CreditCardClustering2 extends App{

  // https://www.kaggle.com/arjunbhasin2013/ccdata

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

  val kmeans = new KMeans().setSeed(1234L)

  val pipeline = new Pipeline()
    .setStages(Array(assembler, kmeans))

  val paramGrid = new ParamGridBuilder()
    .addGrid(kmeans.k, 3 to 10)
    .build()

  val evaluator = new ClusteringEvaluator()
  val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setParallelism(2)

  val model = cv.fit(completedf)
  val predictions = model.transform(completedf)
  val silhouette = evaluator.evaluate(predictions)
  println(s"silhouette on our test set: ${silhouette}")

  val bestModel = model.bestModel
  val kmeanStage = bestModel.asInstanceOf[PipelineModel].stages(1)
  println(kmeanStage.asInstanceOf[KMeansModel].getK)





}
