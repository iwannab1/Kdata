package spark.regression

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

object BikeSharingPrediction extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  // http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
  val df = spark.read.option("header","true").option("inferSchema","true").csv("./src/main/resources/Bike-Sharing-Dataset/hour.csv")
  df.show(false)
  df.printSchema

  val df2 = df.drop("instant").drop("dteday").drop("casual").drop("registered")
  df2.describe().show(false)

  val featureCols = df2.columns.dropRight(1)

  val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("rawFeatures")
  val vectorindexer = new VectorIndexer().setInputCol("rawFeatures").setOutputCol("features").setMaxCategories(4)
  val gbt = new GBTRegressor().setLabelCol("cnt")

  val paramGrid = new ParamGridBuilder()
    .addGrid(gbt.maxIter, Array(2, 200))
    .addGrid(gbt.maxDepth, Array(10, 30))
    .build()

  val evaluator = new RegressionEvaluator()
    .setMetricName("rmse")
    .setLabelCol(gbt.getLabelCol)
    .setPredictionCol(gbt.getPredictionCol)

  val cv = new TrainValidationSplit()
    .setEstimator(gbt)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setTrainRatio(0.8)
    .setParallelism(2)

  val pipeline = new Pipeline()
    .setStages(Array(assembler, vectorindexer, cv))

  val Array(trainingData, testData) = df2.randomSplit(Array(0.7, 0.3), seed=1234)

  val model = pipeline.fit(trainingData)
  val predictions = model.transform(testData)
  predictions.select("cnt", "prediction").show(false)

  val rmse = evaluator.evaluate(predictions)
  println(s"RMSE on our test set: ${rmse}")




}
