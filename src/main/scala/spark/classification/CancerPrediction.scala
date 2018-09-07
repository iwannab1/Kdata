package spark.classification

import org.apache.spark.SparkFiles
import org.apache.spark.sql.SparkSession
import java.net.URL
import java.io.File

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions.not
import org.apache.spark.sql.types._

import sys.process._

object CancerPrediction extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  val dataurl = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
  val filename = "cancer.csv"
  val csvfile = new File(filename)
  if(!csvfile.exists)
  new URL(dataurl) #> csvfile !!

  val dfSchema = StructType(Array(
    StructField("scn", IntegerType, true),
    StructField("ct", IntegerType, true),
    StructField("ucs", IntegerType, true),
    StructField("ucs2", IntegerType, true),
    StructField("ma", IntegerType, true),
    StructField("secz", IntegerType, true),
    StructField("bn", IntegerType, true),
    StructField("bc", IntegerType, true),
    StructField("nn", IntegerType, true),
    StructField("mit", IntegerType, true),
    StructField("class", StringType, true)))

  val df = spark.read
              .schema(dfSchema)
              .option("header", "false")
              .csv(filename)

  println((df.count(), df.columns.length))
  val nadf = df.na.drop()
  println((nadf.count(), nadf.columns.length))

  nadf.describe().show(false)

  val featureCols = dfSchema.fields.map(_.name).drop(1).dropRight(1).toArray

  val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val df2 = assembler.transform(nadf)
  df2.show

  val labelIndexer = new StringIndexer().setInputCol("class").setOutputCol("label")
  val cancerdf = labelIndexer.fit(df2).transform(df2)
  cancerdf.show

  val Array(trainingData, testData) = cancerdf.randomSplit(Array(0.7, 0.3), seed=1234)


  val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
  val model = lr.fit(trainingData)

  println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

  val predictions = model.transform(testData)
  predictions.select("label", "prediction", "rawPrediction", "probability").show

  val lp = predictions.select( "label", "prediction")
  val total = predictions.count().toDouble

  val TP = lp.filter("label = 0 and prediction = 0").count
  val TN = lp.filter("label = 1 and prediction = 1").count
  val FP = lp.filter("label = 0 and prediction = 1").count
  val FN = lp.filter("label = 1 and prediction = 0").count

  val accuracy	= (TP + TN) / total
  val precision   = (TP + FP) / total
  val recall      = (TP + FN) / total
  val F1		= 2/(1/precision + 1/recall)

  println(s"accuracy: ${accuracy}")
  println(s"precision: ${precision}")
  println(s"recall: ${recall}")
  println(s"F1: ${F1}")


  val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
  val areaUnderROC = evaluator.evaluate(predictions)

  println(s"areaUnderROC: ${areaUnderROC}")

  val evaluator2 = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderPR")
  val areaUnderPR = evaluator2.evaluate(predictions)

  println(s"areaUnderPR: ${areaUnderPR}")

  val newData = testData.select("features")
  val predict = model.transform(newData)
  predict.select( "prediction", "rawPrediction", "probability").show





}
