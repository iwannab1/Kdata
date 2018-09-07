package spark.classification

import java.io.File
import java.net.URL

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import raonbit.spark.ml.DataSplitter

import scala.sys.process._

object CancerPredictionCV extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

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

  val nadf = df.na.drop()
  val featureCols = dfSchema.fields.map(_.name).drop(1).dropRight(1).toArray

  val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val labelIndexer = new StringIndexer().setInputCol("class").setOutputCol("label")
  val lr = new LogisticRegression()

  val pipeline = new Pipeline()
    .setStages(Array(assembler, labelIndexer, lr))

  val paramGrid = new ParamGridBuilder()
    .addGrid(lr.maxIter, Array(5, 10, 20))
    .addGrid(lr.regParam, Array(0.1, 0.3, 0.5))
    .addGrid(lr.elasticNetParam, Array(0.7, 0.8, 0.9))
    .build()

  val cv = new CrossValidator()
    .setEstimator(pipeline)
    .setEvaluator(new BinaryClassificationEvaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(3)
    .setParallelism(2)

//  val cv = new TrainValidationSplit()
//    .setEstimator(pipeline)
//    .setEvaluator(new BinaryClassificationEvaluator)
//    .setEstimatorParamMaps(paramGrid)
//    // 80% of the data will be used for training and the remaining 20% for validation.
//    .setTrainRatio(0.8)
//    // Evaluate up to 2 parameter settings in parallel
//    .setParallelism(2)

  val Array(trainingData, testData) = new DataSplitter(nadf, Array(0.7, 0.3), "class").split

  val model = cv.fit(trainingData)

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

  model.write.overwrite().save("./model/cancer-classification-cvmodel")


}
