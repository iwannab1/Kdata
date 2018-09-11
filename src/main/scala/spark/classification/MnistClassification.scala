package spark.classification

import java.io.File

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import raonbit.spark.ml.MnistDataset
import spark.regression.BikeSharingPrediction.rmse

object MnistClassification extends App{

  // http://yann.lecun.com/exdb/mnist/

  val location = "C:\\machinelearning\\IdeaProject\\Kdata\\src\\main\\resources\\mnist\\mnist.csv"
  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  val df = spark.read.option("header","true").option("inferSchema","true").csv(location)
  val featureCols = df.columns.drop(1)

  val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
  val layers = Array(28*28, 14*14, 5*5, 10)
  val mlp = new MultilayerPerceptronClassifier()
    .setLayers(layers)

  val paramGrid = new ParamGridBuilder()
    .addGrid(mlp.layers, Array(Array(28*28, 14*14, 5*5, 10), Array(28*28, 1024, 10)))
    .addGrid(mlp.stepSize, Array(0.01, 0.05, 0.1, 0.5))
    .build()

  val evaluator = new MulticlassClassificationEvaluator()
    .setMetricName("accuracy")

  val cv = new CrossValidator()
    .setEstimator(mlp)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(3)
    .setParallelism(2)

  val pipeline = new Pipeline()
    .setStages(Array(assembler, cv))

  val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed=1234)

  val model = pipeline.fit(trainingData)
  val predictions = model.transform(testData)
  val accuracy = evaluator.evaluate(predictions)
  println(s"accuracy on our test set: ${accuracy}")


}
