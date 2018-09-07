package spark.classification

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.sql.SparkSession

import scala.sys.process._

object CancerPredictionCV2 extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  val model = CrossValidatorModel.load("./model/cancer-classification-cvmodel")

  val bestModel = model.bestModel
  val lrStage = bestModel.asInstanceOf[PipelineModel].stages(2)
  println(lrStage.asInstanceOf[LogisticRegressionModel].coefficients)
  println(lrStage.asInstanceOf[LogisticRegressionModel].intercept)
  println(lrStage.asInstanceOf[LogisticRegressionModel].getMaxIter)
  println(lrStage.asInstanceOf[LogisticRegressionModel].getElasticNetParam)
  println(lrStage.asInstanceOf[LogisticRegressionModel].getRegParam)

}
