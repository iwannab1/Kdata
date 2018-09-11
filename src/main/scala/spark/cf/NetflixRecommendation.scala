package spark.cf

import java.io.File

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

object NetflixRecommendation extends App{

  // https://www.kaggle.com/c/expedia-hotel-recommendations/

  val location = "C:\\machinelearning\\IdeaProject\\Kdata\\src\\main\\resources\\mnist\\mnist.csv"
  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  import spark.implicits._

  def getListOfFiles(dir: String):List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  case class Rating(userId: Int, movieId: Int, rating: Float, date:String)
  def parseRating(str: String, movieId: String): Rating = {
    val fields = str.split(",")
    Rating(fields(0).toInt, movieId.toInt, fields(1).toFloat, fields(2))
  }

  def getDF(path: String) = {
    val rdd = spark.sparkContext.textFile(path)
    val idline = rdd.first()
    val data = rdd.filter(_ != idline)
    val id = idline.toString().replace(":", "")
    data.map(parseRating(_, id)).toDF
  }
  val files = getListOfFiles("C:\\machinelearning\\IdeaProject\\Kdata\\src\\main\\resources\\netflix\\training_set")

  var masterDF = getDF(files(0).getAbsolutePath)
  for(idx <- 1 to files.length-1){
    val df = getDF(files(idx).getAbsolutePath)
    masterDF = masterDF.union(df)
    println(masterDF.count())
  }

  val Array(training, test) = masterDF.randomSplit(Array(0.8, 0.2))

  val als = new ALS()
    .setMaxIter(5)
    .setRegParam(0.01)
    .setUserCol("userId")
    .setItemCol("movieId")
    .setRatingCol("rating")

  val model = als.fit(training)

  // Evaluate the model by computing the RMSE on the test data
  // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
  model.setColdStartStrategy("drop")
  val predictions = model.transform(test)

  val evaluator = new RegressionEvaluator()
    .setMetricName("rmse")
    .setLabelCol("rating")
    .setPredictionCol("prediction")
  val rmse = evaluator.evaluate(predictions)
  println(s"Root-mean-square error = $rmse")

  // Generate top 10 movie recommendations for each user
  val userRecs = model.recommendForAllUsers(10)
  // Generate top 10 user recommendations for each movie
  val movieRecs = model.recommendForAllItems(10)

  // Generate top 10 movie recommendations for a specified set of users
  val users = masterDF.select(als.getUserCol).distinct().limit(3)
  val userSubsetRecs = model.recommendForUserSubset(users, 10)
  // Generate top 10 user recommendations for a specified set of movies
  val movies = masterDF.select(als.getItemCol).distinct().limit(3)
  val movieSubSetRecs = model.recommendForItemSubset(movies, 10)

}
