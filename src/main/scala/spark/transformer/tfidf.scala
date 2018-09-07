package spark.transformer

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

object tfidf extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()


  val sentenceData = spark.createDataFrame(Seq(
    (0.0, "Hi I heard about Spark"),
    (0.0, "I wish Java could use case classes"),
    (1.0, "Logistic regression models are neat")
  )).toDF("label", "sentence")

  val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
  val wordsData = tokenizer.transform(sentenceData)

  val hashingTF = new HashingTF()
    .setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

  val featurizedData = hashingTF.transform(wordsData)
  featurizedData.show(false)
  // alternatively, CountVectorizer can also be used to get term frequency vectors

  val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
  val idfModel = idf.fit(featurizedData)

  val rescaledData = idfModel.transform(featurizedData)
  rescaledData.select("label", "features").show(false)

}
