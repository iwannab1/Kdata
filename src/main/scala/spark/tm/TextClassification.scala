package spark.tm

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession

object TextClassification extends App{

  // https://www.kaggle.com/c/sf-crime/data

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR")

  val filename = "C:\\machinelearning\\IdeaProject\\Kdata\\src\\main\\resources\\crime\\train.csv"

  val df = spark.read
              .option("header", "true")
              .option("inferSchema", "true")
              .csv(filename)

  df.select("Descript", "Category").show(false)

  import org.apache.spark.sql.functions._
  df.groupBy("Category")
    .count()
    .orderBy(desc("count"))
    .show()

  // word split
  val regexTokenizer = new RegexTokenizer()
    .setInputCol("Descript")
    .setOutputCol("words")
    .setPattern("\\W")

  // stop word remove
  val add_stopwords = Array("http", "https", "amp","rt","t","c","the")
  val stopwordsRemover = new StopWordsRemover()
    .setInputCol("words")
    .setOutputCol("filtered")
    .setStopWords(add_stopwords)

  // vectorize
  val countVectors = new CountVectorizer()
    .setInputCol("filtered")
    .setOutputCol("features")
    .setVocabSize(10000)
    .setMinDF(5)

  val indexer = new StringIndexer()
    .setInputCol("Category")
    .setOutputCol("label")

  val pipeline = new Pipeline()
    .setStages(Array(regexTokenizer, stopwordsRemover, countVectors, indexer))

  val data = pipeline.fit(df).transform(df)
  data.show(5)

}
