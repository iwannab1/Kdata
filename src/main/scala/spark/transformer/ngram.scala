package spark.transformer
import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.SparkSession

object ngram extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  val wordDataFrame = spark.createDataFrame(Seq(
    (0, Array("Hi", "I", "heard", "about", "Spark")),
    (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
    (2, Array("Logistic", "regression", "models", "are", "neat"))
  )).toDF("id", "words")

  val ngram = new NGram().setN(3).setInputCol("words").setOutputCol("ngrams")

  val ngramDataFrame = ngram.transform(wordDataFrame)
  ngramDataFrame.select("words","ngrams").show(false)

}
