package raonbit.spark.ml

import java.io.File
import java.net.URL

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._

import scala.collection.mutable.ListBuffer

class DataSplitter(df: DataFrame, ratio: Array[Double], target: String, seed : Int = 1234) {

  val categories = df.select(target).distinct.collect.flatMap(_.toSeq)
  val df2 = df.withColumn(target, df.col(target).cast(StringType))
  val cnt = ratio.length

  def split() = {
    val groups = categories.map(cat => df2.filter(target + " == " + cat))
    var spdata = groups(0).randomSplit(ratio, seed=seed)

    for(i <- 1 to categories.length-1){
      val temp = groups(i).randomSplit(ratio, seed=seed)
      for(j <- 0 to cnt-1){
        spdata(j) = spdata(j).union(temp(j))
      }
    }
    spdata
  }

}
