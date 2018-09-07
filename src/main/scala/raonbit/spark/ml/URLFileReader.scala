package raonbit.spark.ml

import java.io.File
import java.net.URL

import sys.process._
import org.apache.spark.sql.{DataFrame, SparkSession}

class URLFileReader(spark: SparkSession, url: String) {

  val tmp = System.getProperty("java.io.tmpdir")
  val downloadfile = new File(tmp, getFileName())

  def getFileName(): String = {
    val splitURL = url.split("/")
    splitURL(splitURL.length-1)
  }

  def downLoad(overwrite:Boolean = true): Unit ={
    if(overwrite)
      new URL(url) #> downloadfile !!
    else if(!downloadfile.exists)
      new URL(url) #> downloadfile !!
  }

  def readFile(header:String="true", sep:String=",", quote:String="\"", inferSchema:String="true"): DataFrame = {
    spark.read.option("header", header).option("delimiter", sep).option("quote", quote).option("inferSchema", inferSchema)
      .csv(downloadfile.getAbsolutePath)
  }

}
