package spark.transformer

import java.util.Arrays

import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.StructType


object sclicer extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()


  val data = Arrays.asList(
    Row(Vectors.sparse(3, Seq((0, -2.0), (1, 2.3)))),
    Row(Vectors.dense(-2.0, 2.3, 0.0))
  )

  val defaultAttr = NumericAttribute.defaultAttr
  val attrs = Array("f1", "f2", "f3").map(defaultAttr.withName)
  val attrGroup = new AttributeGroup("userFeatures", attrs.asInstanceOf[Array[Attribute]])

  val dataset = spark.createDataFrame(data, StructType(Array(attrGroup.toStructField())))

  dataset.show(false)

  val slicer = new VectorSlicer().setInputCol("userFeatures").setOutputCol("features")

  // slicer.setIndices(Array(1)).setNames(Array("f3"))
  slicer.setIndices(Array(0, 2)) //setNames(Array("f2", "f3"))

  val output = slicer.transform(dataset)
  output.show(false)

}
