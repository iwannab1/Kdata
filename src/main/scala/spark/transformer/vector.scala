package spark.transformer

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object vector extends App{

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("Example")
    .getOrCreate()

  // SparseVector(size: Int, indices: Array[Int], values: Array[Double])
  val dv = Vectors.dense(1.0, 0.0, 3.0)
  println(dv.toSparse)
  val sv1 = Vectors.sparse(3, Array(0, 2), Array(1.0, 3.0))
  println(sv1.toDense)
  val sv2 = Vectors.sparse(3, Seq((0, 1.0), (1, 3.0)))
  println(sv2.toDense)

  val sv3 = Vectors.sparse(2, Array(), Array())
  println(sv3.toDense)

}
