package kdata.fp

object SomeOption extends App{

  def toInt(in: String): Option[Int] = {
    try {
      Some(Integer.parseInt(in.trim))
    } catch {
      case e: NumberFormatException => None
    }
  }

  val bag = List("1", "2", "foo", "3", "bar")

  val sum = bag.flatMap(toInt).sum

  println(sum)

}
