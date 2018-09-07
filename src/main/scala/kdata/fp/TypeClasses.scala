package kdata.fp

object TypeClasses extends App{

  trait NumberLike[T] {
    def plus(x: T, y: T): T
    def divide(x: T, y: Int): T
    def minus(x: T, y: T): T
  }

  implicit object NumberLikeDouble extends NumberLike[Double] {
    def plus(x: Double, y: Double): Double = x + y
    def divide(x: Double, y: Int): Double = x / y
    def minus(x: Double, y: Double): Double = x - y
  }

  implicit object NumberLikeInt extends NumberLike[Int] {
    def plus(x: Int, y: Int): Int = x + y
    def divide(x: Int, y: Int): Int = x / y
    def minus(x: Int, y: Int): Int = x - y
  }

  def mean[T](numbers: Seq[T])(implicit number: NumberLike[T]): T = {
    number.divide(numbers.reduce(number.plus), numbers.size)
  }

  def mean2[T: NumberLike](numbers: Seq[T]) = {
    val number = implicitly[NumberLike[T]]
    number.divide(numbers.reduce(number.plus), numbers.size)
  }

  println(mean(List[Int](1, 2, 3, 6, 8)))
  println(mean(Seq(1, 2, 3, 6, 8)))

  println(mean2(List[Int](1, 2, 3, 6, 8)))
  println(mean2(Seq(1, 2, 3, 6, 8)))


}
