package kdata.fp

object Functions extends App{

  // partially applied function
  def line(a: Int, b: Int, x: Int): Int = a * x + b

  def partialLine_v1 = (b: Int, x: Int) => line(2, b, x)
  println(partialLine_v1(0, 1))

  def partialLine_v2 = line(2, _: Int, _: Int)
  println(partialLine_v2(0, 1))


  //curried function
  def line2(a: Int, b: Int, x: Int): Int = a * x + b

  def curriedLine(a: Int)(b: Int)(x: Int): Int = line2(a, b, x)
  println(curriedLine(2)(0)(1))

  def defaultLine(x: Int): Int = curriedLine(2)(0)(x)
  println(defaultLine(1))

  def curriedLine2 = (line2 _).curried
  println(curriedLine2(2)(0)(1))

//  Seq(1, 2, 3).foldLeft(Seq[Int]()) { (is, i) =>
//    is :+ i * i
//  }
//
//  Seq(1, 2, 3).foldLeft(Seq[Int](), { (is: List[Int], i: Int) =>
//    is :+ i * i
//  })

  var more = 10
  var increase = (x: Int) => x + more

  println(increase(10))

  more = 100
  println(increase(10))





}
