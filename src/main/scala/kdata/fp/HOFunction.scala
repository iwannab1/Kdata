package kdata.fp

object HOFunction extends App{

//  def sumInts(a: Int, b: Int): Int =
//    if (a > b) 0 else a + sumInts(a + 1, b)
//
//  print(sumInts(3, 4))
//
    def cube(x: Int): Int = x * x * x
//
//  def sumCubes(a: Int, b: Int): Int =
//    if (a > b) 0 else cube(a) + sumCubes(a + 1, b)
//
//  print(sumCubes(3, 4))
//
  def factorial(x: Int): Int = x match {
    case 0 => 1
    case _ => factorial(x-1)
  }
//
//  def sumFactorials(a: Int, b: Int): Int =
//    if (a > b) 0 else factorial(a) + sumFactorials(a + 1, b)

//  print(sumFactorials(3, 4))

  def sum(f: Int => Int, a: Int, b: Int): Int =
    if (a > b) 0
    else f(a) + sum(f, a + 1, b)

  def id(x: Int): Int = x
//  def sumInts(a: Int, b: Int) = sum(id, a, b)
//  def sumCubes(a: Int, b: Int) = sum(cube, a, b)
//  def sumFactorials(a: Int, b: Int) = sum(factorial, a, b)

  def sumInts(a: Int, b: Int) = sum((x:Int) => x, a, b)
  def sumCubes(a: Int, b: Int) = sum(x => x*x*x, a, b)


  val isFactorOf  =  ( num :Int ) => {

    ( factor :Int ) => num % factor == 0
  }
  val isFactorOfHundred = isFactorOf( 100 )
  val isFactorOfNinetyNine = isFactorOf( 99 )
  val isFactorOfThousand = isFactorOf( 1000 )

  print(isFactorOfHundred(3))



}
