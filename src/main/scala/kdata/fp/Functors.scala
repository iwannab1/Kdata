package kdata.fp

object Functors extends App{

  trait Functor[F[_]] {
    def map[A, B](fa: F[A])(f: A => B): F[B]
  }

  implicit val listFunctor: Functor[List] = new Functor[List] {
    def map[A, B](fa: List[A])(f: A => B): List[B] = fa.map(f)
  }

  def inc(list: List[Int])(implicit func: Functor[List]) = func.map(list)(_ + 1)

  println(inc(List(1, 2, 3)))

  trait Applicative[F[_]] extends Functor[F] {
    def pure[A](a: => A): F[A]
    def <*>[A,B](fa: => F[A])(f: => F[A => B]): F[B]
  }


}
