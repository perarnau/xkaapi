#include <iostream>
#include <stdlib.h>
#include "test_main.h"

#include "kaapi++"

/* */
struct TaskWrite : public ka::Task<3>::Signature<
  int, 
  int, 
  ka::W<int>
> {};

template<>
struct TaskBodyCPU<TaskWrite> {
  void operator()(int s, int n, ka::pointer_w<int> data)
  {
    if (s >0) sleep(s);
    *data = n;
  }
};

/* */
struct TaskReadWrite : public ka::Task<2>::Signature<
  int, 
  ka::RW<int>
> {};

template<>
struct TaskBodyCPU<TaskReadWrite> {
  void operator()(int n, ka::pointer_rw<int> data)
  {
    std::cout << "Read:" << *data << ", should be:" << n << std::endl;
    if (n != *data)
    {
      std::cout << "Fail" << std::endl;
    }
    kaapi_assert( n == *data );
  }
};

/* Main of the program
*/
void doit::operator()(int argc, char** argv )
{
	int data = 0;
	ka::Spawn<TaskWrite>()( 1, 1, &data );
	ka::Spawn<TaskReadWrite>()( 1, &data );
	ka::Spawn<TaskWrite>()( 0, 2, &data );
	ka::Spawn<TaskReadWrite>()( 2, &data );
  ka::Sync();
}

