#include <iostream>
#include <stdlib.h>
#include "test_main.h"

#include "kaapi++"

/* last entry i in solution[i] */
#define MAX_CHESSBOARD_SIZE  23 

long solution[] = {
0, /* chessboard size = 0 */
1,
0,
0,
2,
10,
4,
40,
92,
352,
724,
2680,
14200,
73712,
365596,
2279184,
14772512,
95815104,
666090624,
4968057848,
39029188884,
314666222712,
2691008701644,
24233937684440
};

#define THRESHOLD 5
/* */
struct TaskSolve : public ka::Task<5>::Signature<
  int, 
  int, 
  int, 
  ka::ST<ka::range1d<int> >,
  ka::CW<int>
> {};

template<>
struct TaskBodyCPU<TaskSolve> {
  void operator()(int n, int val, int col, ka::range1d_stack<int> hist, ka::pointer_cw<int> count)
  {
    if (col == n) {
      *count += 1;
      return;
    }

    if (col >0)
      hist[col-1] = val;    
   
#	define attack(i, j) (hist[j] == i || abs(hist[j] - i) == col - j)
    for (int i = 0, j = 0; i < n; i++) 
    {
      for (j = 0; j < col && !attack(i, j); j++);
      if (j < col) continue;
   
      if (col < THRESHOLD)
        ka::Spawn<TaskSolve>()(n, i, col + 1, hist, count);
      else
        (*this)(n, i, col + 1, hist, count);
    }
  }
};

 
/* Main of the program
*/
void doit::operator()(int argc, char** argv )
{
  int n;
	if (argc <= 1 || (n = atoi(argv[1])) <= 0) 
    n = 8;
  
  /* the stack pointer: */
  int count = 0;
	int* hist = (int*)alloca(sizeof(int)*n);
	ka::Spawn<TaskSolve>()(n, 0, 0, ka::range1d<int>(hist,n), &count);
  ka::Sync();

  std::cout << "Total number of solutions: " << count << std::endl;
  if (n <MAX_CHESSBOARD_SIZE)
  {
    kaapi_assert( count == solution[n] );
  }
}

