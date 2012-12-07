#include <stdio.h>
#include <stdlib.h>

#include "kaapic.h"

#define THRESHOLD 5

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

/* */
void solve(int n, int val, int col, int* hist, unsigned long* count)
{
  if (col == n) {
    *count += 1UL;
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
      kaapic_spawn(0,
                   5,
                   solve,
                   KAAPIC_MODE_V,  KAAPIC_TYPE_INT, 1, n,
                   KAAPIC_MODE_V,  KAAPIC_TYPE_INT, 1, i,
                   KAAPIC_MODE_V,  KAAPIC_TYPE_INT, 1, col + 1,
                   KAAPIC_MODE_S,  KAAPIC_TYPE_INT, n, hist,
                   KAAPIC_MODE_CW, KAAPIC_REDOP_PLUS, KAAPIC_TYPE_ULONG, 1, count
      );
    else {
      int c = col+1;
      solve(n, i, c, hist, count);
    }
  }
}

 
/* Main of the program
*/
int main(int argc, char** argv )
{
  unsigned long n;
	if (argc <= 1 || (n = atoi(argv[1])) <= 0) 
    n = 8;

  int count = 0;
	int* hist = (int*)alloca(sizeof(int)*n);

  kaapic_init(KAAPIC_START_ONLY_MAIN);

  kaapic_begin_parallel (KAAPIC_FLAG_DEFAULT);
  
  /*  */
	kaapic_spawn(0, 
               5,
               solve, 
               KAAPIC_MODE_V,  KAAPIC_TYPE_INT, 1, n, 
               KAAPIC_MODE_V,  KAAPIC_TYPE_INT, 1, 0, 
               KAAPIC_MODE_V,  KAAPIC_TYPE_INT, 1, 0, 
               KAAPIC_MODE_S,  KAAPIC_TYPE_INT, n, hist, 
               KAAPIC_MODE_CW, KAAPIC_REDOP_PLUS, KAAPIC_TYPE_ULONG, 1, &count
  );

  kaapic_sync();
  kaapic_end_parallel (KAAPIC_FLAG_DEFAULT);

  printf("Total number of solutions: %i\n", count);
  if (n <MAX_CHESSBOARD_SIZE)
  {
    kaapi_assert( count == solution[n] );
  }
  kaapic_finalize();
}

