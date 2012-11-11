#include <stdio.h>
#include <stdlib.h>

#include "kaapic.h"

#define THRESHOLD 5

/* */
void solve(int n, int val, int col, int* hist, int* count)
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
      kaapic_spawn(0,
                   5,
                   solve,
                   KAAPIC_MODE_V,  KAAPIC_TYPE_INT, 1, n, 
                   KAAPIC_MODE_V,  KAAPIC_TYPE_INT, 1, i, 
                   KAAPIC_MODE_V,  KAAPIC_TYPE_INT, 1, col + 1, 
                   KAAPIC_MODE_S,  KAAPIC_TYPE_INT, n, hist, 
                   KAAPIC_MODE_CW, KAAPIC_REDOP_PLUS, KAAPIC_TYPE_INT, 1, count
      );
    else
      solve(n, i, col + 1, hist, count);
  }
}

 
/* Main of the program
*/
int main(int argc, char** argv )
{
  int n;
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
               KAAPIC_MODE_CW, KAAPIC_REDOP_PLUS, KAAPIC_TYPE_INT, 1, &count
  );

  kaapic_sync();
  kaapic_end_parallel (KAAPIC_FLAG_DEFAULT);

  printf("Total number of solutions: %i\n", count);
  kaapic_finalize();
}

