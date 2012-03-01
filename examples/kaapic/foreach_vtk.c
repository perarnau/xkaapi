
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "kaapic.h"


typedef struct Point {
  double p[3];
} Point;

typedef struct Matrix {
  double m[3][3];
} Matrix;

static inline void Matvect_inplace( Point* p, const Matrix* m)
{
  p->p[0] = m->m[0][0] * p->p[0] + m->m[0][1]* p->p[1] + m->m[0][2]* p->p[2];
  p->p[1] = m->m[1][0] * p->p[0] + m->m[1][1]* p->p[1] + m->m[1][2]* p->p[2];
  p->p[2] = m->m[2][0] * p->p[0] + m->m[2][1]* p->p[1] + m->m[2][2]* p->p[2];
}

static void Matbody (
  int32_t i, int32_t j, int32_t tid, 
  Point* array, const Matrix* mat
)
{
  int k;
  for (k = i; k < j; ++k)
    Matvect_inplace(&array[k], mat);  
}


int main(int argc, char** argv)
{
  int32_t size = 100000 * 48;
  Point* array;
  static Matrix mat;
  int grain;
  int32_t i,j;

  kaapic_foreach_attr_t attr;
  
  double start, stop;
  
  kaapic_init(1);
  
  if (argc >1)
    grain = atoi(argv[1]);
  else
    grain = size / kaapi_getconcurrency();

  if (argc >2)
    size = atoi(argv[2]);

  array = (Point*)malloc(sizeof(Point)*size);

  for (i = 0; i < size; ++i) 
  {
    array[i].p[0] = drand48();
    array[i].p[1] = drand48();
    array[i].p[2] = drand48();
  }

  for (i = 0; i < 3; ++i) 
    for (j = 0; j < 3; ++j)
      mat.m[i][j] = drand48();

    
  kaapic_foreach_attr_init(&attr);
  kaapic_foreach_attr_set_grains(&attr, grain, grain);
  
  start = kaapic_get_time();
  
  for (i = 0; i < 10; ++i)
    kaapic_foreach(0, size, &attr, 2, Matbody, array, &mat);
      
  stop = kaapic_get_time();
  
  printf("Grain:%i\ttime: %lf\n", grain, (stop - start)/10);
  
  kaapic_finalize();

  return 0;
}
