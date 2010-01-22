#define RANDOM_INIT
//#define __DEBUG

#include <stdlib.h>
#include <iostream>
#include <sys/types.h>
#include <unistd.h>

typedef double val_t;



#ifndef RANDOM_INIT
/*********************************************************************/
val_t* initialisation (int size, int mult)
{
  val_t *input = new val_t[size];

  for (int i = 0; i < size; i++)
  {
    input[i] = mult * i;
#ifdef __DEBUG
    std::cout << "Value : " << input[i] << std::endl;
#endif
  }

  return input;
}
/*********************************************************************/


#else


/*********************************************************************/
val_t my_random ()
{
  static int init = 0;

  if (init == 0)
  {
#ifdef __DEBUG
    std::cout << "Init : " << init << std::endl;
#endif
    srand48(getpid());
    init++;
  }

  long int rand = lrand48();

#ifdef __DEBUG
    std::cout << "Rand : " << rand << std::endl;
#endif

  return (val_t) rand;
}
/*********************************************************************/



/*********************************************************************/
val_t* initialisation (int size, int mult)
{
  val_t *input = new val_t[size];

  for (int i = 0; i < size; i++)
  {
    if (mult == 0) input[i] = 0;
    else input[i] = my_random();
#ifdef __DEBUG
    std::cout << "Value : " << input[i] << std::endl;
#endif
  }

#ifdef BENCH_MERGE
  std::sort(input, input+size);
#endif

  return input;
}
/*********************************************************************/
#endif
