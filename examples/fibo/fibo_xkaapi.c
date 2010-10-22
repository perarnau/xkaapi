/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#include "kaapi.h"
#include <stdio.h>
#include <stddef.h>

int fiboseq(int n)
{ return (n<2 ? n : fiboseq(n-1)+fiboseq(n-2) ); }

void sum_body( void* taskarg, kaapi_thread_t* thread );
void fibo_body( void* taskarg, kaapi_thread_t* thread );
void print_body( void* taskarg, kaapi_thread_t* thread );

typedef struct sum_arg_t {
  kaapi_access_t result;
  kaapi_access_t subresult1;
  kaapi_access_t subresult2;
} sum_arg_t;

KAAPI_REGISTER_TASKFORMAT( sum_format,
    "sum",
    sum_body,
    sizeof(sum_arg_t),
    3,
    (kaapi_access_mode_t[])   { KAAPI_ACCESS_MODE_W, KAAPI_ACCESS_MODE_R, KAAPI_ACCESS_MODE_R },
    (kaapi_offset_t[])        { offsetof(sum_arg_t, result), offsetof(sum_arg_t, subresult1), offsetof(sum_arg_t, subresult2) },
    (const struct kaapi_format_t*[]) { kaapi_int_format, kaapi_int_format, kaapi_int_format },
    0
)

void sum_body( void* taskarg, kaapi_thread_t* thread )
{
  sum_arg_t* arg0 = (sum_arg_t*)taskarg;
  *KAAPI_DATA(int, arg0->result) = *KAAPI_DATA(int, arg0->subresult1) + *KAAPI_DATA(int, arg0->subresult2);
}

typedef struct fibo_arg_t {
  int  n;
  kaapi_access_t result;
} fibo_arg_t;

KAAPI_REGISTER_TASKFORMAT( fibo_format,
    "fibo",
    fibo_body,
    sizeof(fibo_arg_t),
    2,
    (kaapi_access_mode_t[])   { KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_W },
    (kaapi_offset_t[])        { offsetof(fibo_arg_t, n), offsetof(fibo_arg_t, result) },
    (const struct kaapi_format_t*[]) { kaapi_int_format, kaapi_int_format },
    0
)

void fibo_body( void* taskarg, kaapi_thread_t* thread )
{
  fibo_arg_t* arg0 = (fibo_arg_t*)taskarg;
  if (arg0->n < 2)
  {
    *KAAPI_DATA(int, arg0->result) = arg0->n; //fiboseq(arg0->n);
  }
  else {
    fibo_arg_t* argf1;
    fibo_arg_t* argf2;
    sum_arg_t*  args;
    kaapi_task_t* task_sum;
    kaapi_task_t* task1;
    kaapi_task_t* task2;

    task1 = kaapi_thread_toptask(thread);
    kaapi_task_initdfg( task1, fibo_body, kaapi_thread_pushdata(thread, sizeof(fibo_arg_t)) );
    argf1 = kaapi_task_getargst( task1, fibo_arg_t );
    argf1->n = arg0->n - 1;
    kaapi_thread_allocateshareddata( &argf1->result, thread, sizeof(int) );
    kaapi_thread_pushtask(thread);

    task2 = kaapi_thread_toptask(thread);
    kaapi_task_initdfg( task2, fibo_body, kaapi_thread_pushdata(thread, sizeof(fibo_arg_t)) );
    argf2 = kaapi_task_getargst( task2, fibo_arg_t);
    argf2->n      = arg0->n - 2;
    kaapi_thread_allocateshareddata( &argf2->result, thread, sizeof(int) );
    kaapi_thread_pushtask(thread);

    task_sum = kaapi_thread_toptask(thread);
    kaapi_task_initdfg( task_sum, sum_body, kaapi_thread_pushdata(thread, sizeof(sum_arg_t)) );
    args = kaapi_task_getargst( task_sum, sum_arg_t);
    args->result.data     = arg0->result.data;
    args->subresult1.data = argf1->result.data;
    args->subresult2.data = argf2->result.data;
    kaapi_thread_pushtask(thread);
 }
}

typedef struct print_arg_t {
  double delay;
  int n;
  int niter;
  kaapi_access_t result;
} print_arg_t;

KAAPI_REGISTER_TASKFORMAT( print_format,
    "print",
    print_body,
    sizeof(print_arg_t),
    4,
    (kaapi_access_mode_t[])   { KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_RW },
    (kaapi_offset_t[])        { offsetof(print_arg_t, delay), offsetof(print_arg_t, n), offsetof(print_arg_t, niter), offsetof(print_arg_t, result) },
    (const struct kaapi_format_t*[]) { kaapi_double_format, kaapi_int_format, kaapi_int_format, kaapi_int_format },
    0
)

void print_body( void* taskarg, kaapi_thread_t* thread )
{
  print_arg_t* arg0 = (print_arg_t*)taskarg;
  printf("Fibo(%i)=%i\n", arg0->n, *KAAPI_DATA(int, arg0->result));
  printf("Time: %g\n", arg0->delay/arg0->niter);
}

int main(int argc, char** argv)
{
  kaapi_frame_t frame;
  double t0, t1;
  kaapi_access_t result1;
  long value_result;
  int n;
  int niter;
  int i;
  fibo_arg_t* argf;
  print_arg_t* argp;
  kaapi_task_t* task;
  kaapi_thread_t* thread;
  
  if (argc >1)
    n = atoi(argv[1]);
  else 
    n = 30;
  if (argc >2)
    niter =  atoi(argv[2]);
  else 
    niter = 1;

  kaapi_init();
  thread = kaapi_self_thread();
  kaapi_thread_save_frame(thread, &frame);
  
  t0 = kaapi_get_elapsedtime();
  for ( i=-1; i<niter; ++i)
  {
    if (i ==0) t0 = kaapi_get_elapsedtime();
  
    kaapi_access_init( &result1, &value_result );

    task = kaapi_thread_toptask(thread);
    kaapi_task_init( task, fibo_body, kaapi_thread_pushdata(thread, sizeof(fibo_arg_t)) );
    argf = kaapi_task_getargst( task, fibo_arg_t );
    argf->n      = n;
    argf->result = result1;
    kaapi_thread_pushtask(thread);
  }
  kaapi_sched_sync( );
  t1 = kaapi_get_elapsedtime();

  /* push print task */
  task = kaapi_thread_toptask(thread);
  kaapi_task_init( task, print_body, kaapi_thread_pushdata(thread, sizeof(print_arg_t)) );
  argp = kaapi_task_getargst( task, print_arg_t );
  argp->delay  = t1-t0;
  argp->n      = n;
  argp->niter  = niter;
  argp->result = result1;
  kaapi_thread_pushtask(thread);

  kaapi_sched_sync( );
  
  printf("After sync: Fibo(%i)=%li\n", n, value_result);
  printf("Time Fibo(%i): %f\n", n, t1-t0);

  kaapi_finalize();
  
  return 0;
}
