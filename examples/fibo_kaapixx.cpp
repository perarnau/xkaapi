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
#include <iostream>
#include "kaapi++" // this is the new C++ interface for Kaapi

extern long cutoff;

// --------------------------------------------------------------------
/* Sequential fibo function
 */
long fiboseq( const long n ) {
  if( n<2 )
    return n;
  else
    return fiboseq(n-1)+fiboseq(n-2);
}

long fiboseq_On(const long n){
  if(n<2){
    return n;
  }else{

    long fibo=1;
    long fibo_p=1;
    long tmp=0;
    int i=0;
    for( i=0;i<n-2;i++){
      tmp = fibo+fibo_p;
      fibo_p=fibo;
      fibo=tmp;
    }
    return fibo;
  }
}


/* Sum two integers
 * this task reads a and b (read acces mode) and write their sum to res (write access mode)
 * it will wait until previous write to a and b are done
 * once finished, further read of res will be possible
 */
struct TaskSum : public ka::Task<3>::Signature<ka::W<long>, ka::R<long>, ka::R<long> > {};

template<>
struct TaskBodyCPU<TaskSum> //: public TaskSum
{
  void operator() ( ka::pointer_w<long> res, 
                    ka::pointer_r<long> a, 
                    ka::pointer_r<long> b ) 
  {
    /* write is used to write data to a Shared_w
     * read is used to read data from a Shared_r
     */
    *res = *a + *b;
  }
};
static ka::RegisterBodyCPU<TaskSum> dummy_object0;


/* Kaapi Fibo task.
   A Task is a type with respect a given signature. The signature specifies the number of arguments (2),
   and the type and access mode for each parameters.
   Here the first parameter is declared with a write mode. The second is passed by value.
 */
struct TaskFibo : public ka::Task<2>::Signature<ka::W<long>, const long > {};


/* Implementation for CPU machine 
*/
template<>
struct TaskBodyCPU<TaskFibo> /* : public TaskFibo */ 
{
  void operator() ( ka::pointer_w<long> res, const long n )
  {  
    if (n < 2){ //cutoff) {
      *res = n; //fiboseq(n);
      return;
    }
    else {
      ka::pointer<long> res1 = ka::Alloca<long>();
      ka::pointer<long> res2 = ka::Alloca<long>();

      /* the Spawn keyword is used to spawn new task
       * new tasks are executed in parallel as long as dependencies are respected
       */
      ka::Spawn<TaskFibo>() ( res1, n-1 );
      ka::Spawn<TaskFibo>() ( res2, n-2 );

      /* the Sum task depends on res1 and res2 which are written by previous tasks
       * it must wait until thoses tasks are finished
       */
      ka::Spawn<TaskSum>() ( res, res1, res2 );      
    }
  }
};

static ka::RegisterBodyCPU<TaskFibo> dummy_object;
