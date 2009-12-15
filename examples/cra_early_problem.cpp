/*
 *  cra_early_problem.cpp
 *  xkaapi
 *  Exemple on how to iterate through a vector with a sliding window of constant size.
 *  Created by TG on 9/12/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#include "kaapi.h"
#include <algorithm>
#include <athapascan-1>
#include <iostream>
#include <math.h>


/**
*/
struct BasicOperation 
{
  void operator()( double& value )
  {
    double d = value;
    int max = (int)d;
    for (int i=0; i<max; ++i) d = sin(d);
    value = d;
  }
};


/** Stucture of a work for CRA problem with early termination
*/
template<class InputIterator, class Function, class Predicate>
class CRAWork {
public:
  /* cstor */
  CRAWork(
    InputIterator ibeg,
    InputIterator iend,
    Function&  func
    Predicate& op
  ) : _ibeg(ibeg), _iend(iend), _func(func), _op(op)
  {
  }
  
  /* do computation */
  void doit(kaapi_task_t* task, kaapi_stack_t* data);

  /* */
  typedef CRAWork<InputIterator,Function,Predicate> Self_t;
  
protected:  
  InputIterator  _ibeg; /*                          */
  InputIterator  _iend; /*                          */
  Function&      _func; /* one copy per thread */
  Predicate&     _op;   /* one copy per thread */
  
  /* Entry in case of main execution */
  static void static_entrypoint(kaapi_task_t* task, kaapi_stack_t* data)
  {
    Self_t* w = kaapi_task_getargst(task, Self_t);
    w->doit(task, data);
  }

  /** splitter_work is called within the context of the steal point
  */
  static int splitter( kaapi_stack_t* stack, kaapi_task_t* self_task, 
                       int count, kaapi_request_t* request, 
                       double* ibeg, double** local_iend
                      )
  {
    int i = 0;

    size_t blocsize, size = (*local_iend - ibeg);
    InputIterator thief_end = *local_iend;

    Self_t* output_work =0; 

    /* threshold should be defined ? (...) */
    if (size < 32) return 0;

    /* Sliding window: do not give to thiefs more than WINDOW_SIZE size of work 
       Cannot occurs on the thefts because they have a initial work less thant WINDOW_SIZE.
    */
    if (size > WINDOW_SIZE) 
    {
      size = WINDOW_SIZE;
      thief_end = ibeg + WINDOW_SIZE;
    }
    
    /* keep a bloc for myself */
    blocsize = size / (1+count); 
    
    /* adjust the number of bloc in order to do not have bloc of size less than 1 */
    if (blocsize < 1) { count = size-1; blocsize = 1; }
    
    int reply_count = 0;
    /* reply to all thiefs */
    while (count >0)
    {
      if (kaapi_request_ok(&request[i]))
      {
        kaapi_stack_t* thief_stack = request[i].stack;
        kaapi_task_t*  thief_task  = kaapi_stack_toptask(thief_stack);
        kaapi_task_initadaptive( thief_stack, thief_task, KAAPI_TASK_ADAPT_DEFAULT);
        kaapi_task_setbody( thief_task, &static_entrypoint );
        kaapi_task_setargs( thief_task, kaapi_stack_pushdata(thief_stack, sizeof(CRAWork)) );
        output_work = kaapi_task_getargst(thief_task, CRAWork);

        output_work->_iend = thief_end;
        output_work->_ibeg = thief_end-blocsize;
        thief_end -= blocsize;
        kaapi_assert( output_work->_iend - output_work->_ibeg >0);
        
        kaapi_stack_pushtask(thief_stack);
        
//        atha::logfile() << "I'm split work to a the thief" << std::endl;

        /* reply ok (1) to the request */
        kaapi_request_reply( stack, self_task, &request[i], thief_stack, 1);
        --count; ++reply_count;
      }
      ++i;
    }
    /* mute the end of input work of the victim */
    *local_iend  = thief_end;
    kaapi_assert( *local_iend - ibeg >0);
    return reply_count;      
  }


  /* Called by the main thread to collect work from one other thief thread
  */
  static bool reducer( kaapi_stack_t* stack, kaapi_task_t* self_task,
                       CRAWork* thief_work,
                       CRAWork* victim_work,
                       double** ibeg, double** local_iend
                      )
  {
    if ((thief_work ==0) || (thief_work->_ibeg == thief_work->_iend))
    {
      if (victim_work->_ibeg == victim_work->_iend)
      {
        return false;
      }
      *local_iend = victim_work->_iend;
      return true;
    }

    /* master get unfinished work of the thief */
    *ibeg = thief_work->_ibeg;
    *local_iend = thief_work->_iend + (WINDOW_SIZE - (thief_work->_iend-thief_work->_ibeg));
    if (*local_iend > victim_work->_iend) 
      *local_iend = victim_work->_iend;

    return true;
  }
  
  static void display_reducer(kaapi_stack_t* stack, kaapi_task_t* self_task, void* arg_from_victim, CRAWork* mywork )
  {
    CRAWork* victim_work = (CRAWork*)arg_from_victim;
  }
};


/** Main entry point
*/
template<class InputIterator, class Function, class Predicate>
void CRAWork::doit( kaapi_task_t* task, kaapi_stack_t* stack )
{

  /* local iterator for the nano loop */
  InputIterator nano_iend;
  
  /* amount of work per iteration of the nano loop */
  const size_t unit_size = 32;  /* should be automatically computed */
  size_t tmp_size = 0;

  /* maximum iend to be done in sequential */
  InputIterator local_iend = _iend;

redo_work:  
  while (local_iend != _ibeg)
  {
    /* 
    */
    kaapi_stealpoint( stack, task, &splitter, _ibeg, &local_iend );

    tmp_size = local_iend-_ibeg;
    if (tmp_size < unit_size ) {
       nano_iend = local_iend;
    } else {
       nano_iend = _ibeg + unit_size;
    }

    /* sequential computation */
    std::for_each( _ibeg, nano_iend, BasicOperation() );

    _ibeg = nano_iend;

    /*
    */
    if (kaapi_preemptpoint( stack, task,     /* context */
                            display_reducer, /* function to call in case of preemption */
                            this,            /* arg to pass to the thread that do preemption */
                            this             /* here possible extra arguments to pass to the function call */
                        )) return;      
  }
  
  /* Try to preempt each theft thread in the reverse order defined by steal reponse (see Daouda order).
     Return true iff some work have been preempted and should be processed locally.
     0 is the value passed to the victim thread (no value). this, ibeg and local_iend are passed to 
     the call to reducer function.
     If work has been preempted, then ibeg != local_iend and should be done locally. See reducer function.
     If no more work has been preempted, it means that all the computation is finished.
  */
  if (kaapi_preempt_nextthief( stack, task, 
                                     this,                      /* arg to pass to the thief */
                                     &reducer,                  /* function to call if a preempt exist */
                                     this, &_ibeg, &local_iend  /* arg to pass to the function call reducer */
                            ))
  {
    goto redo_work;
  }

  /* Definition of the finalization point where main thread waits all the works.
     After this point, we enforce memory synchronisation: all data that has been writen before terminaison of the thiefs,
     could be read (...)
  */
  kaapi_finalize_steal( stack, task );
}


/**
*/
template<class InputIterator, class Function, class Predicate>
void cra_problem ( InputIterator begin, InputIterator end, Function& func, Predicate& op )
{
  /* push new adaptative task with following flags:
    - master finalization: the main thread will waits the end of the computation.
    - master preemption: the main thread will be able to preempt all thiefs.
  */
  kaapi_stack_t* stack = kaapi_self_stack();

  CRAWork<InputIterator,Function,Predicate> work( begin, end, func, op);

  /* create the task on the top of the stack */
  kaapi_task_t* task = kaapi_stack_toptask(stack);
  kaapi_task_initadaptive( stack, task, KAAPI_TASK_ADAPT_DEFAULT);
  kaapi_task_setargs(task, &work );
  
  /* push_it task on the top of the stack */
  kaapi_stack_pushtask(stack);

  work.doit( task, stack );
}



/** My Fast & Stupid function. 
*/
struct MyFunction {
  double operator()( double data ) const
  { return data / 2; }
};


/** My Fast & Stupid predicate. 
*/
struct MyPredicate {
  MyPredicate( double value ) : _value(value) {}
  bool operator()( double& data ) const
  { return data > _value; }
};


/** Container for iterator over interger interval [b,e) 
*/
class AbstractContainer {
public:
  AbstractContainer(int b, int e)
   : _beg(b), _end(e)
  {}
  AbstractContainer()
   : _beg(0), _end(0)
  {}
  
  class const_iterator {
    int _curr;
    const_iterator( int val ) : _curr(val) {}
    friend class AbstractContainer;
  public:
    double operator*() const 
    { return double(_curr); }
    const_iterator& operator++() 
    { ++_curr; return *this; }
    const_iterator operator++(int)
    { const_iterator retval(*this); ++_curr; return retval; }
    bool operator== (const const_iterator& it)
    { return _curr == it._curr; }
    bool operator!= (const const_iterator& it)
    { return _curr != it._curr; }
  };

  const_iterator begin() 
  { return const_iterator(_beg); }
  
  const_iterator end()
  { return const_iterator(_end); }

private:
  int _beg;
  int _end;
};


/** Abstract container for iterator over infinite uniform random number in [0,..,maxvalue)
*/
class AbstractRandomContainer {
public:
  AbstractRandomContainer(int seed, int maxval)
   : _seed(s), _maxval(e)
  {}
  AbstractRandomContainer()
   : _seed(rand()), _maxval(100)
  {}
  
  class const_iterator {
    int _curr;
    int _seed;
    int _maxval;
    const_iterator( int seed, int maxval ) : _seed(seed), _maxval(maxval) {}
    friend class AbstractRandomContainer;
  public:
    double operator*() const 
    { return double(_curr); }
    const_iterator& operator++() 
    { 
      _curr = rand_r(_seed) % _maxval;
      return *this; 
    }
    const_iterator operator++(int)
    { 
      const_iterator retval(*this); 
      _curr = rand_r(_seed) % _maxval;
      return retval; 
    }
    bool operator== (const const_iterator& it)
    { return false; }
    bool operator!= (const const_iterator& it)
    { return true; }
  };

  const_iterator begin() 
  { return const_iterator(_seed,_maxval); }
  
  const_iterator end()
  { return const_iterator(0,0); }

private:
  int _seed;
  int _maxval;
};


/**
*/
int main( int argc, char** argv )
{
  /* */
  AbstractRandomContainer randomsequence;
  
  
  MyFunction  func;
  MyPredicate op(1024);
  cra_problem( randomsequence.begin(), randomsequence.end(), func, op );
  return 0;
}
