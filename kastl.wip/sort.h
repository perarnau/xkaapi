/*
 *  test_sort.cpp
 *  xkaapi
 *
 *  Created by DT on juin 2009.
 *  Created by FLM on decembre 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _XKAAPI_SORT_H
#define _XKAAPI_SORT_H
#include "kaapi.h"
#include "kaapi_utils.h"
#include <algorithm>
#include<numeric>
#include<functional>
#include<list>


template<class RandomAccessIterator, class Compare>
void  sort ( RandomAccessIterator begin, RandomAccessIterator end, Compare comp);

template<class RandomAccessIterator>
void  sort ( RandomAccessIterator begin, RandomAccessIterator end);

typedef ptrdiff_t Dis_type;


template <typename T, typename cmp=std::less<T> >
struct csort_compare_to_median
{
  T _value;
  csort_compare_to_median(){};
  csort_compare_to_median(T value):_value(value){ }
  bool operator()(T value) {
     return cmp()(value, _value);
  }
};


/** Stucture of a work for sort
*/
template<class RandomAccessIterator, class Compare>
class SortStruct {
public:
  typedef SortStruct <RandomAccessIterator, Compare> Self_t;
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type val_t;
  typedef typename std::pair<RandomAccessIterator, RandomAccessIterator> Interval_t;

  /* cstor */
  SortStruct(
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    Compare comp) : _ibeg(ibeg), _iend(iend), _comp(comp)
  {
    _work_list = new std::list< Interval_t >();
  }
  
  /* do sort */
  void doit(kaapi_task_t* task, kaapi_stack_t* stack);
 
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  Compare _comp;
  std::list< Interval_t >* _work_list;

  // request handler
  struct request_handler
  {
    RandomAccessIterator local_beg;
    RandomAccessIterator local_end;
    ptrdiff_t pargrain;

    request_handler(RandomAccessIterator& _local_beg,
		    RandomAccessIterator& _local_end,
		    ptrdiff_t _pargrain)
      : local_beg(_local_beg),
	local_end(_local_end),
	pargrain(_pargrain)
    {}

    bool operator() (Self_t* self_work, Self_t* output_work)
    {
      const Dis_type size = local_end - local_beg;
     
      /** split work if no work to steal */
      const int nb_work = self_work->_work_list->size();
      if(!nb_work || size < pargrain )
	{
	  // stop request processing
	  return false;
	}

      Interval_t interval = self_work->_work_list->back();

      output_work->_work_list = new std::list<Interval_t>();
      output_work->_ibeg = interval.first;
      output_work->_iend = interval.second;
      output_work->_comp = self_work->_comp;

      self_work->_work_list->pop_back();

      return true;
    }
  };

  typedef struct request_handler request_handler_t;

  /** splitter_work is called within the context of the steal point
  */
  int splitter( kaapi_stack_t* victim_stack, 
		kaapi_task_t* task,
		int count,
		kaapi_request_t* request )
  {
    const int total_count = count;
    int replied_count = 0;
    Dis_type par_grain = 512;
    int nb_work = _work_list->size();

    /* threshold should be defined (...) */
    if(!nb_work) {
       //if (size < par_grain) goto reply_failed;
      goto finish_splitter;
    }

    // iterate over requests
    {
      request_handler_t handler(_ibeg, _iend, par_grain);

      replied_count =
	kaapi_utils::foreach_request
	(
	 victim_stack, task,
	 count, request,
	 handler, this
	 );

      // mute victim state after processing
      _iend  = handler.local_end;
      _ibeg  = handler.local_beg;

      kaapi_assert_debug(_iend - _ibeg > 0);
    }

  finish_splitter:
    {
      // fail the remaining requests

      const int remaining_count = total_count - replied_count;

      if (remaining_count)
	{
	  kaapi_utils::fail_requests
	    (
	     victim_stack,
	     task,
	     remaining_count,
	     request + replied_count
	     );
	}
    }

    // all requests have been replied to
    return total_count;
 }


#if 0 // TODO_REDUCER
/* Called by the victim thread to collect work from one other thread
  */
 static void reducer( kaapi_steal_context_t* sc, void* thief_data,
                     SortStruct<RandomAccessIterator, Compare>* victim_data)
 {

   //std::cout << "my id = " << sc->_stack->_index << std::endl;
    SortStruct<RandomAccessIterator, Compare>* thief_work =
      (SortStruct<RandomAccessIterator, Compare>* )thief_data;

   //std::cout << "thief_work->rem_size = " << thief_work->_iend-thief_work->_ibeg << std::endl;
   //std::cout << "thief_work->list_size = " << thief_work->_work_list->size()  << std::endl;
   //thief_list.splice(++thief_list.begin(), tl); //merge with stolen thieft_list

    if (thief_work->_ibeg != thief_work->_iend)
    {
  #if defined(SEQ_SUBSORT)
      std::sort( thief_work->_ibeg, thief_work->_iend, thief_work->_comp );
  #else
      SortStruct<RandomAccessIterator, Compare>
        work( sc,
              thief_work->_ibeg,
              thief_work->_iend,
              thief_work->_comp);
      work.doit();
  #endif 
    }
 }
#endif // TODO_REDUCER


};


/** Adaptive sort
*/
template<class RandomAccessIterator, class Compare>
void SortStruct<RandomAccessIterator, Compare>::doit
(kaapi_task_t* task, kaapi_stack_t* stack)
{
  /* amount of work per iteration of the nano loop */
  Dis_type unit_size  = 1024;

  /* call of std::partition to split work into two parts*/
  Dis_type tmp_size = _iend-_ibeg ;
  bool not_finished = true;
  int nb_work = 0;   
 
   //To anticipate the work to steal
   if(tmp_size > unit_size) {
        //std::cout << "1 tmp_size = " << tmp_size << std::endl;
      // split_work( );

      Dis_type sz = _iend-_ibeg;
      val_t median = val_t(std:: __median(*_ibeg, *(_ibeg+sz/2), *(_ibeg+sz-1)));
     //Partion into two parts
      //RandomAccessIterator split = std::partition(_ibeg, _iend, csort_compare_to_median<val_t, _comp>(median));
      RandomAccessIterator __first = _ibeg;
      RandomAccessIterator __last = _iend;
     
      while (true)
      {
          while (_comp(*__first, median))
            ++__first;
          --__last;
          while (_comp(median, *__last))
            --__last;
          if (!(__first < __last))
            break;
          std::iter_swap(__first, __last);
          ++__first;
      }

     RandomAccessIterator split = __first;

     //Push the big size interval in list
      if( (_iend-split) >= (split-_ibeg) ) {
         _work_list->push_front(std::make_pair(split, _iend));
         _iend = split;
      } else {
         _work_list->push_front(std::make_pair(_ibeg, split));
        _ibeg = split;
      }
      kaapi_stealpoint( stack, task, kaapi_utils::static_splitter<Self_t> );
   }

  //Local computation
  do {
    /* definition of the steal point where steal_work may be called in case of steal request
       -here size is pass as parameter and updated in case of steal.
    */

     //kaapi_stealpoint( _sc, splitter, this);

     tmp_size = _iend-_ibeg;
     /* call of std::partition to split work into two parts*/
     if(tmp_size > unit_size) {
        //std::cout << "1 tmp_size = " << tmp_size << std::endl; 
      //split_work( );
      Dis_type sz = _iend-_ibeg;
      val_t median = val_t(std:: __median(*_ibeg, *(_ibeg+sz/2), *(_ibeg+sz-1)));
      //Partion into two parts
      RandomAccessIterator split =
	std::partition(_ibeg, _iend, csort_compare_to_median<val_t>(median));

      if( (_iend-split) >= (split-_ibeg) ) {
         _work_list->push_front(std::make_pair(split, _iend));
         _iend = split;
      } else {
        _work_list->push_front(std::make_pair(_ibeg, split));
        _ibeg = split;
      }
#if 0 // TODO_REDUCER
      kaapi_stealpoint( _sc, splitter, this);
#endif
     } 
     else {
       //std::cout << "2 tmp_size = " << tmp_size << std::endl;
     /** Local sorting */
      std::sort(_ibeg, _iend, _comp);
      

      nb_work = _work_list->size();

      if(!nb_work) { not_finished = false; _iend=_ibeg; }
      else {
        /** pop next work */
	Interval_t interval = _work_list->front();

        _ibeg = interval.first;
        _iend = interval.second;
        _work_list->pop_front();
        _ibeg = interval.first;
        _iend = interval.second;

        not_finished = true;
       }
#if 0 // TODO_REDUCER
       kaapi_stealpoint( _sc, splitter, this);
#endif
     }
   } while ( not_finished  );
}

/**
*/
template<class RandomAccessIterator>
void sort(RandomAccessIterator begin, RandomAccessIterator end)
{
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type val_t;
  typedef typename std::iterator_traits<RandomAccessIterator>::pointer ptr_type;

  SortStruct<RandomAccessIterator, std::less<val_t> >
    work( begin, end, std::less<val_t>());

  kaapi_utils::start_adaptive_task(&work);
}

#endif
