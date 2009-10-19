/*
 *  test_sort.cpp
 *  ckaapi
 *
 *  Created by DT on juin 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_SORT_H
#define _CKAAPI_SORT_H
#include "kaapi_adapt.h"
#include <algorithm>
#include<numeric>
#include<functional>
#include<list>


template<class RandomAccessIterator, class Compare>
void  sort ( kaapi_steal_context_t* stealcontext, 
                    RandomAccessIterator begin, RandomAccessIterator end, Compare comp);

template<class RandomAccessIterator>
void  sort ( kaapi_steal_context_t* stealcontext, 
                    RandomAccessIterator begin, RandomAccessIterator end);

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
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type val_t;
  typedef typename std::pair<RandomAccessIterator, RandomAccessIterator> Interval_t;
  /* cstor */
  SortStruct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    Compare comp) : _sc(sc), _ibeg(ibeg), _iend(iend), _comp(comp)
  {
    _work_list = new std::list< Interval_t >();
  }
  
  /* do sort */
  void doit();


 
protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  Compare _comp;
  std::list< Interval_t >* _work_list;
 
   /* split_work */
  void split_work( ) 
  {
      Dis_type sz = _iend-_ibeg;
      val_t median = val_t(std:: __median(*_ibeg, *(_ibeg+sz/2), *(_ibeg+sz-1)));
     //Partion into two parts
      RandomAccessIterator split = std::partition(_ibeg, _iend, csort_compare_to_median<val_t>(median));
      
      if( (_iend-split) >= (split-_ibeg) ) {
         _work_list->push_front(std::make_pair(split, _iend));
         _iend = split;
      } else {
        _work_list->push_front(std::make_pair(_ibeg, split));
        _ibeg = split;
      }
  
  }

  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    SortStruct<RandomAccessIterator, Compare>* w = (SortStruct<RandomAccessIterator, Compare>*)data;
    w->_sc = sc;
    w->_work_list = new std::list<typename std::pair<RandomAccessIterator, RandomAccessIterator> >();
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request,
                        SortStruct<RandomAccessIterator, Compare>* victim_data)
  {
    
    Dis_type size = victim_data->_iend-victim_data->_ibeg; 
    RandomAccessIterator local_end = victim_data->_iend;
    RandomAccessIterator local_beg = victim_data->_ibeg;
    Dis_type par_grain = 512;
    SortStruct<RandomAccessIterator, Compare>* output_work =0;
    int i = 0;
   
    int nb_work = victim_data->_work_list->size();    

    /* threshold should be defined (...) */
    if(!nb_work) {
       //if (size < par_grain) goto reply_failed;
       goto reply_failed;
    }
 
    while (count >0)
    {

      size =  local_end - local_beg; 
     
      /** split work if no work to steal */
       nb_work = victim_data->_work_list->size();
       if(!nb_work || size < par_grain ) goto reply_failed;  

#if 0
       if(!nb_work) 
       {
       
         if ( (size < par_grain) ) goto reply_failed;
        
          //std::cout << "splitting......" << std::endl;
         // split reamining work into two parts 
         val_t median = val_t(std:: __median(*local_beg, *(local_beg+size/2), *(local_beg+size-1)));
         //Partion into two parts
         RandomAccessIterator split = std::partition(local_beg, local_end, csort_compare_to_median<val_t>(median));

        if( (local_end-split) >= (split-local_beg) ) {
           work_list->push_front(std::make_pair(split, local_end));
           local_end = split;
        } else {
          work_list->push_front(std::make_pair(local_beg, split));
          local_beg = split;
        }
 
      }
#endif

      if (request[i] !=0)
      {
        
         //static int cpt = 0;
         //if(stealcontext->_stack->_index==0) std::cout << "id 0 : nb_steal = " << ++cpt << std::endl;

        if (kaapi_steal_context_alloc_result( stealcontext,
                                              request[i],
                                              (void**)&output_work,
                                              sizeof(SortStruct<RandomAccessIterator, Compare>)
                                             ) ==0)
        {



         std::pair<RandomAccessIterator, RandomAccessIterator> interval = victim_data->_work_list->back();
         output_work->_ibeg = interval.first;
         output_work->_iend = interval.second;
         output_work->_comp = victim_data->_comp;
         //output_work->_work_list = new std::list<typename std::pair<RandomAccessIterator, RandomAccessIterator> >(); 
         victim_data->_work_list->pop_back();

         //ckaapi_assert( output_work->_iend - output_work->_ibeg >=0);
        //std::cout << "output_work->_iend-output_work->_ibeg=" << output_work->_iend-output_work->_ibeg << std::endl;
          /* reply ok (1) to the request */
         kaapi_request_reply( request[i], stealcontext, &thief_entrypoint, 1, CKAAPI_MASTER_FINALIZE_FLAG);
          
          std::cout << "split_size = " << output_work->_iend - output_work->_ibeg << std::endl;

       }
        else {
        /* reply failed (=last 0 in parameter) to the request */
        kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
     }
        --count;
    }
    ++i;
  }
  /* mute the beg and end of input work of the victim */
  victim_data->_iend  = local_end;
  victim_data->_ibeg  = local_beg;
  ckaapi_assert( victim_data->_iend - victim_data->_ibeg >=0);
  return;

   reply_failed :
     while (count >0)
    {
      if (request[i] !=0)
      {
        /* reply failed (=last 0 in parameter) to the request */
        kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
        --count;
      }
      ++i;
    }
   /* mute the beg and end of input work of the victim */
  victim_data->_iend  = local_end;
  victim_data->_ibeg  = local_beg;
  ckaapi_assert( victim_data->_iend - victim_data->_ibeg >=0);
 }


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


};


/** Adaptive sort
*/
template<class RandomAccessIterator, class Compare>
void SortStruct<RandomAccessIterator, Compare>::doit()
{

  /* amount of work per iteration of the nano loop */
  Dis_type unit_size  = 1024;

  /* call of std::partition to split work into two parts*/
  Dis_type tmp_size = tmp_size = _iend-_ibeg ;
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
      kaapi_stealpoint( _sc, splitter, this);
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
      RandomAccessIterator split = std::partition(_ibeg, _iend, csort_compare_to_median<val_t>(median));

      if( (_iend-split) >= (split-_ibeg) ) {
         _work_list->push_front(std::make_pair(split, _iend));
         _iend = split;
      } else {
        _work_list->push_front(std::make_pair(_ibeg, split));
        _ibeg = split;
      }
      kaapi_stealpoint( _sc, splitter, this);
     } 
     else {
       //std::cout << "2 tmp_size = " << tmp_size << std::endl;
     /** Local sorting */
      std::sort(_ibeg, _iend, _comp);
      

      nb_work = _work_list->size();

      if(!nb_work) { not_finished = false; _iend=_ibeg; }
      else {
        /** pop next work */
        std::pair<RandomAccessIterator, RandomAccessIterator> interval = _work_list->front();
        _ibeg = interval.first;
        _iend = interval.second;
        _work_list->pop_front();
        _ibeg = interval.first;
        _iend = interval.second;

        not_finished = true;
       }
       kaapi_stealpoint( _sc, splitter, this);
     }
   } while ( not_finished  );

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this);

}

/**
*/
template<class RandomAccessIterator, class Compare>
void sort(kaapi_steal_context_t* stealcontext, 
                 RandomAccessIterator begin, RandomAccessIterator end, Compare comp)
{
  typedef typename std::iterator_traits<RandomAccessIterator>::pointer ptr_type;
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type val_t;
  Dis_type size = end-begin;
  SortStruct<RandomAccessIterator, Compare> work( stealcontext, begin, end, comp);
  work.doit();
}

template<class RandomAccessIterator>
void sort(kaapi_steal_context_t* stealcontext, 
                 RandomAccessIterator begin, RandomAccessIterator end)
{
  kaapi_steal_context_initpush( stealcontext );
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type val_t;
  sort(stealcontext, begin, end, std::less<val_t>());
}

#endif
