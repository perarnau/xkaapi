/*
 *  test_merge.cpp
 *  xkaapi
 *
 *  Created by DT on fevrier 2009.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#ifndef _CKAAPI_MERGE_H
#define _CKAAPI_MERGE_H
#include "kaapi_adapt.h"
#include <algorithm>
#include<numeric>
#include<functional>

template<class RandomAccessIterator, class RandomAccessIterator2, class RandomAccessIterator3, class Compare>
RandomAccessIterator3  merge ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, 
                        RandomAccessIterator end, RandomAccessIterator2 begin2, RandomAccessIterator2 end2,
                        RandomAccessIterator3 res, Compare comp);

template<class RandomAccessIterator, class RandomAccessIterator2, class RandomAccessIterator3>
RandomAccessIterator3  merge ( kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, 
                  RandomAccessIterator end, RandomAccessIterator2 begin2, RandomAccessIterator2 end2, 
                  RandomAccessIterator3 res);

/** Stucture of a work for merge
*/
template<class RandomAccessIterator, class RandomAccessIterator2, class RandomAccessIterator3, class Compare>
class MergeStruct {
public:
  /* cstor */
  MergeStruct(
    kaapi_steal_context_t* sc,
    RandomAccessIterator ibeg,
    RandomAccessIterator iend,
    RandomAccessIterator2 ibeg2,
    RandomAccessIterator2 iend2,
    RandomAccessIterator3 res,
    Compare comp) : _sc(sc), _ibeg(ibeg), _iend(iend), _ibeg2(ibeg2), _iend2(iend2), _res(res),  _comp(comp)
  {}
  
  /* do merge */
  void doit();

 
protected:  
  kaapi_steal_context_t* _sc;
  RandomAccessIterator  _ibeg;
  RandomAccessIterator  _iend;
  RandomAccessIterator2 _ibeg2;
  RandomAccessIterator2 _iend2;
  RandomAccessIterator3 _res;
  Compare _comp;
  /* Entry in case of thief execution */
  static void thief_entrypoint(kaapi_steal_context_t* sc, void* data)
  {
    MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare>* w = (MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare>*)data;
    w->_sc = sc;
    w->doit();
  }

  /** splitter_work is called within the context of the steal point
  */
  static void splitter( kaapi_steal_context_t* stealcontext, int count, kaapi_steal_request_t** request, 
                        RandomAccessIterator ibeg, RandomAccessIterator& iend, 
                        RandomAccessIterator2 ibeg2, RandomAccessIterator2& iend2, 
                        RandomAccessIterator3 res, Compare comp)
  {

    ptrdiff_t size1 = iend-ibeg;
    ptrdiff_t size2 = iend2-ibeg2;
    ptrdiff_t pargrain = 32;
    MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare>* output_work =0;
    RandomAccessIterator local_end = iend;
    RandomAccessIterator2 local_end2 = iend2;
    int i = 0;

    /* threshold should be defined (...) */
    if ((size1 < pargrain)&&(size2 < pargrain)) goto reply_failed;

    while (count >0)
    {
 
      size1 = local_end - ibeg;
      size2 = local_end2 - ibeg2;

      if ((size1 < pargrain)&&(size2 < pargrain)) goto reply_failed;
 
      if (request[i] !=0)
      {
        if (kaapi_steal_context_alloc_result( stealcontext,
                                              request[i],
                                              (void**)&output_work,
                                              sizeof(MergeStruct<RandomAccessIterator, RandomAccessIterator2,
                                                 RandomAccessIterator3, Compare>)
                                             ) ==0)
        {


        if(size1 > size2) {
        
         RandomAccessIterator mid = local_end - (size1/2);
         RandomAccessIterator2  split2 = std::lower_bound(ibeg2, local_end2, *mid, comp);
 
         output_work->_ibeg = mid;
         output_work->_iend = local_end;
         local_end = mid;
         output_work->_ibeg2 = split2;
         output_work->_iend2 = local_end2;
         local_end2 = split2;
         output_work->_res = res + ((local_end - ibeg) + (local_end2 - ibeg2));
         //xkaapi_assert( output_work->_iend - output_work->_ibeg >=0);
         //xkaapi_assert( output_work->_iend2 - output_work->_ibeg2 >=0);
         output_work->_comp   = comp;

         /* reply ok (1) to the request */
         kaapi_request_reply( request[i], stealcontext, &thief_entrypoint, 1, CKAAPI_MASTER_FINALIZE_FLAG);
      
        } else {

        RandomAccessIterator2 mid2 = local_end2 - (size2/2);
        RandomAccessIterator  split1 = std::upper_bound(ibeg, local_end, *mid2, comp);

        output_work->_ibeg = split1;
        output_work->_iend = local_end;
        local_end = split1;
        output_work->_ibeg2 = mid2;
        output_work->_iend2 = local_end2;
        local_end2 = mid2;
        output_work->_res = res + ((local_end - ibeg) + (local_end2 - ibeg2));
        xkaapi_assert( output_work->_iend - output_work->_ibeg >=0);
        xkaapi_assert( output_work->_iend2 - output_work->_ibeg2 >=0);
        output_work->_comp   = comp;

        /* reply ok (1) to the request */
        kaapi_request_reply( request[i], stealcontext, &thief_entrypoint, 1, CKAAPI_MASTER_FINALIZE_FLAG);     

       }
     }
     else {
        /* reply failed (=last 0 in parameter) to the request */
        kaapi_request_reply( request[i], stealcontext, 0, 0, CKAAPI_DEFAULT_FINALIZE_FLAG);
     }
        --count;
    }
    ++i;
  }
  /* mute the end of input work of the victim */
  iend  = local_end;
  iend2  = local_end2;
  xkaapi_assert( iend - ibeg >=0);
  xkaapi_assert( iend2 - ibeg2 >=0);
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
   /* mute the end of input work of the victim */
   iend  = local_end;
   iend2  = local_end2;
   //xkaapi_assert( iend - ibeg >=0);
   //xkaapi_assert( iend2 - ibeg2 >=0);

   
  }


  /* Called by the victim thread to collect work from one other thread
  */
  static void reducer( kaapi_steal_context_t* sc, void* thief_data, 
                    MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, 
                    Compare>* victim_data )
  {
    MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare>* thief_work =
     (MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare>*) thief_data;


   //std::cout << "thief_work->_iend-thief_work->_ibeg=" << thief_work->_iend-thief_work->_ibeg << std::endl;
   //std::cout << "thief_work->_iend2-thief_work->_ibeg2=" << thief_work->_iend2-thief_work->_ibeg2 << std::endl;
 
    if ((thief_work->_ibeg != thief_work->_iend) || (thief_work->_ibeg2 != thief_work->_iend2))
    {
    #if defined(SEQ_SUBMERGE)
      std::merge(thief_work->_ibeg, thief_work->_iend, thief_work->_ibeg2, thief_work->_iend2,
                 thief_work->_res, thief_work->_comp );
    #else
      MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare>
        work( sc,
              thief_work->_ibeg,
              thief_work->_iend,
              thief_work->_ibeg2,
              thief_work->_iend2,
              thief_work->_res,
              thief_work->_comp);
      work.doit();
     #endif
    }
  }

};

/** Adaptive merge
*/
template<class RandomAccessIterator, class RandomAccessIterator2, class RandomAccessIterator3, 
         class Compare>
void MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare>::doit()
{
  /* local iterator for the nano loop */
  RandomAccessIterator nano_iend  =  _ibeg;
  RandomAccessIterator2 nano_iend2 =  _ibeg2; 


  /* amount of work per iteration of the nano loop */
  ptrdiff_t unit_size  = 512;
  ptrdiff_t unit_size2 = 512;
  ptrdiff_t tmp_size;

  while ( (_iend != _ibeg) || (_iend2 != _ibeg2))
  {
    /* definition of the steal point where steal_work may be called in case of steal request 
       -here size is pass as parameter and updated in case of steal.
    */
    kaapi_stealpoint( _sc, splitter, _ibeg, _iend, _ibeg2, _iend2, _res, _comp);

    /* nano computation of range1 is finished*/
    if((_iend != _ibeg) && (_ibeg == nano_iend) || (_iend < nano_iend)) {
       tmp_size = _iend-_ibeg;
       if(tmp_size < unit_size ) {
         unit_size = tmp_size; nano_iend = _iend;
       } else {
       nano_iend = _ibeg + unit_size;
       }
    }

    /* nano computation of range2 is finished*/
    if((_iend2 != _ibeg2) && (_ibeg2 == nano_iend2) || (_iend2 < nano_iend2) ) {
       tmp_size = _iend2-_ibeg2;
       if(tmp_size < unit_size2 ) {
         unit_size2 = tmp_size; nano_iend2 = _iend2;
       } else {
       nano_iend2 = _ibeg2 + unit_size;
       }
    }

    /* sequential computation */ 
    if(_ibeg >= nano_iend) { 
       _res = std::copy(_ibeg2, nano_iend2, _res);
       _ibeg2 = nano_iend2;
    }
    else if(_ibeg2 >= nano_iend2) {
       _res = std::copy(_ibeg, nano_iend, _res);
        _ibeg = nano_iend;
    }
    else {
          while((_ibeg!=nano_iend) && (_ibeg2!=nano_iend2)) {
             if(_comp(*_ibeg2, *_ibeg))  *_res++ = *_ibeg2++;
             else *_res++ = *_ibeg++;
          }
    }
 
    //TG a voir apres if (kaapi_preemptpoint( _sc, 0 )) return ;
  }

  /* definition of the finalization point where all stolen work a interrupt and collected */
  kaapi_finalize_steal( _sc, 0, (kaapi_reducer_function_t)&reducer, this );

  /* Here the thiefs have finish the computation and returns their inits which have been reduced using reducer function. */  
}


/**
*/
template<class RandomAccessIterator, class RandomAccessIterator2, class RandomAccessIterator3, class Compare>
RandomAccessIterator3 merge(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, 
                  RandomAccessIterator end, RandomAccessIterator2 begin2, RandomAccessIterator2 end2,
                  RandomAccessIterator3 res, Compare comp)
{
  MergeStruct<RandomAccessIterator, RandomAccessIterator2, RandomAccessIterator3, Compare> work( stealcontext, 
         begin, end, begin2, end2, res, comp);
  work.doit();

 return res + ((end-begin) + (end2-begin2));
}

template<class RandomAccessIterator, class RandomAccessIterator2, class RandomAccessIterator3>
RandomAccessIterator3 merge(kaapi_steal_context_t* stealcontext, RandomAccessIterator begin, 
            RandomAccessIterator end, RandomAccessIterator2 begin2, RandomAccessIterator2 end2, 
            RandomAccessIterator3 res)
{
  kaapi_steal_context_initpush( stealcontext );
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type val_t;
 return merge(stealcontext, begin, end, begin2,  end2, res, std::less<val_t>());
}

#endif
