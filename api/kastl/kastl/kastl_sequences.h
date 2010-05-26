/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@gmail.com / fabien.lementec@imag.fr
 
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
#ifndef _KASTL_SEQUENCES_H_
#define _KASTL_SEQUENCES_H_
#include "kastl/kastl_workqueue.h"
#include <limits>


namespace kastl {
  
namespace rts {

// dummy type used to template non valid types
struct dummy_type {};


/* -------------------------------------------------------------------- */
/* Sequence representation. Using specialisation                        */
/* -------------------------------------------------------------------- */
template <
    typename RandomIterator1, 
    typename RandomIterator2=dummy_type, 
    typename RandomIterator3=dummy_type, 
    typename RandomIterator4=dummy_type 
>
struct SequenceRep {
  typedef SequenceRep<RandomIterator1,RandomIterator2,RandomIterator3,RandomIterator4> Self_t;
  SequenceRep() {}
  SequenceRep( RandomIterator1 r1,
               RandomIterator2 r2,
               RandomIterator3 r3,
               RandomIterator4 r4
  ) : ri1(r1), ri2(r2), ri3(r3), ri4(r4)
  {}
  RandomIterator1 ri1;
  RandomIterator2 ri2;
  RandomIterator3 ri3;
  RandomIterator4 ri4;
  void shift( const Self_t& o, int dist )
  { 
    ri1 = o.ri1+dist; 
    ri2 = o.ri2+dist; 
    ri3 = o.ri3+dist; 
    ri4 = o.ri4+dist; 
  }
};

/*  specialisation to gain space because sizeof(dummy_type) != 0 */
template <typename RandomIterator1>
struct SequenceRep<RandomIterator1,dummy_type,dummy_type,dummy_type> {
  typedef SequenceRep<RandomIterator1,dummy_type,dummy_type,dummy_type> Self_t;
  SequenceRep() {}
  SequenceRep( RandomIterator1 r1
  ) : ri1(r1)
  {}
  RandomIterator1 ri1;
  
  void shift( const Self_t& o, int dist )
  { ri1 = o.ri1+dist; }
};

template <typename RandomIterator1, typename RandomIterator2>
struct SequenceRep<RandomIterator1,RandomIterator2,dummy_type,dummy_type> {
  typedef SequenceRep<RandomIterator1,RandomIterator2,dummy_type,dummy_type> Self_t;
  SequenceRep() {}
  SequenceRep( RandomIterator1 r1,
               RandomIterator2 r2
  ) : ri1(r1), ri2(r2)
  {}
  RandomIterator1 ri1;
  RandomIterator2 ri2;
  void shift( const Self_t& o, int dist )
  { 
    ri1 = o.ri1+dist; 
    ri2 = o.ri2+dist; 
  }
};

template <typename RandomIterator1, typename RandomIterator2, typename RandomIterator3>
struct SequenceRep<RandomIterator1,RandomIterator2,RandomIterator3,dummy_type> {
  typedef SequenceRep<RandomIterator1,RandomIterator2,RandomIterator3,dummy_type> Self_t;
  SequenceRep() {}
  SequenceRep( RandomIterator1 r1,
               RandomIterator2 r2,
               RandomIterator3 r3
  ) : ri1(r1), ri2(r2), ri3(r3)
  {}
  RandomIterator1 ri1;
  RandomIterator2 ri2;
  RandomIterator3 ri3;
  void shift( const Self_t& o, int dist )
  { 
    ri1 = o.ri1+dist; 
    ri2 = o.ri2+dist; 
    ri3 = o.ri3+dist; 
  }
};



/* -------------------------------------------------------------------- */
/* Range : a Sequence Rep + size                                        */
/* -------------------------------------------------------------------- */
template <
    typename RandomIterator1, 
    typename RandomIterator2=dummy_type, 
    typename RandomIterator3=dummy_type, 
    typename RandomIterator4=dummy_type 
>
class Range {
public:
  typedef SequenceRep<RandomIterator1,RandomIterator2,RandomIterator3,RandomIterator4> iterator_type;
  typedef range_t<64>::size_type size_type;

  typedef RandomIterator1 iterator1_type;
  typedef RandomIterator2 iterator2_type;
  typedef RandomIterator3 iterator3_type;
  typedef RandomIterator4 iterator4_type;

  Range() : _size(0)
  {}

  Range(const iterator1_type& beg, const size_type& size)
    : _beg(beg), _size(size)
  {}

  /**/
  bool is_empty() const
  { return _size ==0; }

  /**/
  size_type size() const
  { return _size; }
  
  /**/
  RandomIterator1 begin()
  { return _beg.ri1; }
  /**/
  RandomIterator1 end()
  { return _beg.ri1+_size; }


  /**/
  RandomIterator1 begin1()
  { return _beg.ri1; }

  /**/
  RandomIterator2 begin2()
  { return _beg.ri2; }

  /**/
  RandomIterator3 begin3()
  { return _beg.ri3; }

  /**/
  RandomIterator4 begin4()
  { return _beg.ri4; }

  /**/
  RandomIterator1 end1()
  { return _beg.ri1+_size; }

  /**/
  RandomIterator2 end2()
  { return _beg.ri2+_size; }

  /**/
  RandomIterator3 end3()
  { return _beg.ri3+_size; }

  /**/
  RandomIterator4 end4()
  { return _beg.ri4+_size; }
  
public: /* access by Sequence */
  iterator_type                _beg;
  size_type                    _size;
};



/* -------------------------------------------------------------------- */
/* Sequence: SequenceRep + WorkQueue                                    */
/* -------------------------------------------------------------------- */
template <
    typename RandomIterator1, 
    typename RandomIterator2=dummy_type, 
    typename RandomIterator3=dummy_type, 
    typename RandomIterator4=dummy_type 
>
class Sequence {
public:
  /* it is called iterator_type because it represents the vector of iterators.... */
  typedef SequenceRep<RandomIterator1,RandomIterator2,RandomIterator3,RandomIterator4> iterator_type;
  typedef Range<RandomIterator1,RandomIterator2,RandomIterator3,RandomIterator4>       range_type;
  typedef work_queue_t<64>::size_type                                                  size_type;
  
  /* default ctsor */
  Sequence( )
  {}

  /* ctsor */
  Sequence( const RandomIterator1& i1, size_type n )
   : _wq(work_queue_t<64>::range_type(0,n)), _rep(i1)
  {}
  Sequence( const RandomIterator1& i1, const RandomIterator2& i2, size_type n )
   : _wq(work_queue_t<64>::range_type(0,n)), _rep(i1,i2)
  {}
  Sequence( const RandomIterator1& i1, const RandomIterator2& i2, const RandomIterator3& i3, size_type n )
   : _wq(work_queue_t<64>::range_type(0,n)), _rep(i1,i2,i3)
  {}
  Sequence( const RandomIterator1& i1, const RandomIterator2& i2, const RandomIterator3& i3, const RandomIterator4& i4, size_type n )
   : _wq(work_queue_t<64>::range_type(0,n)), _rep(i1,i2,i3,i4)
  {}

  Sequence(const Sequence& seq, range_type& r)
   : _wq(work_queue_t<64>::range_type(0, r.size())), _rep(seq._rep)
  {}

  /* pop: owner extraction 
     Return 0 iff the sequence is empty.
     Else return the size of the poped sequence and initialize p 
     to the origin of the poped iteration vector.
  */
  bool pop( range_type& p, work_queue_t<64>::size_type sz_max )
  {
    work_queue_t<64>::range_type r;  
    bool retval = _wq.pop( r, sz_max );
    if (!retval) return false;
    p._size = r.size();
    p._beg.shift( _rep, r.first );
    return r.size();
  }

  /* pop_safe: owner extraction, call workqueue::pop_safe
     Return 0 iff the sequence is empty.
     Else return the size of the poped sequence and initialize p 
     to the origin of the poped iteration vector.
  */
  bool pop_safe( range_type& p, work_queue_t<64>::size_type sz_max )
  {
    work_queue_t<64>::range_type r;  
    bool retval = _wq.pop_safe( r, sz_max );
    if (!retval) return false;
    p._size = r.size();
    p._beg.shift( _rep, r.first );
    return r.size();
  }

  /* steal: thief extraction 
  */
  bool steal( range_type& p, work_queue_t<64>::size_type sz_max )
  {
    work_queue_t<64>::range_type r;  
    bool retval = _wq.steal( r, sz_max );
    if (!retval) return false;
    p._size = r.size();
    p._beg.shift( _rep, r.first );
    return r.size();
  }
  
  /* size 
  */
  size_type size() const 
  {
    return _wq.size();
  }
  
public:
  work_queue_t<64> _wq;
  iterator_type    _rep;
};


} /* rts namespace */

} /* kastl namespace */

#endif
