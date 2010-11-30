/* KAAPI public interface */
/*
** athapascan-1.h
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
#ifndef _ATHAPASCAN_1_H_H
#define _ATHAPASCAN_1_H_H

#include "kaapi.h"
#include "atha_error.h"
#include <vector>
#include <typeinfo>

namespace atha {

/** Only use specialisation of this class
*/
template< int dim, class T>
class Array;

/* fwd declaration */
class Range;
/* fwd declaration */
std::ostream& operator<<( std::ostream& sout, const Range& a );


/** Range.
    Allows to define interval and composition of Ranges is a Range
*/
class Range {
private:
  struct ReallyPrivate {  };
  static const ReallyPrivate CReallyPrivate;

  /** given range */
  Range(ReallyPrivate x, int b, int e) : _begin(b), _end(e) {}

public:
  /** empty range */
  Range() : _begin(0), _end(0) {}
  
  /** given range */
  Range(int b, int e, int s=1) : _begin(b), _end(e)
  {
    kaapi_assert_debug( b >=0 );
    kaapi_assert_debug( e >=0 );
  }
  
  /** Full range */
  static const Range Full; //

  /** size of entry, (size_t)-1U if full range */
  size_t size() const 
  {  
    if (_begin == -1) return (size_t)-1U;
    return (_end-_begin);
  }
  
  /** iterator over values of a range */
  class const_iterator {
  public:
    /** default cstor */
    const_iterator() : _begin(0){}
    
    /** assignment */
    const_iterator& operator=(const const_iterator& it)
    { _begin = it._begin; return *this; }
    
    /** equality */
    bool operator==(const const_iterator& it) const
    { return (_begin == it._begin); }
    
    /** not equal */
    bool operator!=(const const_iterator& it) const
    { return (_begin != it._begin); }
    
    /** pre increment */
    const_iterator& operator++()
    {
      ++_begin;
      return *this; 
    }
    
    /** post increment */
    const_iterator operator++(int)
    {
      const_iterator it = *this;
      ++_begin;
      return it;
    }
    
    /** indirection */
    int operator*()
    { return _begin; }
    
    /** indirection */
    int* operator->()
    { return &_begin; }
    
  private:
    template <int size, class T> friend class Array;
    friend class Range;
    const_iterator( int b) : _begin(b){}
    int _begin;
  };
  
  /** begin */
  const_iterator begin() const { return const_iterator(_begin); }

  /** end */
  const_iterator end() const { return const_iterator(_end); }

  /** Return the i-th value of the range. Return -1 in case of a Full range */
  int operator[](int i) const
  { 
    if (_begin == -1) return -1;
    kaapi_assert_debug( ((unsigned int)i < size()) && (i>=0) );
    return _begin + i;
  }
  
  /** compose Range :  R = A(B)
       A(B)[i] is the i-th value of A o B, ie it is equal to A[B[i]]
  */
  Range operator[](const Range& r) const
  { 
    if (_begin == -1) return r;
    if (r._begin == -1) return *this;
    kaapi_assert_debug( (r.size() <= size()) );

    Range retval;
    retval._begin  = _begin + r._begin;
    retval._end    = _begin + r._end;
    return retval;
  }
  
private:
  template <int size, class T> friend class Array;
  friend std::ostream& operator<<( std::ostream& sout, const Range& a );

  int _begin;
  int _end;
};


/**
     to specialized to have a compact representation
*/
template<class T>
struct ChunkTrait {
  typedef T value_t;
  typedef T chunk_t;
  typedef T* iterator;
  typedef T* const_iterator;
};

/** Specialization for bool: 
*/
template<>
struct ChunkTrait<bool> {
  typedef bool value_t;

  typedef uint64_t chunk_t;
  enum { bitchunk = (8*sizeof(chunk_t))};

  struct iterator {
    iterator() : _data(0), _bit(0);
    
    value_ref_t operator*() 
    { return value_ref_t(_data, _bit); }

    bool operator*() const
    { return *_data & ( 1 << _bit); }

    iterator& operator++() 
    { 
      ++_bit; 
      if (_bit > sizeof(chunk_t)*8) { _bit = 0; ++_data; }
      return *this;
    }

    iterator& operator--() 
    { 
      --_bit; 
      if (_bit <0) { _bit = 0; --_data; }
      return *this;
    }

    iterator operator--(int) 
    {
      iterator it(*this);
      --*this;
      return it; 
    }
    
    value_ref_t operator[](int i)
    { return value_ref_t( _data + i/bitchunk, i % bitchunk); }

    bool operator[](int i) const
    { return *(_data + i/bitchunk) & (1 << (i % bitchunk)); }

  protected:
    chunk_t* _data;
    int      _bit;
  };


  struct const_iterator {
    const_iterator() : _data(0), _bit(0);
    
    bool operator*() const
    { return *_data & ( 1 << _bit); }

    const_iterator& operator++() 
    { 
      ++_bit; 
      if (_bit > sizeof(chunk_t)*8) { _bit = 0; ++_data; }
      return *this;
    }

    const_iterator& operator--() 
    { 
      --_bit; 
      if (_bit <0) { _bit = 0; --_data; }
      return *this;
    }

    const_iterator operator--(int) 
    {
      iterator it(*this);
      --*this;
      return it; 
    }
    
    bool operator[](int i) const
    { return *(_data + i/bitchunk) & (1 << (i % bitchunk)); }

  protected:
    chunk_t* _data;
    int      _bit;
  };
};


/** 
    Storage class
*/
template<class T>
class Storage {
public:
  /* */
  typedef ChunkTrait<T>::chunk_t              chunk_t;
  typedef Chunk_t*                            chunk_iterator;
  typedef const Chunk_t*                      const_chunk_iterator;
  typedef ChunkTrait<T>::chunk_iterator       iterator;
  typedef ChunkTrait<T>::const_chunk_iterator const_iterator;

  
private:
  size_t   _size;
  Chunk_t* _data;
};



/** Array<1,T>
    onedimensional Array of shared objects of type T
*/
template<class T>
class Array<1,T> {
public:
  typedef typename T  value_t;
  typedef Storage<T>  storage_t;
  typedef Array<1,T>  self_t;
  
  typedef T* iterator;
  typedef const T* const_iterator;
  
  // Dflt cstor
  Array<1,T>() 
   : _ni(0), _data(0) 
  {}

  // Recopy cstor
  Array<1,T>(const Array<1,T>& a) 
   : _ni(a._ni), _data(a._data)
  {}

  // Cstor
  Array<1,T>(size_t ni)
   : _ni(ni), _data(0)
  {
    if (_ni !=0)
    {
      _data = new T[ni];
    }
  }
  
  // Assignment
  Array<1,T>& operator=( const Array<1,T>& a )
  {
    _ni   = a._ni;
    _data = a._data;    
    return *this;
  }

  // Assignment to a value
  Array<1,T>& operator=( const T& value )
  {
    for (size_t i=0; i< _ni; i+=_stride)
      _data[i] = value;
    return *this;
  }

  // size of the Array
  size_t size() const 
  { return _ni; }

  // swap
  void swap( Array<1,T>& a )
  {
    std::swap( _ni, a._ni );
    std::swap( _data, a._data );
  }
  
  // Set the (i) element to value
  void set (int i, const T& value) {
    kaapi_assert_debug( (i >= 0) && (i < int(_ni)) );
	  _data[i]=value;
  }

  // Access to the (i) element (to rewrite)
  T& get (int i) const {
    kaapi_assert_debug( (i >= 0) && (i < int(_ni)) );
	  return _data[i];
  }

  // Access to the (i) element (to rewrite)
  T& operator[] (int i) {
    kaapi_assert_debug( (i >= 0) && (i < int(_ni)) );
	  return _data[i];
  }

  // Access to the (i) element (to rewrite)
  const T& operator[] (int i) const {
    kaapi_assert_debug( (i >= 0) && (i < int(_ni)) );
	  return _data[i];
  }

  // Access to the (i) element (to rewrite)
  const Array<1,T> operator[] (const Range& r) const 
  {
    if (r._begin == -1) return *this;
    kaapi_assert_debug( (r._begin >= 0) && (r._end <= _ni) );
	  return Array<1,T>( r._end-r._begin, _data+r._begin );
  }

  // Access to the (i) element (to rewrite)
  Array<1,T> operator[] (const Range& r)  
  {
    if (r._begin == -1) return *this;
    kaapi_assert_debug( (r._begin >= 0) && (r._end <= _ni) );
	  return Array<1,T>( r._end-r._begin, _data+r._begin );
  }

  //
  iterator begin(int i)
  { return _data+i; }

  //
  iterator end(int i)
  { return _data+i; }

  //
  iterator begin()
  { return _data; }

  //
  iterator end()
  { return _data+_ni; }

  //
  const_iterator begin()
  { return _data; }

  //
  const_iterator end()
  { return _data+_ni; }
  
protected:
  // Cstor of a 1-D Array from a pointer 'data' of size 'count' with stride 'stride' access from data
  Array<1,T>(size_t count, T* data)
   : _ni(count), _data(data)
  {}
  
public:
  size_t  _ni;              // number of items
  T*      _data;            // pointer to the data
  
  friend class Array<2,T>;
};



/** bidimensional Array of shared objects of type T
*/
template<class T>
class Array<2,T> {
public:
  // Dflt cstor
  Array<2,T>() 
   : _ni(0), _nj(0), _data(0) {}

  // Cstor
  Array<2,T>(size_t ni, size_t nj)
   : _ni(ni), _nj(nj), _data(0)
  {
    if ((_ni !=0) && (_nj !=0)) 
    {
      _data = new T[ni*nj];
    }
  }
  
  // Recopy cstor
  Array<2,T>(const Array<2,T>& a) 
   : _ni(a._ni), _nj(a._nj), _data(0)
  {
    _data = new T[_ni*_nj];
    std::copy(a._data, a._data+_ni*_nj, _data );
  }

  // Assignment
  Array<2,T>& operator=( const Array<2,T>& a )
  {
    _ni = a._ni;
    _nj = a._nj;
    if (_data !=0) delete [] _data;
    _data = new T[ni*nj];
    std::copy(a._data, a._data+_ni*_nj, _data );
    
    return *this;
  }
  
  // Assignment
  Array<2,T>& operator=( const T& value )
  {
    struct assign {
      assign( const T& value) : _value(value) {}
      void operator( T& datai ) { datai = _value; }
      const T& value;
    };
    
    std::for_each(_data, _data+_ni*_nj, assign(value) );
    return *this;
  }
  
  // size of the Array
  size_t size() const 
  { return _ni*_nj; }

  // size of the Array
  size_t size_i() const 
  { return _ni; }

  // size of the Array
  size_t size_j() const 
  { return _nj; }

protected:
  // row storage of the Array: if different memory organization then
  // col, row should be changed also.
  size_t index(size_t i, size_t j) const
  { 
    KAAPI_ASSERT_DEBUG_M( (i < _ni) && (j<_nj), "[Array::index] index out of bounds");
    return i*_nj+j; 
  }

public:
  // Access to the (i,j) element (to rewrite)
  void set (int i, int j, const T& value) {
    KAAPI_ASSERT_DEBUG_M( (i >= 0) && (i < int(_ni)), "[Array::index] index out of bounds");
    KAAPI_ASSERT_DEBUG_M( (j >= 0) && (j < int(_nj)), "[Array::index] index out of bounds");
	  _data[index(i,j)]=value;
  }

  // Access to the (i,j) element (to rewrite)
  T& get (int i, int j ) const {
    KAAPI_ASSERT_DEBUG_M( (i >= 0) && (i < int(_ni)), "[Array::index] index out of bounds");
    KAAPI_ASSERT_DEBUG_M( (j >= 0) && (j < int(_nj)), "[Array::index] index out of bounds");
	  return _data[index(i,j)];
  }

  // Access to the [i] element after linearization of elements
  const T& operator[] (int i ) const {
    KAAPI_ASSERT_DEBUG_M( (i >= 0) && (i < int(_ni*_nj)), "[Array::index] index out of bounds");
	  return _data[i];
  }

  // Access to the [i] element after linearization of elements
  const T& operator() (int i ) const {
    KAAPI_ASSERT_DEBUG_M( (i >= 0) && (i < _ni*_nj), "[Array::index] index out of bounds");
	  return _data[i];
  }

  // Access to the (i,j) element (to rewrite)
  const T& operator() (int i, int j ) const {
    KAAPI_ASSERT_DEBUG_M( (i >= 0) && (i < int(_ni)), "[Array::index] index out of bounds");
    KAAPI_ASSERT_DEBUG_M( (j >= 0) && (j < int(_nj)), "[Array::index] index out of bounds");
	  return _data[index(i,j)];
  }

  //
  T& operator() (int i, int j ) {
    KAAPI_ASSERT_DEBUG_M( (i >= 0) && (i < int(_ni)), "[Array::index] index out of bounds");
    KAAPI_ASSERT_DEBUG_M( (j >= 0) && (j < int(_nj)), "[Array::index] index out of bounds");
	  return _data[index(i,j)];
  }

  // Return the j-th col
  Array<1,T> col(int j)
  { 
    return Array<1,T>( _ni*_nj,          // size of the Array<1>
                       _nj,              // stride to access to the next item in the returned columns
                       _data+index(0,j) 
                     ); 
  }

  // Return the i-th row
  Array<1,T> row(int i)
  { return Array<1,T>( _nj, 1, _data+index(i,0) ); }

protected:
public:
  size_t  _ni;          // number of rows
  size_t  _nj;          // number of columns
  T*      _data;        // pointer to the data
};




/** Following definitions are required in order to define the way Array are passed to parameters
    of a tasks.
*/
}
#endif

