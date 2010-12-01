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
#ifndef _KAAPI_UTIL_ARRAY1D2D_H_
#define _KAAPI_UTIL_ARRAY1D2D_H_

namespace ka {

/** Only use specialisation of this class
    -1D specialisation
    -2D specialisation
*/
template< int dim, class T>
class array;

class range;
std::ostream& operator<<( std::ostream& sout, const range& a );


/** 1D range.
    Allows to define strided interval and composition of ranges is a range
*/
class range {
public:
  typedef size_t index_t;

  /** empty range */
  range() : _begin(0), _end(0), _stride(1) {}
  
  /** given range */
  range(index_t b, index_t e, index_t s=1) : _begin(b), _end(e), _stride(s) 
  {
    kaapi_assert_debug( b >=0 );
    kaapi_assert_debug( e >=0 );
    kaapi_assert_debug( s >0 );
  }
  
  /** Full range */
  static const range Full; //

  /** size of entry, (index_t)-1U if full range */
  index_t size() const 
  {  
    if (_begin == (index_t)-1) return (index_t)-1U;
    return (_end-_begin+_stride -1)/_stride; 
  }
  
  /** iterator over values of a range */
  class const_iterator {
  public:
    /** default cstor */
    const_iterator() : _begin(0), _end(0), _stride(0) {}
    
    /** assignment */
    const_iterator& operator=(const const_iterator& it)
    { _begin = it._begin; _end = it._end; _stride = it._stride; return *this; }
    
    /** equality */
    bool operator==(const const_iterator& it) const
    { return (_begin == it._begin); }
    
    /** not equal */
    bool operator!=(const const_iterator& it) const
    { return (_begin != it._begin); }
    
    /** pre increment */
    const_iterator& operator++()
    {
      _begin += _stride; 
      if (_begin > _end) _begin = _end;
      return *this; 
    }
    
    /** post increment */
    const_iterator operator++(int)
    {
      const_iterator it = *this;
      ++*this;
      return it;
    }
    
    /** indirection */
    index_t operator*()
    {
      return _begin;
    }
    
    /** indirection */
    index_t* operator->()
    {
      return &_begin;
    }
    
  private:
    template <int dim, class T> friend class array;
    friend class range;
    const_iterator( index_t b, index_t e, index_t s) : _begin(b), _end(e), _stride(s) {}
    index_t _begin;
    index_t _end;
    index_t _stride;
  };
  
  /** begin */
  const_iterator begin() const { return const_iterator(_begin, _end, _stride); }

  /** end */
  const_iterator end() const { return const_iterator(_end, _end, _stride); }

  /** Return the i-th value of the range. Return -1 in case of a Full range */
  index_t operator[](index_t i) const
  { 
    if (_begin == (index_t)-1) return (index_t)-1;
    kaapi_assert_debug( (i < size()) && (i>=0) );
    return _begin + i*_stride;
  }
  
  /** compose range :  R = A(B) return range::Full iff A or B is full range  
       A(B)[i] is the i-th value of A o B, ie it is equal to A[B[i]]
  */
  range operator[](const range& r) const
  { 
    if (_begin == (index_t)-1) return r;
    if (r._begin == (index_t)-1) return *this;
    kaapi_assert_debug( (r.size() <= size()) );

    range retval;
    retval._begin  = (*this)[r._begin];
    retval._end    = (*this)[r._end];
    retval._stride = _stride * r._stride;
    return retval;
  }

  /** compose range :  R = A(B).
      Shift the range to 0 and return the shift index_t value.
      If A(B) is the full range return -1
  */
  index_t compose_shift0(const range& r)
  { 
    index_t retval;
    if (_begin == (index_t)-1) 
    {
      retval = r._begin;
      if (retval == (index_t)-1) 
        return (index_t)-1;
      _begin  = 0;
      _end    = r._end-retval;
      _stride = r._stride;
      return retval;
    }
    if (r._begin == (index_t)-1)
    {
      retval = _begin;
      if (retval == (index_t)-1) 
        return (index_t)-1;
      _begin = 0;
      _end -= retval;
      return retval;
    }
    kaapi_assert_debug( (r.size() <= size()) );
    retval = (*this)[r._begin];
    _end   = (*this)[r._end] - retval;
    _begin = 0;
    _stride *= r._stride;
    return retval;
  }

private:
  template <int dim, class T> friend class array;
  friend std::ostream& operator<<( std::ostream& sout, const range& a );

  index_t _begin;
  index_t _end;
  index_t _stride;
};



/** One dimensional array of shared objects of type T
    Recopy constructor makes an alias to the same 
    memory region.
    Assignement has a 'copy semantic', ie each elements
    of the source array is copied to the destination array. 
    Subarray with stride may be extracted.
    It was to the responsability to the caller to ensure
    that the container of data (either the sub array built
    from some constructor, either the memory region given
    to build the array) has a scope bigger that all of its
    subarray.
*/
template<class T>
class array<1,T> {
public:
  typedef range::index_t index_t;
  typedef T              value_t;
  
  // Dflt cstor
  array<1,T>() 
   : _range(), _data(0) 
  {}

  // Recopy cstor
  array<1,T>(const array<1,T>& a) 
   : _range(a._range), _data(a._data)
  {
  }

  // Cstor
  // Cstor of a 1-D array from a pointer 'data' of size 'count' with stride 'stride' access from data
  array<1,T>(index_t count, T* data, index_t stride = 1)
   : _range(0,count,stride), _data(data)
  {
  }
  
  // Access to the (i) element (to rewrite)
  T& operator[] (index_t i) 
  {
    kaapi_assert_debug( (i >= 0) && (i < size()) );
	  return _data[_range[i]];
  }

  // Access to the (i) element (to rewrite)
  const T& operator[] (index_t i) const 
  {
    kaapi_assert_debug( (i >= 0) && (i < size()));
	  return _data[_range[i]];
  }

  // size of the array
  size_t size() const 
  { return _range.size(); }

  // Assignment
  array<1,T>& operator=( const array<1,T>& a )
  {
    kaapi_assert_debug( size() == a.size() );
    kaapi_assert_debug( _range._begin == 0 );
    kaapi_assert_debug( a._range._begin == 0 );
    size_t sz = _range.size();
    if ((_range._stride ==1) && (a._range._stride ==1))
    {
        for (index_t i=0; i<sz; ++i)
        _data[i] = a._data[i];
    }
    else if ((_range._stride !=1) && (a._range._stride !=1))
    {
      for (index_t i=0, j=0, k=0; i<sz; ++i, j+= _range._stride, k+= a._range._stride)
        _data[j] = a._data[k];
    }
    else if ((_range._stride ==1) && (a._range._stride !=1))
    {
      for (index_t i=0, k=0; i<sz; ++i, k+= a._range._stride)
        _data[i] = a._data[k];
    }
    else if ((_range._stride !=1) && (a._range._stride ==1))
    {
      for (index_t i=0, j=0; i<sz; ++i, j+=_range._stride)
        _data[j] = a._data[i];
    }
    return *this;
  }

  // Assignment to a value
  array<1,T>& operator=( const T& value )
  {
    size_t sz = _range.size();
    if (sz ==0) return *this;
    if (_range._stride == 1)
    {
      for (index_t i=0; i< sz; ++i)
        _data[i] = value;
    }
    else 
    {
      for (index_t i=0, j=0; i< sz; ++i, j+=_range._stride)
        _data[j] = value;
    }
    return *this;
  }

  // swap
  template<class Y>
  void swap( array<1,Y>& a )
  {
    std::swap( _range, a._range );
    std::swap( _data,  a._data );
  }
  
  // Set the (i) element to value
  void set (index_t i, const T& value) 
  {
    kaapi_assert_debug( (i >= 0) && (i < size()) );
	  _data[_range[i]]=value;
  }

  // Access to the (i) element (to rewrite)
  T& get (index_t i) const 
  {
    kaapi_assert_debug( (i >= 0) && (i < size()) );
	  return _data[_range[i]];
  }


  // Access to the (i) element (to rewrite)
  // 0 1 2 3 4 5 6 7 8 9 10 11 12 = A / size = 13
  // 0   2   4   6   8   10    12 = A[range(0,14,2)] = B / size = 7
  //     2           8         12 = B[range[1,9,3)] stride 3 = C / size = 6
  const array<1,T> operator[] (const range& r) const 
  {
    if (r._begin == (index_t)-1) return *this;
    kaapi_assert_debug( r.size() < _range.size() );
    range rthis = _range;
    index_t shift = rthis.compose_shift0(r);
	  return array<1,T>( rthis.size(), _data+shift, rthis._stride );
  }

  // Access to the (i) element (to rewrite)
  array<1,T> operator[] (const range& r)  
  {
    if (r._begin == (index_t)-1) return *this;
    kaapi_assert_debug( r.size() < _range.size() );
    range rthis = _range;
    index_t shift = rthis.compose_shift0(r);
	  return array<1,T>( rthis.size(), _data+shift, rthis._stride );
  }

public:
  range   _range;        // always shift to 0 during composition
  T*      _data;         // pointer to the data
};


#if 0
/** bidimensional array of objects of type T
*/
template<class T>
class array<2,T> {
public:
  // Dflt cstor
  array<2,T>() 
   : _ni(0), _nj(0), _data(0) {}

  // Recopy cstor
  array<2,T>(const array<2,T>& a) 
   : _ni(a._ni), _nj(a._nj), _data(a._data)
  {
  }

  // Cstor
  array<2,T>(index_t ni, index_t nj)
   : _ni(ni), _nj(nj), _data(0)
  {
    if ((_ni !=0) && (_nj !=0)) 
      _data = new T[ni*nj];
  }
  
  // size of the array
  index_t size() const 
  { return _ni*_nj; }

  // Assignment
  array<2,T>& operator=( const array<2,T>& a )
  {
    kaapi_assert_debug(size() != a.size());
    
    if (_refcount !=0) ++*_refcount;
    return *this;
  }
  
  // Assignment
  array<2,T>& operator=( const T& value )
  {
    if ((_refcount !=0) && (*_refcount-- ==0)) delete [] _data;

    for (index_t i=0; i< _ni*_nj; ++i)
        _data[i] = value;

    return *this;
  }
  
  // size of the array
  index_t size_i() const 
  { return _ni; }

  // size of the array
  index_t size_j() const 
  { return _nj; }

  // Access to the (i,j) element (to rewrite)
  void set (int i, int j, const T& value) {
    kaapi_assert_debug( (i >= 0) && (i < int(_ni)), "[array::index] index out of bounds");
    kaapi_assert_debug( (j >= 0) && (j < int(_nj)), "[array::index] index out of bounds");
	  _data[index(i,j)]=value;
  }

  // Access to the (i,j) element (to rewrite)
  const T& get (int i, int j ) const {
    kaapi_assert_debug( (i >= 0) && (i < int(_ni)), "[array::index] index out of bounds");
    kaapi_assert_debug( (j >= 0) && (j < int(_nj)), "[array::index] index out of bounds");
	  return _data[index(i,j)];
  }

  // Access to the (i,j) element (to rewrite)
  T& ref (int i, int j ) {
    kaapi_assert_debug( (i >= 0) && (i < int(_ni)), "[array::index] index out of bounds");
    kaapi_assert_debug( (j >= 0) && (j < int(_nj)), "[array::index] index out of bounds");
	  return _data[index(i,j)];
  }

  // Access to the [i] element after linearization of elements
  const T& operator[] (int i ) const {
    kaapi_assert_debug( (i >= 0) && (i < int(_ni*_nj)), "[array::index] index out of bounds");
	  return _data[i];
  }

  // Access to the [i] element after linearization of elements
  const T& operator() (int i ) const {
    kaapi_assert_debug( (i >= 0) && (i < _ni*_nj), "[array::index] index out of bounds");
	  return _data[i];
  }

  // Access to the (i,j) element (to rewrite)
  const T& operator() (int i, int j ) const {
    kaapi_assert_debug( (i >= 0) && (i < int(_ni)), "[array::index] index out of bounds");
    kaapi_assert_debug( (j >= 0) && (j < int(_nj)), "[array::index] index out of bounds");
	  return _data[index(i,j)];
  }

  //
  T& operator() (int i, int j ) {
    kaapi_assert_debug( (i >= 0) && (i < int(_ni)), "[array::index] index out of bounds");
    kaapi_assert_debug( (j >= 0) && (j < int(_nj)), "[array::index] index out of bounds");
	  return _data[index(i,j)];
  }

  // Return the j-th col
  array<1,T> col(int j)
  { 
    return array<1,T>( _ni*_nj,              // size of the array<1>
                       _nj,              // stride to access to the next item in the returned columns
                       _refcount, 
                       _data+index(0,j) 
                     ); 
  }

  // Return the i-th row
  array<1,T> row(int i)
  { return array<1,T>( _nj, 1, _refcount, _data+index(i,0) ); }

protected:
  // row storage of the array: if different memory organization then
  // col, row should be changed also.
  index_t index(index_t i, index_t j) const
  { 
    kaapi_assert_debug( (i < _ni) && (j<_nj), "[array::index] index out of bounds");
    return i*_nj+j; 
  }
public:
  index_t  _ni;          // number of rows
  index_t  _nj;          // number of columns
  T*      _data;        // pointer to the data
};
template<class T>
std::ostream& operator<<( std::ostream& sout, const array<2,T>& a )
{
  if (a._refcount ==0) return sout << "[]";

  sout << "[";
  for (index_t i=0; i< a._ni; i+=1)
  {
    sout << "[";
    for (index_t j=0; j< a._nj; j+=1)
      sout << a(i,j) << ( j +1 < a._nj ? ", " : "");
    sout << "]" << ( i +1 < a._ni ? ", " : "") << '\n';
  }
  return sout << "]";
}


#endif /** #if 0 array2 */

inline std::ostream& operator<<( std::ostream& sout, const range& a )
{
  return sout << "[" << a._begin << "," << a._stride << "," << a._end << ")";
}

template<class T>
std::ostream& operator<<( std::ostream& sout, const array<1,T>& a )
{
  sout << "[";
  size_t sz = a.size();
  for (typename array<1,T>::index_t i=0; i< sz; ++i)
    sout << a[i] << ( i+1 < sz ? ", " : "");
  return sout << "]";
}


} // namespace
#endif
