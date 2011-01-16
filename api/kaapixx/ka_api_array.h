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

template<int dim, class T>
class array_rep;


// --------------------------------------------------------------------
/** base array: defined to export range type
*/
class base_array {
public:
  typedef int index_t;

  /** 1D range.
      Allows to define strided interval and composition of ranges is a range
  */
  class range {
  public:
    typedef base_array::index_t index_t;

    /** empty range */
    range() : _begin(0), _end(0) {}
    
    /** given range */
    range( index_t b, index_t e ) : _begin(b), _end(e)
    {
      kaapi_assert_debug( b >=0 );
      kaapi_assert_debug( e >=0 );
    }
    
    /** Full range */
    static const range full; //

    /** size of entry, (index_t)-1U if full range */
    index_t size() const 
    {  
      if (_begin == (index_t)-1) return (index_t)-1U;
      return (_end-_begin);
    }
    
    /** return the length of the range, ie end-begin */
    index_t length() const 
    {  
      return (_end-_begin);
    }
    
    /** first entry, (index_t)-1U if full range */
    index_t first() const 
    { return _begin; }

    /** last entry, (index_t)-1U if full range */
    index_t last() const 
    { return _end; }

    /** return true iff the range is full range */
    bool isfull() const
    { return _begin == -1; }
    
    /** iterator over values of a range */
    class const_iterator {
    public:
      /** default cstor */
      const_iterator() : _begin(0), _end(0) {}
      
      /** assignment */
      const_iterator& operator=(const const_iterator& it)
      { _begin = it._begin; _end = it._end; return *this; }
      
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
      template<int dim, typename T>
      friend class array;
      friend class range;
      const_iterator( index_t b, index_t e) : _begin(b), _end(e) {}
      index_t _begin;
      index_t _end;
    };
    
    /** begin */
    const_iterator begin() const 
    { return const_iterator(_begin, _end); }

    /** end */
    const_iterator end() const {
     return const_iterator(_end, _end); }

    /** Return the i-th value of the range. Return -1 in case of a Full range */
    index_t operator[](index_t i) const
    { 
      if (_begin == (index_t)-1) return (index_t)-1;
      kaapi_assert_debug( (i <= size()) && (i>=0) );
      return _begin + i;
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
      _end   = _begin + r._end - retval;
      _begin = 0;
      return retval;
    }

  private:
    /* specific cstor to build full static data member */
    struct CstorFull { CstorFull() {} };
    static const CstorFull init_full;
    range( CstorFull ) 
     : _begin((index_t)-1), _end((index_t)-1)
    {}

    template<int dim, typename T>
    friend class array;
//    friend std::ostream& operator<<( std::ostream& sout, const typename array<dim,T>::range& a );

    index_t _begin;
    index_t _end;
  };
};

/** Range index: export range in global namespace ka::
*/
typedef base_array::range rangeindex;


// --------------------------------------------------------------------
/**
*/
template<class T>
class array_rep<1,T> : public base_array {
public:
  typedef typename range::index_t index_t;
  typedef T*                      pointer_t;
  typedef T&                      reference_t;
  typedef const T&                const_reference_t;

  /** */
  array_rep<1,T>() : _data(0) {}

  /** */
  array_rep<1,T>(T* p, index_t size) : _data(p), _size(size) {}

  /** */
#if 0  
  array_rep<1,T>& operator=(T* p) 
  { 
    _data = p; 
    return *this;
  }
#endif

  /** */
  void setptr( pointer_t p) 
  { _data =p; }

  /** */
  pointer_t ptr() 
  { return _data; }

  /** */
  pointer_t const ptr() const
  { return _data; }

  /** */
  size_t size() const
  { return _size; }

  /** */
  reference_t operator[](index_t i)
  { 
    kaapi_assert_debug( (i>=0) && (i < _size) );
    return _data[i]; 
  }

  /** */
  const_reference_t operator[](index_t i) const
  { 
    kaapi_assert_debug( (i>=0) && (i < _size) );
    return _data[i]; 
  }

  /** */
  pointer_t operator+(index_t i) const
  { return _data+ i; }
  
  // Set the (i) element to value
  void set (index_t i, reference_t value) 
  {
    kaapi_assert_debug( (i>=0) && (i < _size) );
	  _data[i] = value;
  }

  // Access to the (i) element (to rewrite)
  const_reference_t get (index_t i) const 
  {
    kaapi_assert_debug( (i>=0) && (i < _size) );
	  return _data[i];
  }
  
protected:
  T*      _data;
  index_t _size;
};



// --------------------------------------------------------------------
/** Specialization for 2 dimensional array as for blas submatrix
*/
template<class T>
class array_rep<2,T> : public base_array {
public:
  typedef typename range::index_t index_t;
  typedef T*                      pointer_t;
  typedef T&                      reference_t;
  typedef const T&                const_reference_t;

  /** */
  array_rep<2,T>() : _data(0), _n(0), _m(0), _lda(0) {}

  /** */
  array_rep<2,T>(T* p, index_t n, index_t m, index_t lda) : _data(p), _n(n), _m(m), _lda(lda) {}

  /** */
  pointer_t ptr() 
  { return _data; }

  /** */
  pointer_t const ptr() const
  { return _data; }

  /** */
  index_t lda() const
  { return _lda; }

  /** */
  reference_t operator()(index_t i, index_t j)
  { 
    kaapi_assert_debug( (i>=0) && (i < _n) );
    kaapi_assert_debug( (j>=0) && (j < _m) );
    return _data[i*_lda+j]; 
  }

  /** */
  const_reference_t operator()(index_t i, index_t j) const
  { 
    kaapi_assert_debug( (i>=0) && (i < _n) );
    kaapi_assert_debug( (j>=0) && (j < _m) );
    return _data[i*_lda+j]; 
  }
  
  // Set the (i) element to value
  void set (index_t i, index_t j, reference_t value) 
  {
    kaapi_assert_debug( (i>=0) && (i < _n) );
    kaapi_assert_debug( (j>=0) && (j < _m) );
	  _data[i*_lda+j] = value;
  }

  // Access to the (i) element (to rewrite)
  const_reference_t get (index_t i, index_t j ) const 
  {
    kaapi_assert_debug( (i>=0) && (i < _n) );
    kaapi_assert_debug( (j>=0) && (j < _m) );
	  return _data[i*_lda+j];
  }
  
protected:
  T*      _data;
  index_t _n;
  index_t _m;
  index_t _lda;
};



// --------------------------------------------------------------------
/** One dimensional array of shared objects of type T
    Recopy constructor makes an alias to the same 
    memory region.
    Assignement has a 'copy semantic', ie each elements
    of the source array is copied to the destination array. 
    It was to the responsability to the caller to ensure
    that the container of data (either the sub array built
    from some constructor, either the memory region given
    to build the array) has a scope bigger that all of its
    subarray.
*/
template<class T>
class array<1,T> : public array_rep<1, T> {
public:
  typedef typename base_array::range      range;
  typedef typename base_array::index_t    index_t;
  typedef T                               value_t;
  typedef array_rep<1,T>                  rep_t;
  
  // Dflt cstor
  array<1,T>() 
   : array_rep<1, T>() 
  {}

  // Recopy cstor
  array<1,T>(const array<1,T>& a) 
   : array_rep<1,T>(a)
  {
  }

  // Cstor of a 1-D array from a pointer 'data' of size 'size' 
  array<1,T>(T* data, index_t size)
   : array_rep<1,T>(data, size)
  {
  }
  
  // Cstor, used to convert type in closure to formal parameter type
  template<class InternalRep>
  explicit array<1,T>(InternalRep& ir )
   : array_rep<1,T>(ir)
  {
  }

  explicit array<1,T>(array<1,T>& a )
   : array_rep<1,T>(a)
  {
  }
  
  // Access to the (i) element (to rewrite)
  typename rep_t::reference_t operator[] (index_t i) 
  { return array_rep<1,T>::operator[](i); }

  // Access to the (i) element (to rewrite)
  typename rep_t::const_reference_t operator[] (index_t i) const 
  { return array_rep<1,T>::operator[](i); }
  
  // Assignment
  array<1,T>& operator=( const array<1,T>& a )
  {
    kaapi_assert_debug( (array_rep<1,T>::size() == a.size()) );
    size_t sz = array_rep<1,T>::size();
    for (index_t i=0; i<sz; ++i)
      (*this)[i] = a[i];
    return *this;
  }

  // Assignment to a value
  array<1,T>& operator=( const T& value )
  {
    size_t sz = array_rep<1,T>::size();
    if (sz ==0) return *this;
    for (index_t i=0; i< sz; ++i)
      (*this)[i] = value;
    return *this;
  }

  // swap
  template<class Y>
  void swap( array<1,Y>& a )
  {
    std::swap( (array_rep<1,T>&)(*this),  (array_rep<1,T>&)a );
  }

  
  // Access to the (i) element (to rewrite)
  // 0 1 2 3 4 5 6 7 8 9 10 11 12 = A / size = 13
  // 0   2   4   6   8   10    12 = A[range(0,14,2)] = B / size = 7
  //     2           8         12 = B[range[1,9,3)] stride 3 = C / size = 6
  const array<1,T> operator[] (const range& r) const 
  {
    if (r.isfull()) return *this;
    kaapi_assert_debug( (r.last() <= (index_t)array_rep<1,T>::size()) );
    index_t shift = r.first();
	  return array<1,T>( array_rep<1,T>::ptr()+shift, r.size() );
  }
  const array<1,T> operator() (const range& r) const 
  { return (*this)[r]; }


  // Access to the (i) element (to rewrite)
  array<1,T> operator[] (const range& r)  
  {
    if (r.isfull()) return *this;
    kaapi_assert_debug( (r.last() <= (index_t)array_rep<1,T>::size()) );
    index_t shift = r.first();
	  return array<1,T>( array_rep<1,T>::ptr()+shift, r.size() );
  }
  array<1,T> operator() (const range& r)
  { return (*this)[r]; }
};



/** 2-dimensional array of objects of type T
    Recopy constructor makes an alias to the same 
    memory region.
    Assignement has a 'copy semantic', ie each elements
    of the source array is copied to the destination array. 
    Subarray with stride may be extracted.
    It was to the responsability to the caller to ensure
    that the container of data (either the sub array built
    from some constructor, either the memory region given
    to build the array) has a scope bigger that all of its
    subarrays.
*/
template<class T>
class array<2,T> : public array_rep<2, T> {
public:
  typedef typename array_rep<2, T>::range range;
  typedef typename range::index_t         index_t;
  typedef T                               value_t;
  typedef array_rep<2,T>                  rep_t;

public:  
  // Dflt cstor
  array<2,T>() 
   : array_rep<2, T>()
  {}

  // Recopy cstor
  array<2,T>(const array<2,T>& a) 
    : array_rep<2,T>(a)
  {}

  // Cstor
  // Cstor of a 2-D array from a pointer 'data' of size 'count1'x 'count2'
  // lda is the leading dimension in order to pass to the next line
  // in each dimension
  array<2,T>(T* data, index_t n, index_t m, index_t lda)
   : array_rep<2,T>(data)
  {
    _range[0]=range(0, n, 1);
    _range[1]=range(0, m, 1);
  }
  
  // Cstor, used to convert type in closure to formal parameter type
  template<class InternalRep>
  explicit array<2,T>(InternalRep& ir )
    : array_rep<2,T>(ir)
  { _range = ir._range; }

  // make alias
  explicit array<2,T>(array<2,T>& a )
    : array_rep<2,T>(a)
  { _range = a._range; }
  

  // make alias
  explicit array<2,T>(array<2,T>& a, const range& ri, const range& rj )
    : array_rep<2,T>(a)
  { 
    kaapi_assert_debug( ri.last() < a.dim(0));
    kaapi_assert_debug( rj.last() < a.dim(1));
    _range[0] = ri; 
    _range[1] = rj; 
  }
  
  // Access to the (i,j) element (to rewrite)
  typename rep_t::reference_t operator() (index_t i, index_t j) 
  {
    kaapi_assert_debug( (i >= 0) && (i < dim(0)) );
    kaapi_assert_debug( (j >= 0) && (j < dim(1)) );
	  return array_rep<2,T>::operator()( _range[0][i], _range[1][j] );
  }

  // Access to the (i,j) element (to rewrite)
  typename rep_t::const_reference_t operator() (index_t i, index_t j) const 
  {
    kaapi_assert_debug( (i >= 0) && (i < dim(0)) );
    kaapi_assert_debug( (j >= 0) && (j < dim(1)) );
	  return array_rep<2,T>::operator()( _range[0][i], _range[1][j] );
  }

  // size of the array
  size_t size() const 
  { return _range[0].size()*_range[1].size(); }

  // Range of the array
  size_t dim(int d) const
  { 
    kaapi_assert_debug( (d == 0) || (d ==1) );
    return _range[d].size(); 
  }

  // Range of the array
  range slice(int d) const
  { 
    kaapi_assert_debug( (d == 0) || (d ==1) );
    return _range[d]; 
  }

  // Assignment
  array<2,T>& operator=( const array<2,T>& a )
  {
    kaapi_assert_debug( dim(0) == a.dim(0) );
    kaapi_assert_debug( dim(1) == a.dim(1) );

    for (index_t i=0; i< dim(0); ++i)
      for (index_t j=0; j< dim(1); ++j)
        (*this)(i,j) = a(i,j);
    return *this;
  }

  // Assignment to a value
  array<2,T>& operator=( const T& value )
  {
    /* optimization if stride ==1 */
    for (index_t i=0; i< dim(0); ++i)
      for (index_t j=0; j< dim(1); ++j)
        (*this)(i,j) = value;
    return *this;
  }

  // swap
  template<class Y>
  void swap( array<2,Y>& a )
  {
    std::swap( _range, a._range );
    std::swap( (array_rep<2,T>&)(*this),  (array_rep<2,T>&)a );
  }

  
  // Subarray extraction
  array<2,T> operator() (const range& r1, const range& r2)  
  {
    T* base = &(*this)(r1[0], r2[0]);
    range rnew1 = _range[0];
    rnew1.compose_shift0(r1);
    range rnew2 = _range[1];
    rnew2.compose_shift0(r2);
	  array<2,T> retval( base, rnew1.length(), rnew2.length(), array_rep<2,T>::_lda );
    retval._range[0] = rnew1; 
    retval._range[1] = rnew2;
    return retval;
  }

  // Return the i-th col
  array<1,T> col(index_t j)
  {
    return array<1,T>( &(*this)(0,j), _range[0].length()*array_rep<2,T>::_lda, _range[0].stride()*array_rep<2,T>::_lda );
  }

  // Return the i-th row
  array<1,T> row(index_t i)
  {
    return array<1,T>( &(*this)(i,0), _range[1].length(), _range[1].stride() );
  }

public:
  range         _range[2];        // always shift to 0 during composition
};



template<int dim, typename T>
inline std::ostream& operator<<( std::ostream& sout, const typename array_rep<dim,T>::range& a )
{
  return sout << "[" << a._begin << "," << a._end << ")";
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
