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
typedef enum { ColMajor, RowMajor } Storage2DClass;

template< int dim, class T, Storage2DClass s2dc = RowMajor>
class array;

template<int dim, class T, Storage2DClass s2dc>
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
      template<int dim, typename T, Storage2DClass>
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

    template<int dim, typename T,Storage2DClass>
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
class array_rep<1,T,RowMajor> : public base_array {
public:
  typedef typename range::index_t index_t;
  typedef T*                      pointer_t;
  typedef T&                      reference_t;
  typedef const T&                const_reference_t;

  /** */
  array_rep<1,T,RowMajor>() : _data(0) {}

  /** */
  array_rep<1,T,RowMajor>(T* p, index_t sz) : _data(p), _size(sz) {}

  /** */
  void setptr( pointer_t p) 
  { _data =p; }

  /** */
  pointer_t ptr() 
  { return _data; }

  /** */
  pointer_t ptr() const
  { return _data; }

  /** */
  size_t size() const
  { return _size; }

  /** */
  size_t dim(int d) const
  { 
    kaapi_assert_debug( (d == 0) );
    return _size;
  }

  /** */
  reference_t operator[](index_t i)
  { 
    kaapi_assert_debug( (i>=0) && (i < _size) );
    return _data[i]; 
  }
  /** */
  reference_t operator()(index_t i)
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
  const_reference_t operator()(index_t i) const
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

  // Assignment
  array_rep<1,T,RowMajor>& operator=( const array_rep<1,T,RowMajor>& a )
  {
    kaapi_assert_debug( (size() == a.size()) );
    for (index_t i=0; i<_size; ++i)
      _data[i] = a._data[i];
    return *this;
  }

  // Assignment to a value
  array_rep<1,T,RowMajor>& operator=( const T& value )
  {
    for (index_t i=0; i<_size; ++i)
      _data[i] = value;
    return *this;
  }
  
  /* return the view in byte unit */
  kaapi_memory_view_t get_view() const
  { return kaapi_memory_view_make1d(_size, sizeof(T)); }

  /* set a new view in word */
  void set_view( const kaapi_memory_view_t* view )
  { 
    kaapi_assert_debug( view->type == KAAPI_MEMORY_VIEW_1D );
    kaapi_assert_debug( view->wordsize == sizeof(T) );
    _size   = view->size[0];
  }

protected:
  T*      _data;
  index_t _size;
};



// --------------------------------------------------------------------
/** Specialization for 2 dimensional array as for blas submatrix
*/
template<int dim, class T, Storage2DClass s2dc> 
class array_storage {};

template<class T> 
class array_storage<2,T,RowMajor> {
public:
  typedef typename base_array::index_t index_t;
  typedef T*                           pointer_t;
  typedef T&                           reference_t;
  typedef const T&                     const_reference_t;

  array_storage<2,T,RowMajor>() 
   : _data(0), _n(0), _m(0), _lda(0) {}
  
  array_storage<2,T,RowMajor>( T* d, index_t n, index_t m, index_t l )
   : _data(d), _n(n), _m(m), _lda(l) {}

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
  
  // Assignment
  array_storage<2,T,RowMajor>& operator=( const array<2,T,RowMajor>& a )
  {
    kaapi_assert_debug( (_n == a._n) && (_m == a._m) );
    for (index_t i=0; i<_n; ++i)
    {
      T* dest  = _data+i*_lda;
      const T* src =  a._data+i*a._lda;
      for (index_t j=0; j<_m; ++j)
        dest[j] = src[j];
    }
    return *this;
  }

  // Assignment to a value
  array_storage<2,T,RowMajor>& operator=( const T& value )
  {
    for (index_t i=0; i<_n; ++i)
    {
      T* dest  = _data+i*_lda;
      for (index_t j=0; j<_m; ++j)
        dest[j] = value;
    }
    return *this;
  }

protected:
  T*      _data;
  index_t _n;
  index_t _m;
  index_t _lda;
};



template<class T, Storage2DClass s2dc>
class array_rep<2,T,s2dc> : public base_array, public array_storage<2,T,s2dc> {
public:
  typedef typename range::index_t index_t;
  typedef T*                      pointer_t;
  typedef T&                      reference_t;
  typedef const T&                const_reference_t;

  /** */
  array_rep<2,T,s2dc>(T* p, index_t n, index_t m, index_t l) 
   : array_storage<2,T,s2dc>(p,n,m,l)
  {}

  // Range of the array
  size_t dim(int d) const
  { 
    kaapi_assert_debug( (d == 0) || (d ==1) );
    if (d==0) return array_storage<2,T,s2dc>::_n; 
    else return array_storage<2,T,s2dc>::_m;
  }

  /** */
  void setptr( pointer_t p) 
  { array_storage<2,T,s2dc>::_data =p; }

  /** */
  pointer_t ptr() 
  { return array_storage<2,T,s2dc>::_data; }

  /** */
  pointer_t ptr() const
  { return array_storage<2,T,s2dc>::_data; }

  /** */
  index_t lda() const
  { return array_storage<2,T,s2dc>::_lda; }

  /** */
  reference_t operator()(int i, int j)
  { return array_storage<2,T,s2dc>::operator()(i,j); }

  /** */
  reference_t operator()(unsigned int i, unsigned int j)
  { return array_storage<2,T,s2dc>::operator()(i,j); }

  /** */
  reference_t operator()(size_t i, size_t j)
  { return array_storage<2,T,s2dc>::operator()(i,j); }

  /** */
  const_reference_t operator()(int i, int j) const
  { return array_storage<2,T,s2dc>::operator()(i,j); }

  /** */
  const_reference_t operator()(unsigned int i, unsigned int j) const
  { return array_storage<2,T,s2dc>::operator()(i,j); }

  /** */
  const_reference_t operator()(size_t i, size_t j) const
  { return array_storage<2,T,s2dc>::operator()(i,j); }
   
  /* return the view in word */
  kaapi_memory_view_t get_view() const
  { return kaapi_memory_view_make2d(
      array_storage<2,T,s2dc>::_n,
      array_storage<2,T,s2dc>::_m,
      array_storage<2,T,s2dc>::_lda,
      sizeof(T)); 
  }

  /* set a new view in word */
  void set_view( const kaapi_memory_view_t* view )
  { 
    kaapi_assert_debug( view->type == KAAPI_MEMORY_VIEW_2D );
    kaapi_assert_debug( view->wordsize == sizeof(T) );
    array_storage<2,T,s2dc>::_n   = view->size[0];
    array_storage<2,T,s2dc>::_m   = view->size[1];
    array_storage<2,T,s2dc>::_lda = view->lda;
  }

  /**/
  array_rep<2,T,s2dc>& operator=( const T& value )
  { array_storage<2,T,s2dc>::operator=( value ); return *this; }
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
class array<1,T,RowMajor> : public array_rep<1, T, RowMajor> {
public:
  typedef typename base_array::range      range;
  typedef typename base_array::index_t    index_t;
  typedef T                               value_t;
  typedef array_rep<1,T,RowMajor>                  rep_t;
  
  // Dflt cstor
  array<1,T,RowMajor>() 
   : array_rep<1, T, RowMajor>() 
  {}

  // Recopy cstor
  array<1,T,RowMajor>(const array<1,T,RowMajor>& a) 
   : array_rep<1,T,RowMajor>(a)
  {
  }

  // Cstor of a 1-D array from a pointer 'data' of size 'size' 
  array<1,T,RowMajor>(T* data, index_t size)
   : array_rep<1,T,RowMajor>(data, size)
  {
  }
  
  array<1,T,RowMajor>(T* data, const kaapi_memory_view_t* view)
   : array_rep<1,T,RowMajor>(data, view->size[0])
  {}

#if 0
  // Cstor, used to convert type in closure to formal parameter type
  template<class InternalRep>
  explicit array<1,T,RowMajor>(InternalRep& ir )
   : array_rep<1,T,RowMajor>(ir)
  {
  }
#endif

  explicit array<1,T,RowMajor>(array<1,T,RowMajor>& a )
   : array_rep<1,T,RowMajor>(a)
  {
  }
  
  // Access to the (i) element (to rewrite)
  typename rep_t::reference_t operator[] (index_t i) 
  { return array_rep<1,T,RowMajor>::operator[](i); }

  // Access to the (i) element (to rewrite)
  typename rep_t::const_reference_t operator[] (index_t i) const 
  { return array_rep<1,T,RowMajor>::operator[](i); }
  
  // swap
  void swap( array<1,T,RowMajor>& a )
  {
    std::swap( array_rep<1,T,RowMajor>::_data, a._data );
    std::swap( array_rep<1,T,RowMajor>::_size, a._size );
  }

  // Access to the (i) element (to rewrite)
  const array<1,T,RowMajor> operator[] (const range& r) const 
  {
    if (r.isfull()) return *this;
    kaapi_assert_debug( (r.last() <= (index_t)array_rep<1,T,RowMajor>::size()) );
    index_t shift = r.first();
	  return array<1,T,RowMajor>( array_rep<1,T,RowMajor>::ptr()+shift, r.size() );
  }
  const array<1,T,RowMajor> operator() (const range& r) const 
  { return (*this)[r]; }


  // Access to the (i) element (to rewrite)
  array<1,T,RowMajor> operator[] (const range& r)  
  {
    if (r.isfull()) return *this;
    kaapi_assert_debug( (r.last() <= (index_t)array_rep<1,T,RowMajor>::size()) );
    index_t shift = r.first();
	  return array<1,T,RowMajor>( array_rep<1,T,RowMajor>::ptr()+shift, r.size() );
  }

  array<1,T,RowMajor> operator() (const range& r)
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
template<class T, Storage2DClass s2dc>
class array<2,T,s2dc> : public array_rep<2,T,s2dc> {
public:
  typedef typename base_array::range range;
  typedef typename range::index_t    index_t;
  typedef T                          value_t;
  typedef array_rep<2,T,s2dc>        rep_t;

public:  
  // Dflt cstor
  array<2,T,s2dc>() 
   : array_rep<2,T,s2dc>()
  {}

  // Recopy cstor
  array<2,T,s2dc>(const array<2,T,s2dc>& a) 
    : array_rep<2,T,s2dc>(a)
  {}

  // Cstor
  // Cstor of a 2-D array from a pointer 'data' of size 'count1'x 'count2'
  // lda is the leading dimension in order to pass to the next line
  // in each dimension
  array<2,T,s2dc>(T* data, index_t n, index_t m, index_t lda)
   : array_rep<2,T,s2dc>(data, n, m, lda)
  {}

  // Cstor of a 2-D array from a pointer 'data' of size 'count1'x 'count2'
  // lda is the leading dimension in order to pass to the next line
  // in each dimension
  array<2,T,s2dc>(T* data, const kaapi_memory_view_t* view)
   : array_rep<2,T,s2dc>(data, view->size[0], view->size[1], view->lda)
  {}
  
#if 0
  // Cstor, used to convert type in closure to formal parameter type
  template<class InternalRep>
  explicit array<2,T,s2dc>(InternalRep& ir )
    : array_rep<2,T,s2dc>(ir)
  { _range = ir._range; }
#endif

  // make alias
  explicit array<2,T,s2dc>(array<2,T,s2dc>& a )
    : array_rep<2,T,s2dc>(a)
  { }
  

#if 0
  // make alias to sub array
  explicit array<2,T,s2dc>( array<2,T,s2dc>& a, const range& ri, const range& rj )
    : array_rep<2,T,s2dc>( &a(ri[0], rj[0]), ri.size(), rj.size(), a._lda )
  { 
    kaapi_assert_debug( ri.last() <= a.dim(0));
    kaapi_assert_debug( rj.last() <= a.dim(1));
  }
#endif
  
  // swap
  void swap( array<2,T,s2dc>& a )
  {
    std::swap( array_rep<2,T,s2dc>::_data,  a._data );
    std::swap( array_rep<2,T,s2dc>::_n,  a._n );
    std::swap( array_rep<2,T,s2dc>::_m,  a._m );
    std::swap( array_rep<2,T,s2dc>::_lda,  a._lda );
  }
  
  // Subarray extraction
  T& operator() (const index_t& i, const index_t& j)  
  {
    return array_rep<2,T,s2dc>::operator()( i, j );
  }

  // Subarray extraction
  const T& operator() (const index_t& i, const index_t& j) const
  {
    return array_rep<2,T,s2dc>::operator()( i, j );
  }

  // Subarray extraction
  array<2,T,s2dc> operator() (const range& ri, const range& rj)  
  {
    if (ri.isfull() && rj.isfull()) return *this;
    if (ri.isfull()) 
      return array<2,T,s2dc>(&array_rep<2,T,s2dc>::operator()(0, rj[0]), this->dim(0), rj.size(), array_rep<2,T,s2dc>::_lda );
    if (rj.isfull()) 
      return array<2,T,s2dc>(&array_rep<2,T,s2dc>::operator()(ri[0], 0), ri.size(), this->dim(1), array_rep<2,T,s2dc>::_lda );
    return array<2,T,s2dc>(&array_rep<2,T,s2dc>::operator()(ri[0], rj[0]), ri.size(), rj.size(), array_rep<2,T,s2dc>::_lda );
  }

  // Subarray extraction
  array<2,T,s2dc> operator() (const range& ri, const range& rj) const  
  {
    if (ri.isfull() && rj.isfull()) return *this;
    if (ri.isfull()) 
      return array<2,T,s2dc>((T*)&array_rep<2,T,s2dc>::operator()(0, rj[0]), this->dim(0), rj.size(), array_rep<2,T,s2dc>::_lda );
    if (rj.isfull()) 
      return array<2,T,s2dc>((T*)&array_rep<2,T,s2dc>::operator()(ri[0], 0), ri.size(), this->dim(1), array_rep<2,T,s2dc>::_lda );
    return array<2,T,s2dc>((T*)&array_rep<2,T,s2dc>::operator()(ri[0], rj[0]), ri.size(), rj.size(), array_rep<2,T,s2dc>::_lda );
  }

  // Return the i-th col
  array<1,T,RowMajor> col(index_t j)
  {
    return array<1,T,RowMajor>( &(*this)(0,j), array_rep<2,T,s2dc>::_n, 1, array_rep<2,T,s2dc>::_lda );
  }

  // Return the i-th row
  array<1,T,RowMajor> row(index_t i)
  {
    return array<1,T,RowMajor>( &(*this)(i,0), 1, array_rep<2,T,s2dc>::_m, array_rep<2,T,s2dc>::_lda );
  }
};



template<int dim, typename T, Storage2DClass s2dc>
inline std::ostream& operator<<( std::ostream& sout, const typename array_rep<dim,T,s2dc>::range& a )
{
  return sout << "[" << a._begin << "," << a._end << ")";
}

template<class T>
std::ostream& operator<<( std::ostream& sout, const array<1,T,RowMajor>& a )
{
  sout << "[";
  size_t sz = a.size();
  for (typename array<1,T,RowMajor>::index_t i=0; i< sz; ++i)
    sout << a[i] << ( i+1 < sz ? ", " : "");
  return sout << "]";
}


} // namespace
#endif
