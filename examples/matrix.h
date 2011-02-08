/*
**  matrix.h
**  xkaapi
**
**  Created by Gautier Thierry on 19/02/10.
**  Copyright 2010 CR INRIA. All rights reserved.
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
#ifndef XKAAPI_EX_MATRUX_H
#define XKAAPI_EX_MATRUX_H

#include <kaapi++.h>


/** 1D Range.
    Allows to define interval and composition of Ranges is a Range
*/
class Range {
private:
  struct ReallyPrivate {  };
  static const ReallyPrivate CReallyPrivate;

  /** given range */
  Range(ReallyPrivate x, int b, int e) : _begin(b), _end(e)  {}

public:
  /** empty range */
  Range() : _begin(0), _end(0) {}
  
  /** given range */
  Range(int b, int e) : _begin(b), _end(e)
  {
    kaapi_assert_debug_m( b >=0, "bad assertion" );
    kaapi_assert_debug_m( e >=0, "bad assertion" );
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
    int operator*()
    {
      return _begin;
    }
    
    /** indirection */
    int* operator->()
    {
      return &_begin;
    }
    
  private:
    friend class Range;
    const_iterator( int b, int e) : _begin(b), _end(e) {}
    int _begin;
    int _end;
  };
  
  /** begin */
  const_iterator begin() const { return const_iterator(_begin, _end); }

  /** end */
  const_iterator end() const { return const_iterator(_end, _end); }

  /** Return the i-th value of the range. Return -1 in case of a Full range */
  int operator[](int i) const
  { 
    if (_begin == -1) return -1;
    kaapi_assert_debug_m( ((unsigned int)i < size()) && (i>=0), "index out of bound" );
    return _begin + i;
  }
  
  /** compose Range :  R = A(B) returns Range::Full iff A and B is full range,
       returns A if B is Range::Full, return B iff A is Range::Full.
       A(B)[i] is the i-th value of A o B, ie it is equal to A[B[i]]
  */
  Range operator[](const Range& r) const
  { 
    if (r._begin == -1) return *this;
    if (_begin == -1) return r;
    kaapi_assert_debug_m( (r.size() <= size()), "invalid composition" );

    Range retval;
    retval._begin  = (*this)[r._begin];
    retval._end    = (*this)[r._end];
    return retval;
  }
  
private:
  int _begin;
  int _end;
};


/**
 **
 **
 **
 **/
class MatrixDomain;

template<class T>
class Matrix {
public:

  /** Default Cstor 
  */
  Matrix() : _data(0), _nrow(0), _ncol(0), _lda(0), _owner(false) {}
  
  /** Recopy Cstor 
  */
  Matrix(const Matrix& A) : _data(A._data), _nrow(A._nrow), _ncol(A._ncol), _lda(A._lda), _owner(A._owner) {}

  /** Cstor 
  */
  Matrix(size_t nrow, size_t ncol) 
   : _data(0), _nrow(nrow), _ncol(ncol), _lda(ncol), _owner(true) 
  {
    _data = new T[nrow*ncol];
  }
  
  /** Cstor 
  */
  Matrix(T* data, size_t nrow, size_t ncol) 
   : _data(data), _nrow(nrow), _ncol(ncol), _lda(ncol), _owner(false) 
  {
  }

  /** Cstor 
  */
  Matrix(T* data, size_t nrow, size_t ncol, size_t lda) 
   : _data(data), _nrow(nrow), _ncol(ncol), _lda(lda), _owner(false) 
  {
  }
  
  /** Dstor 
  */
  ~Matrix() { if (_owner & (_data !=0)) delete []_data; }
  
  /** Assignment
  */
  Matrix<T>& operator=( const Matrix<T>& A )
  {
    if (_data ==0) { /* only case where to allocate data */
      _data = new T[A._nrow*A._ncol];
      _nrow = A._nrow;
      _ncol = A._ncol;
      _owner = true;
    }
    kaapi_assert( (_nrow == A._nrow) && (_ncol == A._ncol) );

    T* dst = _data;
    const T* src = A._data;
    for (int i=0; i<_nrow; ++i)
    {
      for (int j=0; j<_ncol; ++j)
        dest[j] = src[j];
      dest += _lda;
      src += A._lda;
    }
    return *this;
  }
  
  
  /* getting sub matrix */
  Matrix<T> operator()( const Range& I, const Range& J)
  {
    kaapi_assert_debug( _data !=0 );
    kaapi_assert_debug( I.size() < _nrow );
    kaapi_assert_debug( J.size() < _ncol );

    Matrix<T> retval;
    retval._lda   = _lda;
    retval._data  = _data + J[0] + I[0]*_lda;
    retval._nrow  = I.size();
    retval._ncol  = J.size();
    retval._owner = false;
    return retval;
  }

  /* accessors */
  int nrow() const { return _nrow; }
  int ncol() const { return _ncol; }
  int lda() const { return _lda; }
  const T* data() const { return _data; }
  T* data() { return _data; }


private: /* blas like format */
  friend class MatrixDomain;
  T*      _data;
  int     _nrow;
  int     _ncol;
  int     _lda;
  bool    _owner;  /* true iff the object is the owner of the data */
};


/**
 **
 **
 **
 **/
class MatrixDomain {
public:
  template<class Matrix>
	void addin(Matrix &A, const Matrix &B) const;

  template<class Matrix>
	void mulin(Matrix &A, const Matrix &B) const;

  template<class Matrix>
	void add(Matrix &R, const Matrix &A, const Matrix &B) const;

  template<class Matrix>
	void mul(Matrix &R, const Matrix &A, const Matrix &B) const;

  template<class Matrix>
	void axpy(Matrix &R, const Matrix& A, const Matrix& X, const Matrix& Y) const;

  template<class Matrix>
	void axpyin(Matrix &R, const Matrix& X, const Matrix& Y) const;
};

/* */
template<class Matrix>
inline void MatrixDomain::addin(Matrix &A, const Matrix &B) const
{
  kaapi_assert( (A._nrow == B._nrow) && (A._ncol == B._ncol) );
  T* dst = A._data;
  const T* src = B._data;
  for (int i=0; i<A._nrow; ++i)
  {
    for (int j=0; j<B._ncol; ++j)
      dest[j] += src[j];
    dest += A._lda;
    src += B._lda;
}

/* */
template<class Matrix>
inline void MatrixDomain::mulin(Matrix &A, const Matrix &B) const
{
  kaapi_assert(0);
}

/* */
template<class Matrix>
inline void MatrixDomain::add(Matrix &R, const Matrix &A, const Matrix &B) const
{
  kaapi_assert( (A._nrow == B._nrow) && (A._ncol == B._ncol) );
  kaapi_assert( (R._nrow == A._nrow) && (R._ncol == A._ncol) );
  T* dst = R._data;
  const T* srcA = A._data;
  const T* srcB = B._data;
  for (int i=0; i<R._nrow; ++i)
  {
    for (int j=0; j<R._ncol; ++j)
      dest[j] = srcA[j] + srcB[j];
    dest += R._lda;
    srcA += A._lda;
    srcB += B._lda;
  }
}

/* */
template<class Matrix>
inline void MatrixDomain::mul(Matrix &R, const Matrix &A, const Matrix &B) const
{
  kaapi_assert( (A._ncol == B._nrow) && (A._nrow == R._nrow) && (B._ncol == R._ncol) );
  T* dst = R._data;
  const T* srcA = A._data;
  const T* srcB;
  for (int i=0; i<R._nrow; ++i)
  {
    for (int j=0; j<R._ncol; ++j)
    {
      dest[j] = 0;
      srcB = B._data+j;
      for (int k=0; k<A._nrow; ++k)
      {
        dest[j] += srcA[k] * *srcB;
        srcB += B._lda;
      }
    }
    dest += R._lda;
    srcA += A._lda;
  }
}

/* */
template<class Matrix>
inline void MatrixDomain::axpy(Matrix &R, const Matrix& A, const Matrix& X, const Matrix& Y) const
{
  kaapi_assert( (A._ncol == X._nrow) && (A._nrow == Y._nrow) && (X._ncol == Y._ncol) );
  kaapi_assert( (R._ncol == Y._ncol) && (R._nrow == Y._nrow) );
  T* dst = R._data;
  const T* srcA = A._data;
  const T* srcY = Y._data;
  const T* srcX;
  for (int i=0; i<R._nrow; ++i)
  {
    for (int j=0; j<R._ncol; ++j)
    {
      dest[j] = 0;
      srcB = X._data+j;
      for (int k=0; k<A._nrow; ++k)
      {
        dest[j] += srcA[k] * *srcX;
        srcX += X._lda;
      }
      dest[j] += srcY[j];
    }
    dest += R._lda;
    srcA += A._lda;
    srcY += Y._lda;
  }
}

/* */
template<class Matrix>
inline void MatrixDomain::axpyin(Matrix &R, const Matrix& X, const Matrix& Y) const
{
  kaapi_assert(0);
}



/**
 **
 ** Task declarations
 **
 **/
template<class Matrix>
struct TaskMatrixAddIn : public ka::Task<3>::Signature< MatrixDomain, 
                                                        ka::RW<Matrix>, 
                                                        ka::R<Matrix> > {};

template<class Matrix>
struct TaskMatrixAdd   : public ka::Task<4>::Signature< MatrixDomain, 
                                                        ka::W<Matrix>, 
                                                        ka::R<Matrix>, 
                                                        ka::R<Matrix> > {};

template<class Matrix>
struct TaskMatrixMulIn : public ka::Task<3>::Signature< MatrixDomain, 
                                                        ka::RW<Matrix>, 
                                                        ka::R<Matrix> > {};

template<class Matrix>
struct TaskMatrixMul   : public ka::Task<4>::Signature< MatrixDomain, 
                                                        ka::W<Matrix>, 
                                                        ka::R<Matrix>, 
                                                        ka::R<Matrix> > {};

template<class Matrix>
struct TaskMatrixAxpyin : public ka::Task<4>::Signature< MatrixDomain, 
                                                         ka::RW<Matrix>, 
                                                         ka::R<Matrix>, 
                                                         ka::R<Matrix> > {};


template<class Matrix>
struct TaskMatrixAxpy  : public ka::Task<5>::Signature< MatrixDomain, 
                                                        ka::W<Matrix>, 
                                                        ka::R<Matrix>, 
                                                        ka::R<Matrix>, 
                                                        ka::R<Matrix> > {};


/**
 **
 ** Task definitions : should be in implementation part !
 **
 **/

/* Specialized by default only on CPU */
template<class Matrix>
struct TaskBodyCPU<TaskMatrixAddIn<Matrix> > {
  void operator()( const MatrixDomain& MD, ka::pointer_rw<Matrix> A, ka::pointer_r<Matrix>& B )
  { 
    MD.addin( *A, *B );
  }
};

/* Specialized by default only on CPU */
template<class Matrix>
struct TaskBodyCPU<TaskMatrixAdd<Matrix> > {
  void operator()( const MatrixDomain& MD, ka::pointer_rw<Matrix> R, ka::pointer_r<Matrix>& A, ka::pointer_r<Matrix>& B )
  { 
    MD.add( *R, *A, *B );
  }
};

/* Specialized by default only on CPU */
template<class Matrix>
struct TaskBodyCPU<TaskMatrixMulIn<Matrix> > {
  void operator()( const MatrixDomain& MD, ka::pointer_rw<Matrix> A, ka::pointer_r<Matrix>& B )
  { 
    MD.mulin( *A, *B );
  }
};

/* Specialized by default only on CPU */
template<class Matrix>
struct TaskBodyCPU<TaskMatrixMul<Matrix> > {
  void operator()( const MatrixDomain& MD, ka::pointer_rw<Matrix> R, ka::pointer_r<Matrix>& A, ka::pointer_r<Matrix>& B )
  { 
    MD.mul( *R, *A, *B );
  }
};

/* Specialized by default only on CPU */
template<class Matrix>
struct TaskBodyCPU<TaskMatrixAxpyin<Matrix> > {
  void operator()( const MatrixDomain& MD, ka::pointer_rw<Matrix> R, ka::pointer_r<Matrix>& X, ka::pointer_r<Matrix>& Y )
  { 
    MD.axpyin( *R, *X, *Y );
  }
};

/* Specialized by default only on CPU */
template<class Matrix>
struct TaskBodyCPU<TaskMatrixAxpyin<Matrix> > {
  void operator()( const MatrixDomain& MD, ka::pointer_rw<Matrix> R, ka::pointer_r<Matrix>& A, ka::pointer_r<Matrix>& X, ka::pointer_r<Matrix>& Y )
  { 
    MD.axpy( *R, *A, *X, *Y );
  }
};
#endif

