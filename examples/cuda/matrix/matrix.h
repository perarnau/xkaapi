#ifndef MATRIX_H_INCLUDED
# define MATRIX_H_INCLUDED

#include "kaapi++"

// required by some signatures
extern "C" {
#include "cblas.h"
#include "clapack.h"
#include "lapacke.h"
}

#define	    CONFIG_IB_CPU	    64 /* from PLASMA (control/auxiliary.c) */
#define	    CONFIG_IB_GPU	    128  /* from MAGMA (control/get_nb_fermi.cpp) */

// task signatures
template<typename T>
struct TaskPrintMatrix : public ka::Task<2>::Signature
<
  std::string,
  ka::R<ka::range2d<T> >
>{};

struct TaskPrintMatrixInt : public ka::Task<2>::Signature
<
  std::string,
  ka::R<ka::range2d<int> >
>{};

template<typename T>
struct TaskGEMM: public ka::Task<8>::Signature
<
  CBLAS_ORDER,			      /* row / col */
  CBLAS_TRANSPOSE,        /* NoTrans/Trans for A */
  CBLAS_TRANSPOSE,        /* NoTrans/Trans for B */
  T,                      /* alpha */
  ka::R<ka::range2d<T> >, /* Aik   */
  ka::R<ka::range2d<T> >, /* Akj   */
  T,                      /* beta */
  ka::RW<ka::range2d<T> > /* Aij   */
>{};

template<typename T>
struct TaskGEMM2: public ka::Task<9>::Signature
<
CBLAS_ORDER,			      /* row / col */
CBLAS_TRANSPOSE,        /* NoTrans/Trans for A */
CBLAS_TRANSPOSE,        /* NoTrans/Trans for B */
T,                      /* alpha */
ka::R<ka::range2d<T> >, /* Aik   */
ka::R<ka::range2d<T> >, /* Akj   */
T,                      /* beta */
ka::RW<ka::range2d<T> >, /* Aij   */
ka::W<int> /* fake */
>{};

template<typename T>
struct TaskSYRK: public ka::Task<7>::Signature
<
  CBLAS_ORDER,			      /* row / col */
  CBLAS_UPLO,             /* CBLAS Upper / Lower */
  CBLAS_TRANSPOSE,        /* transpose flag */
  T,                      /* alpha */
  ka::R<ka::range2d<T> >, /* A */
  T,                      /* beta */
  ka::RW<ka::range2d<T> > /* C */
>{};


template<typename T>
struct TaskTRSM: public ka::Task<8>::Signature
<
  CBLAS_ORDER,            /* row / col */
  CBLAS_SIDE,             /* side */
  CBLAS_UPLO,             /* uplo */
  CBLAS_TRANSPOSE,        /* transA */
  CBLAS_DIAG,             /* diag */
  T,                      /* alpha */
  ka::R<ka::range2d<T> >, /* A */
  ka::RW<ka::range2d<T> > /* B */
>{};


template<typename T>
struct TaskGETRF: public ka::Task<3>::Signature
<
  CBLAS_ORDER,               /* row / col */
  ka::RW<ka::range2d<T> >,   /* A */
  ka::W<ka::range1d<int> >  /* pivot */
>{};

template<typename T>
struct TaskGETF2: public ka::Task<4>::Signature
<
CBLAS_ORDER,               /* row / col */
ka::RW<ka::range2d<T> >,   /* A */
ka::W<ka::range1d<int> >,  /* pivot */
ka::W<int>                  /* info */
>{};

template<typename T>
struct TaskGETRFNoPiv: public ka::Task<2>::Signature
<
  CBLAS_ORDER,            /* row / col */
  ka::RW<ka::range2d<T> > /* A */
>{};

template<typename T>
struct TaskGETF2NoPiv: public ka::Task<2>::Signature
<
  CBLAS_ORDER,            /* row / col */
  ka::RW<ka::range2d<T> > /* A */
>{};

template<typename T>
struct TaskPOTRF: public ka::Task<3>::Signature
<
  CBLAS_ORDER,			      /* row / col */
  CBLAS_UPLO,             /* upper / lower */
  ka::RW<ka::range2d<T> > /* A */
>{};

/* LAPACK auxiliary routine 
DLACPY copies all or part of a two-dimensional matrix A to another
matrix B.
 */
template<typename T>
struct TaskLACPY: public ka::Task<4>::Signature
<
  CBLAS_ORDER,			       /* row / col */
  CBLAS_UPLO,              /* upper / lower */
  ka::RW<ka::range2d<T> >, /* A */
  ka::RW<ka::range2d<T> >  /* B */
>{};

template<typename T>
struct TaskLARNV: public ka::Task<1>::Signature
<
	ka::W<ka::range2d<T> > /* A */
>{};

template<typename T>
struct TaskLASWP: public ka::Task<6>::Signature
<
  CBLAS_ORDER,			       /* row / col */
  ka::RW<ka::range2d<T> >,		/* A */
  int,				/* K1 */
  int,				/* K2 */
  ka::R<ka::range1d<int> > ,		/* IPIV */
  int				/* INC */
>{};

template<typename T>
struct TaskLASWP2: public ka::Task<7>::Signature
<
CBLAS_ORDER,			       /* row / col */
ka::RW<ka::range2d<T> >,		/* A */
int,				/* K1 */
int,				/* K2 */
ka::R<ka::range1d<int> > ,		/* IPIV */
int,				/* INC */
ka::R<int> /* fake */
>{};

#include "timing.inl"

// task definitions
# include "matrix_cpu.inl"

#include "matrix_dot.inl"

#if CONFIG_USE_CUDA
# include "matrix_gpu.inl"
#include "magmablas/magmablas.h"
//# include "matrix_alpha.inl"
#endif

#include "kblas.inl"
#if defined(CONFIG_USE_PLASMA)
#include "kplasma.inl"
#endif

#endif // MATRIX_H_INCLUDED
