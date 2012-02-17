#ifndef MATRIX_H_INCLUDED
# define MATRIX_H_INCLUDED

#include "kaapi++"

#if CONFIG_USE_FLOAT
typedef float double_type;
#elif CONFIG_USE_DOUBLE
typedef double double_type;
#else
typedef float double_type;
#endif


// required by some signatures
extern "C" {
#include "cblas.h"
#include "clapack.h"
#include "lapacke.h"
}


// task signatures
//template<typename T>
struct TaskPrintMatrix : public ka::Task<2>::Signature
<
  std::string,
  ka::R<ka::range2d<double_type> >
>{};

struct TaskDTRSM_left: public ka::Task<2>::Signature
<
  ka::R<ka::range2d<double_type> >, /* Akk */
  ka::RW<ka::range2d<double_type> > /* Akj */
>{};

struct TaskDTRSM_right: public ka::Task<2>::Signature
<
  ka::R<ka::range2d<double_type> >, /* Akk */
  ka::RW<ka::range2d<double_type> > /* Aik */
>{};

struct TaskDGEMM: public ka::Task<8>::Signature
<
  CBLAS_ORDER,			/* row / col */
  CBLAS_TRANSPOSE,             /* NoTrans/Trans for A */
  CBLAS_TRANSPOSE,             /* NoTrans/Trans for B */
  double_type,                      /* alpha */
  ka::R<ka::range2d<double_type> >, /* Aik   */
  ka::R<ka::range2d<double_type> >, /* Akj   */
  double_type,                      /* beta */
  ka::RW<ka::range2d<double_type> > /* Aij   */
>{};

struct TaskDSYRK: public ka::Task<7>::Signature
<
  CBLAS_ORDER,			/* row / col */
  CBLAS_UPLO,                  /* CBLAS Upper / Lower */
  CBLAS_TRANSPOSE,             /* transpose flag */
  double_type,                      /* alpha */
  ka::R<ka::range2d<double_type> >, /* A */
  double_type,                      /* beta */
  ka::RW<ka::range2d<double_type> > /* C */
>{};

struct TaskDTRSM: public ka::Task<8>::Signature
<
  CBLAS_ORDER,			/* row / col */
  CBLAS_SIDE,                  /* side */
  CBLAS_UPLO,                  /* uplo */
  CBLAS_TRANSPOSE,             /* transA */
  CBLAS_DIAG,                  /* diag */
  double_type,                      /* alpha */
  ka::R<ka::range2d<double_type> >, /* A */
  ka::RW<ka::range2d<double_type> > /* B */
>{};

struct TaskDGETRF: public ka::Task<2>::Signature
<
  ka::RW<ka::range2d<double_type> >, /* A */
  ka::W<ka::range1d <int> >  /* pivot */
>{};

struct TaskDGETRFNoPiv: public ka::Task<2>::Signature
<
  CBLAS_ORDER,			/* row / col */
  ka::RW<ka::range2d<double_type> > /* A */
>{};

struct TaskDPOTRF: public ka::Task<3>::Signature
<
  CBLAS_ORDER,			/* row / col */
  CBLAS_UPLO,                  /* upper / lower */
  ka::RW<ka::range2d<double_type> > /* A */
>{};

struct TaskDPOTRF_cpu: public ka::Task<3>::Signature
<
  CBLAS_ORDER,			/* row / col */
  CBLAS_UPLO,                  /* upper / lower */
  ka::RW<ka::range2d<double_type> > /* A */
>{};

/* LAPACK auxiliary routine 
DLACPY copies all or part of a two-dimensional matrix A to another
matrix B.
 */
struct TaskDLACPY: public ka::Task<4>::Signature
<
  CBLAS_ORDER,			/* row / col */
  CBLAS_UPLO,                  /* upper / lower */
  ka::R<ka::range2d<double_type> >, /* A */
  ka::RW<ka::range2d<double_type> > /* B */
>{};

struct TaskDLARNV: public ka::Task<1>::Signature
<
	ka::W<ka::range2d<double_type> > /* A */
>{};

// task definitions
# include "matrix_cpu.inl"
# include "matrix_gpu.inl"

#endif // MATRIX_H_INCLUDED
