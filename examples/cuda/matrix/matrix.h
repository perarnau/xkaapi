#ifndef MATRIX_H_INCLUDED
# define MATRIX_H_INCLUDED

#include "kaapi++"

// required by some signatures
extern "C" {
#include "cblas.h"
#include "clapack.h"
#include "lapacke.h"

#if defined(CONFIG_USE_PLASMA)
#include "plasma.h"
#include "core_blas.h"
#endif
}

#define	    IB	    40

#if 0
static inline int
convertToOrderLapack( const enum CBLAS_ORDER order ) 
{
    switch(order) {
        case CblasRowMajor:
            return LAPACK_ROW_MAJOR;
        case CblasColMajor:
            return LAPACK_COL_MAJOR;
        default:
            return LAPACK_COL_MAJOR;
    }
}

static inline char
convertToOpLapack( const enum CBLAS_TRANSPOSE trans ) 
{
    switch(trans) {
        case CblasNoTrans:
            return 'n';
        case CblasTrans:
            return 't';
        case CblasConjTrans:
            return 'c';                        
        default:
            return 'n';
    }
}

static inline char
convertToFillModeLapack( const enum CBLAS_UPLO uplo ) 
{
    switch (uplo) {
        case CblasUpper:
            return 'u';
	case CblasLower:
        default:
         return 'l';
    }        
}

static inline char
convertToDiagTypeLapack( const enum CBLAS_DIAG diag ) 
{
    switch (diag) {
	case CblasUnit:
            return 'u';
	case CblasNonUnit:
        default:
         return 'n';
    }        
}

static inline char
convertToSideModeLapack( const enum CBLAS_SIDE side ) 
{
    switch (side) {
	case CblasRight:
            return 'r';
	case CblasLeft:
        default:
         return 'l';
    }        
}
#endif

#if defined(KAAPI_TIMING)

#define	    KAAPI_TIMING_BEGIN() \
    double _t0 = kaapi_get_elapsedtime()

#define	    KAAPI_TIMING_END(task,n)			    \
    do{							    \
	double _t1 = kaapi_get_elapsedtime();		    \
	fprintf(stdout, "%s %d %.10f\n", task, n, _t1-_t0);   \
    }while(0)

#if defined(CONFIG_USE_CUDA)

#define	    KAAPI_TIMING_CUDA_BEGIN(stream)	\
    cudaEvent_t _evt0,_evt1;			\
    do{						\
        cudaEventCreate(&_evt0);		\
        cudaEventCreate(&_evt1);		\
        cudaEventRecord(_evt0,stream);		\
    }while(0)

#define	    KAAPI_TIMING_CUDA_END(stream,task,n)	    \
    do{							    \
	cudaEventRecord(_evt1,stream);			    \
	cudaEventSynchronize(_evt1);			    \
	float _tdelta;					    \
	cudaEventElapsedTime(&_tdelta, _evt0, _evt1);	    \
	fprintf(stdout, "%s %d %.10f\n", task, n, (_tdelta/1e3) );\
    }while(0)

#endif /* CONFIG_USE_CUDA */

#else /* KAAPI_TIMING */

#define	    KAAPI_TIMING_BEGIN()
#define	    KAAPI_TIMING_END(task,n)
#define	    KAAPI_TIMING_CUDA_BEGIN(stream)
#define	    KAAPI_TIMING_CUDA_END(stream,task,n)

#endif /* KAAPI_USE_TIMING */

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
struct TaskTRSM_left: public ka::Task<2>::Signature
<
  ka::R<ka::range2d<T> >, /* Akk */
  ka::RW<ka::range2d<T> > /* Akj */
>{};

template<typename T>
struct TaskTRSM_right: public ka::Task<2>::Signature
<
  ka::R<ka::range2d<T> >, /* Akk */
  ka::RW<ka::range2d<T> > /* Aik */
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
struct TaskParallelGEMM: public ka::Task<8>::Signature
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
  ka::W<ka::range1d <int> >  /* pivot */
>{};

template<typename T>
struct TaskGETRFNoPiv: public ka::Task<2>::Signature
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


struct TaskDPOTRF_cpu: public ka::Task<3>::Signature
<
  CBLAS_ORDER,			/* row / col */
  CBLAS_UPLO,                  /* upper / lower */
  ka::RW<ka::range2d<double> > /* A */
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

/*
 * LAPACK QR factorization of a real M-by-N matrix A.
 */
struct TaskPlasmaDGEQRT: public ka::Task<5>::Signature
<
    CBLAS_ORDER,			/* row / col */
    ka::RW<ka::range2d<double> >,	/* A */
    ka::W<ka::range2d<double> >,	/* T */
    ka::W<ka::range1d<double> >,	/* TAU */
    ka::W<ka::range1d<double> >	/* WORK */
>{};

/*  */
struct TaskPlasmaDORMQR: public ka::Task<7>::Signature
<
    CBLAS_ORDER,			/* row / col */
    CBLAS_SIDE,			/* CBLAS left / right */
    CBLAS_TRANSPOSE,             /* transpose flag */
    ka::R<ka::range2d<double> >,	/* A */
    ka::W<ka::range2d<double> >,	/* T */
    ka::RW<ka::range2d<double> >,	/* C */
    ka::RW<ka::range1d<double> >	/* WORK */
>{};

struct TaskPlasmaDTSQRT: public ka::Task<6>::Signature
<
    CBLAS_ORDER,		/* row / col */
    ka::RW<ka::range2d<double> >,	/* A1 */
    ka::RW<ka::range2d<double> >,	/* A2 */
    ka::W<ka::range2d<double> >,	/* T */
    ka::W<ka::range1d<double> >,	/* TAU */
    ka::W<ka::range1d<double> >	/* WORK */
>{};

struct TaskPlasmaDTSMQR: public ka::Task<8>::Signature
<
    CBLAS_ORDER,			/* row / col */
    CBLAS_SIDE,			/* CBLAS left / right */
    CBLAS_TRANSPOSE,             /* transpose flag */
    ka::RW<ka::range2d<double> >,	/* A1 */
    ka::RW<ka::range2d<double> >,	/* A2 */
    ka::R<ka::range2d<double> >,	/* V */
    ka::W<ka::range2d<double> >,	/* T */
    ka::W<ka::range1d<double> >	/* WORK */
>{};


#if 1 /* LU new tasks */

struct TaskPlasmaDGESSM: public ka::Task<4>::Signature
<
  CBLAS_ORDER,			/* row / col */
  ka::R<ka::range1d <int> >,  /* pivot */
  ka::R<ka::range2d<double> >, /* L NB-by-NB lower trianguler tile */
  ka::RW<ka::range2d<double> > /* A, Updated by the application of L. */
>{};

struct TaskPlasmaDTSTRF: public ka::Task<7>::Signature
<
  CBLAS_ORDER,			/* row / col */
  int,				/* block size (algo) */
  ka::RW<ka::range2d<double> >,	    /* U */
  ka::RW<ka::range2d<double> >,	    /* A */
  ka::RW<ka::range2d<double> >,	    /* L */
  ka::W<ka::range1d <int> >,		    /* pivot */
  ka::RW<ka::range2d<double> >	    /* WORK */
>{};

struct TaskPlasmaDSSSSM: public ka::Task<6>::Signature
<
  CBLAS_ORDER,			/* row / col */
  ka::RW<ka::range2d<double> >,	    /* A1 */
  ka::RW<ka::range2d<double> >,	    /* A2 */
  ka::R<ka::range2d<double> >,	    /* L1 */
  ka::R<ka::range2d<double> >,	    /* L2 */
  ka::R<ka::range1d <int> >		    /* pivot */
>{};

#endif

// task definitions
# include "matrix_cpu.inl"

#include "matrix_dot.inl"

#if CONFIG_USE_CUDA
# include "matrix_gpu.inl"
//# include "matrix_alpha.inl"
#endif

#endif // MATRIX_H_INCLUDED
