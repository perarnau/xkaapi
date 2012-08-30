#ifndef MATRIX_H_INCLUDED
# define MATRIX_H_INCLUDED

#include "kaapi++"

// required by some signatures
extern "C" {
#include "cblas.h"
#include "clapack.h"
#include "lapacke.h"
}

#define	    IB	    40

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
struct TaskRecursiveGEMM: public ka::Task<8>::Signature
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

/*
 * LAPACK QR factorization of a real M-by-N matrix A.
 */
template<typename T>
struct TaskGEQRT: public ka::Task<5>::Signature
<
    CBLAS_ORDER,			/* row / col */
    ka::RW<ka::range2d<T> >,	/* A */
    ka::W<ka::range2d<T> >,	/* T */
    ka::W<ka::range1d<T> >,	/* TAU */
    ka::W<ka::range1d<T> >	/* WORK */
>{};

/*  */
template<typename T>
struct TaskORMQR: public ka::Task<7>::Signature
<
    CBLAS_ORDER,			/* row / col */
    CBLAS_SIDE,			/* CBLAS left / right */
    CBLAS_TRANSPOSE,             /* transpose flag */
    ka::R<ka::range2d<T> >,	/* A */
    ka::W<ka::range2d<T> >,	/* T */
    ka::RW<ka::range2d<T> >,	/* C */
    ka::RW<ka::range1d<T> >	/* WORK */
>{};

template<typename T>
struct TaskTSQRT: public ka::Task<6>::Signature
<
    CBLAS_ORDER,		/* row / col */
    ka::RW<ka::range2d<T> >,	/* A1 */
    ka::RW<ka::range2d<T> >,	/* A2 */
    ka::W<ka::range2d<T> >,	/* T */
    ka::W<ka::range1d<T> >,	/* TAU */
    ka::W<ka::range1d<T> >	/* WORK */
>{};

template<typename T>
struct TaskTSMQR: public ka::Task<8>::Signature
<
    CBLAS_ORDER,			/* row / col */
    CBLAS_SIDE,			/* CBLAS left / right */
    CBLAS_TRANSPOSE,             /* transpose flag */
    ka::RW<ka::range2d<T> >,	/* A1 */
    ka::RW<ka::range2d<T> >,	/* A2 */
    ka::R<ka::range2d<T> >,	/* V */
    ka::W<ka::range2d<T> >,	/* T */
    ka::W<ka::range1d<T> >	/* WORK */
>{};


template<typename T>
struct TaskGESSM: public ka::Task<4>::Signature
<
  CBLAS_ORDER,			/* row / col */
  ka::R<ka::range1d <int> >,  /* pivot */
  ka::R<ka::range2d<T> >, /* L NB-by-NB lower trianguler tile */
  ka::RW<ka::range2d<T> > /* A, Updated by the application of L. */
>{};

template<typename T>
struct TaskTSTRF: public ka::Task<7>::Signature
<
  CBLAS_ORDER,			/* row / col */
  int,				/* block size (algo) */
  ka::RW<ka::range2d<T> >,	    /* U */
  ka::RW<ka::range2d<T> >,	    /* A */
  ka::RW<ka::range2d<T> >,	    /* L */
  ka::W<ka::range1d <int> >,		    /* pivot */
  ka::RW<ka::range2d<T> >	    /* WORK */
>{};

template<typename T>
struct TaskSSSSM: public ka::Task<6>::Signature
<
  CBLAS_ORDER,			/* row / col */
  ka::RW<ka::range2d<T> >,	    /* A1 */
  ka::RW<ka::range2d<T> >,	    /* A2 */
  ka::R<ka::range2d<T> >,	    /* L1 */
  ka::R<ka::range2d<T> >,	    /* L2 */
  ka::R<ka::range1d <int> >		    /* pivot */
>{};

// task definitions
# include "matrix_cpu.inl"

#include "matrix_dot.inl"

#if CONFIG_USE_CUDA
# include "matrix_gpu.inl"
//# include "matrix_alpha.inl"
#endif

#endif // MATRIX_H_INCLUDED
