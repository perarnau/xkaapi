
namespace kblas {

  static int IB = 512;

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
    ka::RPWP<ka::range2d<T> > /* B */
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
    ka::RPWP<ka::range2d<T> > /* C */
  >{};


};

template<typename T>
struct TaskBodyCPU<kblas::TaskTRSM<T> > {
  void operator()( 
    CBLAS_ORDER		    order, 
    CBLAS_SIDE        side,
    CBLAS_UPLO        uplo,
    CBLAS_TRANSPOSE   transA,
    CBLAS_DIAG        diag,
    T                 alpha,
    ka::range2d_r <T> A, 
    ka::range2d_rpwp<T> B
  )
  {
    int k, m, n;
//    int lda, ldan, ldb;
//    int tempkm, tempkn, tempmm, tempnn;

    T zone       = (T) 1.0;
//    T mzone      = (T)-1.0;
    T minvalpha  = (T)-1.0 / alpha;
//    T lalpha;

    const int M = A->dim(0);
    const int N = B->dim(1);
    const int IB = kblas::IB;

#if 1
    fprintf(stdout, "TaskCPU kblas::TRSM n=%d m=%d lda=%d A=%p ldc=%d C=%p\n",
		N, M, A->lda(), (void*)A->ptr(), B->lda(), (void*)B->ptr() ); fflush(stdout);
#endif
   /*  PlasmaRight / PlasmaLower / Plasma[Conj]Trans */
    for (k = 0; k < N; k += IB) {
	ka::rangeindex rk(k, k+IB);
//	tempkn = k == B.nt-1 ? B.n-k*B.nb : B.nb;
//	lda = BLKLDD(A, k);
	for (m = 0; m < M; m += IB) {
	    ka::rangeindex rm(m, m+IB);
//	    tempmm = m == B.mt-1 ? B.m-m*B.mb : B.mb;
//	    ldb = BLKLDD(B, m);
	    ka::Spawn<TaskTRSM<T> >( ka::SetArch(ka::ArchHost) )
	      ( order, side, uplo, transA, diag, alpha, A(rk,rk), B(rk,rm));
#if 0
	    QUARK_CORE_dtrsm(
		plasma->quark, &task_flags,
		side, uplo, trans, diag,
		tempmm, tempkn, A.mb,
		alpha, A(k, k), lda,  /* lda * tempkn */
		       B(m, k), ldb); /* ldb * tempkn */
#endif

	    for (n = k+1; n < N; n+= IB) {
	        ka::rangeindex rn(n, n+IB);
		ka::Spawn<TaskGEMM<T> >( ka::SetArch(ka::ArchHost) )
		  ( order, CblasNoTrans, transA, minvalpha, B(rk,rn), A(rk,rn), zone, B(rn,rm));
#if 0
//		tempnn = n == B.nt-1 ? B.n-n*B.nb : B.nb;
//		ldan = BLKLDD(A, n);
		QUARK_CORE_dgemm(
		    plasma->quark, &task_flags,
		    PlasmaNoTrans, trans,
		    tempmm, tempnn, B.mb, A.mb,
		    minvalpha, B(m, k), ldb,  /* ldb  * tempkn */
			       A(n, k), ldan, /* ldan * tempkn */
		    zone,      B(m, n), ldb); /* ldb  * tempnn */
#endif
	    }
	}
    }
  }
};

template<typename T>
struct TaskBodyCPU<kblas::TaskSYRK<T> > {
  void operator()(
    CBLAS_ORDER		    order, 
    CBLAS_UPLO        uplo,
    CBLAS_TRANSPOSE   trans,
    T                 alpha,
    ka::range2d_r <T> A, 
    T                 beta,
    ka::range2d_rpwp<T> C
  )
  {
    int m, n, k;

    const int cM = C->dim(0);
    const int cN = C->dim(1);
//    const int aM = A->dim(0);
    const int aN = A->dim(1);

    T zbeta;
    T zone = (T)1.0;
    const int IB = kblas::IB;

#if 1
    fprintf(stdout, "TaskCPU kblas::SYRK n=%d m=%d lda=%d A=%p ldc=%d C=%p\n",
		cN, cM, A->lda(), (void*)A->ptr(), C->lda(), (void*)C->ptr() ); fflush(stdout);
#endif
    /*
     *  PlasmaNoTrans / PlasmaLower
     */
    for (n = 0; n < cN; n+= IB) {
      ka::rangeindex rn(n, n+IB);

      for (k = 0; k < aN; k+= IB) {
	  ka::rangeindex rk(k, k+IB);
	  zbeta = k == 0 ? beta : zone;
          ka::Spawn<TaskSYRK<T> >( ka::SetArch(ka::ArchHost) )
	    ( order, uplo, trans, alpha, A(rk,rn), zbeta, C(rn,rn));
#if 0
	  QUARK_CORE_dsyrk(
	      plasma->quark, &task_flags,
	      uplo, trans,
	      tempnn, tempkn, A.mb,
	      alpha, A(n, k), ldan, /* ldan * K */
	      zbeta, C(n, n), ldcn); /* ldc  * N */
#endif
      }
      for (m = n+1; m < cM; m+= IB) {
	  ka::rangeindex rm(m, m+IB);

	  for (k = 0; k < aN; k+= IB) {
	      ka::rangeindex rk(k, k+IB);
	      zbeta = k == 0 ? beta : zone;
		ka::Spawn<TaskGEMM<T> >( ka::SetArch(ka::ArchHost) )
		  ( order, trans, CblasTrans, alpha, A(rk,rm), A(rk,rn), zone, C(rn,rm));
#if 0
	      QUARK_CORE_dgemm(
		  plasma->quark, &task_flags,
		  trans, PlasmaTrans,
		  tempmm, tempnn, tempkn, A.mb,
		  alpha, A(m, k), ldam,  /* ldam * K */
			 A(n, k), ldan,  /* ldan * K */
		  zbeta, C(m, n), ldcm); /* ldc  * N */
#endif
	  }
      }
    }
  }
};

