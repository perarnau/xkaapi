
/*
 * KPLASMA - package of PLASMA kernels on CPU or GPU (CUDA)
 */

template<class T>
struct KPLASMA {
  typedef T value_type;
  
  static int gessm_gpu(cudaStream_t stream, int m, int n, int k, int ib,
		   int *ipiv,
		   value_type *l, int ldl,
		   value_type *a, int lda);

  static int ssssm_gpu(cudaStream_t stream, int m1, int n1, int m2, int n2, int k, int ib,
		   value_type *a1, int lda1,
		   value_type *a2, int lda2,
		   value_type *l1, int ldl1,
		   value_type *l2, int ldl2,
		   int *ipiv);
};

template<>
struct KPLASMA<double> {
  typedef double value_type;

  static int gessm_gpu(cudaStream_t stream, int m, int n, int k, int ib,
		   int *ipiv,
		   value_type *l, int ldl,
		   value_type *a, int lda)
  {
    static value_type zone  =  1.0;
    static value_type mzone = -1.0;
    cublasStatus_t status;

    int i, sb;
    int tmp, tmp2;

    for(i = 0; i < k; i += ib) {
        sb = std::min(ib, k-i);
        tmp  = i+1;
        tmp2 = i+sb;

	magmablas<value_type>::laswp(stream, n, a, lda, i+1, i+sb, ipiv, 1);

        /*
         * Compute block row of U.
         */
	status = CUBLAS<value_type>::trsm (
	   convertToSideMode(CblasLeft),
	   convertToFillMode(CblasLower),
	   convertToOp(CblasNoTrans),
	   convertToDiagType(CblasUnit),
	   sb, n,
	   &zone,
           &l[ldl*i+i], ldl,
           &a[i], lda
	  );
	if (status != CUBLAS_STATUS_SUCCESS)
	  printf("TaskGESSM::cublasDtrsm() == %d\n", status);

        if (i+sb < m) {
	    /*
	    * Update trailing submatrix.
	    */
	    status = CUBLAS<value_type>::gemm(
		convertToOp(CblasNoTrans),
		convertToOp(CblasNoTrans),
		m-(i+sb), n, sb,
		&mzone,
		&l[ldl*i+(i+sb)], ldl,
		&a[i], lda,
		&zone, &a[i+sb], lda
	    );
	    if (status != CUBLAS_STATUS_SUCCESS)
	      printf("TaskGESSM::cublasDgemm() == %d\n", status);
	}
    }

    return 0;
  }

  static int ssssm_gpu(cudaStream_t stream, int m1, int n1, int m2, int n2, int k, int ib,
		   value_type *a1, int lda1,
		   value_type *a2, int lda2,
		   value_type *l1, int ldl1,
		   value_type *l2, int ldl2,
		   int *ipiv)
  {
    const value_type zone  = 1.0;
    const value_type mzone =-1.0;
    cublasStatus_t status;

    int i, ii, sb;
    int im, ip;
    ip = 0;

    for(ii = 0; ii < k; ii += ib) {
        sb = std::min(k-ii, ib);

        for(i = 0; i < sb; i++) {
            im = ipiv[ip]-1;

            if (im != (ii+i)) {
                im = im - m1;
		status = CUBLAS<value_type>::swap
		  (
		   kaapi_cuda_cublas_handle(),
		   n1,
		   &a1[ii+i], lda1,
		   &a2[im], lda2
		  );
            }
            ip = ip + 1;
        }

	status = CUBLAS<value_type>::trsm
	  (
	   kaapi_cuda_cublas_handle(),
	   convertToSideMode(CblasLeft),
	   convertToFillMode(CblasLower),
	   convertToOp(CblasNoTrans),
	   convertToDiagType(CblasUnit),
	   sb, n1, &zone,
           &l1[ldl1*ii], ldl1,
           &a1[ii], lda1
	  );
	if (status != CUBLAS_STATUS_SUCCESS)
	  printf("TaskTaskSSSSM::trsm() == %d\n", status);

	status = CUBLAS<value_type>::gemm
	(
	    kaapi_cuda_cublas_handle(),
	    convertToOp(CblasNoTrans),
	    convertToOp(CblasNoTrans),
	    m2, n2, sb,
	    &mzone,
	    &l2[ldl2*ii], ldl2,
            &a1[ii], lda1,
            &zone, a2, lda2
	);
	if (status != CUBLAS_STATUS_SUCCESS)
	  printf("TaskSSSSM::gemm() == %d\n", status);
    }
    return 0;
  }
};

namespace kplasma {
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
};

template<typename T>
struct TaskBodyCPU<kplasma::TaskGESSM<T> > {
  void operator()( 
    CBLAS_ORDER order, 
    ka::range1d_r<int> piv,
    ka::range2d_r<T> L, 
    ka::range2d_rw<T> A
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int k = A->dim(1); /* TODO check */
    const int lda = A->lda();
    const int ldl = L->lda();
    T* const a = A->ptr();
    T* const l = (T*)L->ptr();
    int* const ipiv = (int*)piv->ptr();

    const int ib = CONFIG_IB; // from PLASMA
    PLASMA<T>::gessm(m, n, k, ib, ipiv, l, ldl, a, lda);
#else
    /* TODO */
#endif
  }
};

template<typename T>
struct TaskBodyCPU<kplasma::TaskTSTRF<T> > {
  void operator()( 
    CBLAS_ORDER order, 
    int nb,
    ka::range2d_rw<T> U, 
    ka::range2d_rw<T> A,
    ka::range2d_rw<T> L,
    ka::range1d_w<int> piv,
    ka::range2d_rw<T> WORK
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int lda = A->lda();
    const int ldl = L->lda();
    const int ldu = U->lda();
    const int ldw = WORK->lda();
    T* const a = A->ptr();
    T* const l = L->ptr();
    T* const u = U->ptr();
    T* const work = WORK->ptr();
    int* const ipiv = piv->ptr();

#if 0
    fprintf( stdout, "TaskDTaskTSTRF L(%lu,%lu,%d)\n", L->dim(0), L->dim(1), L->lda() );
    fflush(stdout);
#endif
    const int ib = CONFIG_IB; // from PLASMA
    int info;
    PLASMA<T>::tstrf(m, n, ib, nb, u, ldu, a, lda, l, ldl, ipiv, work, ldw, &info);
#else
    /* TODO */
#endif
  }
};

template<typename T>
struct TaskBodyCPU<kplasma::TaskSSSSM<T> > {
  void operator()( 
    CBLAS_ORDER order, 
    ka::range2d_rw<T> A1, 
    ka::range2d_rw<T> A2,
    ka::range2d_r<T> L1,
    ka::range2d_r<T> L2,
    ka::range1d_r<int> piv
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m1 = A1->dim(0); 
    const int n1 = A1->dim(1);
    const int m2 = A2->dim(0); 
    const int n2 = A2->dim(1);
    const int k = L1->dim(0);
    const int lda1 = A1->lda();
    const int lda2 = A2->lda();
    const int ldl1 = L1->lda();
    const int ldl2 = L2->lda();
    T* const a1 = A1->ptr();
    T* const a2 = A2->ptr();
    T* const l1 = (T*)L1->ptr();
    T* const l2 = (T*)L2->ptr();
    int* const ipiv = (int*)piv->ptr();

#if 0
    fprintf( stdout, "TaskDTaskSSSSM L(%lu,%lu), k=%d\n",
	    L1->dim(0), L1->dim(1), k );
#endif
    const int ib = CONFIG_IB; // from PLASMA
    PLASMA<T>::ssssm(m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, l1, ldl1, l2, ldl2, ipiv);
#else
    /* TODO */
#endif
  }
};

#if defined(CONFIG_USE_CUDA)
template<typename T>
struct TaskBodyGPU<kplasma::TaskGESSM<T> > {
  void operator()( 
    ka::gpuStream stream,
    CBLAS_ORDER order, 
    ka::range1d_r<int> piv,
    ka::range2d_r<T> L, 
    ka::range2d_rw<T> A
  )
  {
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int k = A->dim(1); /* TODO check */
    const int lda = A->lda();
    const int ldl = L->lda();
    T* const a = A->ptr();
    T* const l = (T*)L->ptr();
    int* const ipiv = (int*)piv->ptr();
//    const int ib = CONFIG_IB; // from PLASMA
    const int ib = k; // from PLASMA

    /* TODO: kaapi call */
    int* hipiv = (int*)calloc( k, sizeof(int) );
    cudaMemcpy( hipiv, ipiv, k * sizeof(int), 
	    cudaMemcpyDeviceToHost );

#if 0
    fprintf( stdout, "TaskGPU TaskGESSM A(%lu,%lu) a=%p lda=%d  L(%lu,%lu) l=%p ldl=%d\n",
	    A->dim(0), A->dim(1), (void*)a, A->lda(),
	    L->dim(0), L->dim(1), (void*)l, L->lda()
	   );
    fflush(stdout);
#endif
    KPLASMA<T>::gessm_gpu((cudaStream_t)stream.stream, m, n, k, ib, hipiv, l, ldl, a, lda);
    free(hipiv);
  }
};

template<typename T>
struct TaskBodyGPU<kplasma::TaskSSSSM<T> > {
  void operator()( 
    ka::gpuStream stream,
    CBLAS_ORDER order, 
    ka::range2d_rw<T> A1, 
    ka::range2d_rw<T> A2,
    ka::range2d_r<T> L1,
    ka::range2d_r<T> L2,
    ka::range1d_r<int> piv
  )
  {
    int m1 = A1->dim(0); 
    int n1 = A1->dim(1);
    int m2 = A2->dim(0); 
    int n2 = A2->dim(1);
    int k = L1->dim(0);
    int lda1 = A1->lda();
    int lda2 = A2->lda();
    int ldl1 = L1->lda();
    int ldl2 = L2->lda();
    T* const a1 = A1->ptr();
    T* const a2 = A2->ptr();
    T* const l1 = (T*)L1->ptr();
    T* const l2 = (T*)L2->ptr();
    int* const ipiv = (int*)piv->ptr();
//    const int ib = CONFIG_IB; // from PLASMA
    const int ib = k; // from PLASMA

#if 0
    fprintf( stdout, "TaskGPU DTaskSSSSM A1(%lu,%lu) a1=%p lda1=%d "
	   "A2(%lu,%lu) a2=%p lda2=%d "
	   "L1(%lu,%lu) l1=%p ldl1=%d "
	   "L2(%lu,%lu) l2=%p ldl2=%d ipiv=%p\n",
	   A1->dim(0), A1->dim(1), (void*)a1, lda1,
	   A2->dim(0), A2->dim(1), (void*)a2, lda2,
	   L1->dim(0), L1->dim(1), (void*)l1, ldl1,
	   L2->dim(0), L2->dim(1), (void*)l2, ldl2,
	   ipiv
	);
    fflush(stdout);
#endif
    /* TODO KAAPI call */
    int* hipiv = (int*)calloc( piv->size(), sizeof(int) );
    cudaMemcpy( hipiv, ipiv, piv->size() * sizeof(int), 
	    cudaMemcpyDeviceToHost );

    KPLASMA<T>::ssssm_gpu((cudaStream_t)stream.stream, m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, l1, ldl1, l2, ldl2, hipiv);
    free(hipiv);
  }
};

#endif /* CONFIG_USE_CUDA */

#if 0

template<typename T>
struct TaskBodyGPU<kplasma::TaskTSTRF<T> > {
  void operator()( 
    ka::gpuStream stream,
    CBLAS_ORDER order, 
    int nb,
    ka::range2d_rw<T> U, 
    ka::range2d_rw<T> A,
    ka::range2d_rw<T> L,
    ka::range1d_w<int> piv,
    ka::range2d_rw<T> WORK
  )
  {
#if defined(CONFIG_USE_MAGMA)
    int m = A->dim(0); 
    int n = A->dim(1);
    int lda = A->lda();
    int ldl = L->lda();
    int ldu = U->lda();
    int ldw = WORK->lda();
    T* const a = A->ptr();
    T* const l = L->ptr();
    T* const u = U->ptr();
    T* const work = WORK->ptr();
    int* const ipiv = piv->ptr();
    const int ib = CONFIG_IB; // from PLASMA

#if 1
    fprintf( stdout, "TaskGPU DTaskTSTRF A(%lu,%lu) a=%p lda=%d "
	    "L(%lu,%lu) l=%p ldl=%d "
	    "U(%lu,%lu) u=%p ldu=%d ipiv=%p\n",
	    A->dim(0), A->dim(1), (void*)a, lda,
	    L->dim(0), L->dim(1), (void*)l, ldl,
	    U->dim(0), U->dim(1), (void*)u, ldu,
	    (void*)ipiv 
    );
    fflush(stdout);
#endif

    T* ha = (T*)malloc( A->dim(0)*A->dim(1) * sizeof(T) );
    T* hl = (T*)malloc( L->dim(0)*L->dim(1) * sizeof(T) );
    T* hu = (T*)malloc( U->dim(0)*U->dim(1) * sizeof(T) );
    T* hwork = (T*)malloc( WORK->dim(0)*WORK->dim(1) * sizeof(T) );
    cublasGetMatrix( m, n, sizeof(T), u, ldu, hu, ldu );
    cublasGetMatrix( m, ib, sizeof(T), a, lda, ha, lda );
    memset( hl, 0, L->dim(0)*L->dim(1)*sizeof(T) );
    int* hipiv = (int*)calloc( piv->size(), sizeof(int) );
    cudaMemcpy( hipiv, ipiv, piv->size() * sizeof(int), 
	    cudaMemcpyDeviceToHost );

    int info =
      MAGMA<T>::tstrf( 'f', m, n, ib, L->dim(1), hu, ldu, u, ldu, ha, lda, a, lda, hl, ldl, l, ldl, hipiv, hwork, ldw, work, lda );
    if(info){
      fprintf( stdout, "TaskTaskTSTRF ERROR (%d)\n", info );
      fflush(stdout);
    }
    /* TODO */
    cudaMemcpyAsync( hipiv, ipiv, piv->size() * sizeof(int), 
	    cudaMemcpyHostToDevice,
	    (cudaStream_t)stream.stream
     );
#else
    /* TODO */
#endif
  }
};

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
struct TaskBodyCPU<TaskORMQR<T> > {
  void operator()( 
	CBLAS_ORDER		order,
	CBLAS_SIDE		side,
	CBLAS_TRANSPOSE		trans,
	ka::range2d_r<T>  A,
	ka::range2d_w<T>  _T,
	ka::range2d_rw<T> C,
	ka::range1d_rw<T> WORK
  )
  {
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int lda = A->lda();
    const int ldt = _T->lda();
    T* const a = A->ptr();
    T* const t = _T->ptr();
    T* const work = WORK->ptr();
    const int ib = CONFIG_IB; // PLASMA(control/auxiliary.c)

#if defined(CONFIG_USE_PLASMA)
    const int k = std::min(m, n);
    const int ldc = C->lda();
    const int ldwork = WORK->size();
    T* const c = C->ptr();
    PLASMA<T>::ormqr(side, trans, m, n, k, ib, a, lda, t, ldt, c, ldc, work, ldwork);
#endif
  }
};

template<typename T>
struct TaskBodyCPU<TaskTSQRT<T> > {
  void operator()( 
	CBLAS_ORDER order,
	ka::range2d_rw<T> A1,
	ka::range2d_rw<T> A2,
	ka::range2d_w<T>  _T,
	ka::range1d_w<T>  TAU,
	ka::range1d_w<T>  WORK
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m = A2->dim(1); 
    const int n = A1->dim(0);
    const int lda1 = A1->lda();
    const int lda2 = A2->lda();
    const int ldt = _T->lda();
    T* const a1 = A1->ptr();
    T* const a2 = A2->ptr();
    T* const t = _T->ptr();
    T* const work = WORK->ptr();
    T* const tau = TAU->ptr();
    const int ib = CONFIG_IB; // PLASMA(control/auxiliary.c)

    PLASMA<T>::tsqrt( m, n, ib, a1, lda1, a2, lda2, t, ldt, tau, work );
#endif
  }
};

template<typename T>
struct TaskBodyCPU<TaskTSMQR<T> > {
  void operator()( 
	CBLAS_ORDER		order,
	CBLAS_SIDE		side,
	CBLAS_TRANSPOSE		trans,
	ka::range2d_rw<T> A1,
	ka::range2d_rw<T> A2,
	ka::range2d_r<T>  V,
	ka::range2d_w<T>  _T,
	ka::range1d_w<T>  WORK
  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m1 = A1->dim(0); 
    const int n1 = A1->dim(1);
    const int m2 = A2->dim(0); 
    const int n2 = A2->dim(1);
    const int lda1 = A1->lda();
    const int lda2 = A2->lda();
    const int ldv = V->lda();
    const int ldt = _T->lda();
    T* const a1 = A1->ptr();
    T* const a2 = A2->ptr();
    T* const v = V->ptr();
    T* const t = _T->ptr();
    T* const work = WORK->ptr();
    const int ib = CONFIG_IB; // PLASMA(control/auxiliary.c)
    const int k = A1->dim(1);
    const int ldwork = WORK->size();

    PLASMA<T>::tsmqr( side, trans, m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, v, ldv, t, ldt, work, ldwork );
#endif
  }
};

template<typename T>
struct TaskBodyCPU<TaskGEQRT<T> > {
  void operator()( 
	CBLAS_ORDER order,
	ka::range2d_rw<T> A,
	ka::range2d_w<T>  _T,
	ka::range1d_w<T>  TAU,
	ka::range1d_w<T>  WORK
  )
  {
    const int m = A->dim(0); 
    const int n = A->dim(1);
    const int lda = A->lda();
    const int ldt = _T->lda();
    T* const a = A->ptr();
    T* const t = _T->ptr();
    T* const work = WORK->ptr();
    const int ib = CONFIG_IB; // PLASMA(control/auxiliary.c)

#if defined(CONFIG_USE_PLASMA)
    T* const tau = TAU->ptr();
    int res = PLASMA<T>::geqrt( m, n, ib, a, lda, t, ldt, tau, work );
    if(res){
      fprintf(stdout, "TaskGEQRT error (%d) from PLASMA\n", res );
      fflush(stdout);
    }
#else
    LAPACKE<T>::geqrt(
	((order == CblasColMajor) ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR),
	m, n, ib, a, lda, t, ldt, work );
#endif
  }
};

#endif
