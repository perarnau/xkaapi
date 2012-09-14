
/*
 * KPLASMA - package of PLASMA kernels on CPU or GPU (CUDA)
 */

namespace kplasma {
  template<typename T>
  struct GESSM: public ka::Task<4>::Signature
  <
    CBLAS_ORDER,			/* row / col */
    ka::RPWP<ka::range1d <int> >,  /* pivot */
    ka::R<ka::range2d<T> >, /* L NB-by-NB lower trianguler tile */
    ka::RW<ka::range2d<T> > /* A, Updated by the application of L. */
  >{};

  template<typename T>
  struct TSTRF: public ka::Task<7>::Signature
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
  struct SSSSM: public ka::Task<6>::Signature
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
struct TaskBodyCPU<kplasma::GESSM<T> > {
  void operator()( 
    CBLAS_ORDER order, 
    ka::range1d_rpwp<int> piv,
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
struct TaskBodyCPU<kplasma::TSTRF<T> > {
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
    fprintf( stdout, "TaskDTSTRF L(%lu,%lu,%d)\n", L->dim(0), L->dim(1), L->lda() );
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
struct TaskBodyCPU<kplasma::SSSSM<T> > {
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
    fprintf( stdout, "TaskDSSSSM L(%lu,%lu), k=%d\n",
	    L1->dim(0), L1->dim(1), k );
#endif
    const int ib = CONFIG_IB; // from PLASMA
    PLASMA<T>::ssssm(m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, l1, ldl1, l2, ldl2, ipiv);
#else
    /* TODO */
#endif
  }
};

template<typename T>
struct TaskBodyGPU<kplasma::GESSM<T> > {
  void operator()( 
    ka::gpuStream stream,
    CBLAS_ORDER order, 
    ka::range1d_rpwp<int> piv,
    ka::range2d_r<T> L, 
    ka::range2d_rw<T> A
  )
  {
    int m = A->dim(0); 
    int n = A->dim(1);
    int k = A->dim(1);
    int lda = A->lda();
    int ldl = L->lda();
    T* const a = A->ptr();
    T* const l = (T*)L->ptr();
    int* const ipiv = (int*)piv->ptr();
    const int ib = CONFIG_IB; // from PLASMA

#if 1
    fprintf( stdout, "TaskGPU DGESSM A(%lu,%lu) a=%p lda=%d  L(%lu,%lu) l=%p ldl=%d\n",
	    A->dim(0), A->dim(1), (void*)a, A->lda(),
	    L->dim(0), L->dim(1), (void*)l, L->lda()
	   );
    fflush(stdout);
#endif

#if defined(CONFIG_USE_MAGMA)
    int* hipiv = (int*)calloc( piv->size(), sizeof(int) );
    cudaMemcpy( hipiv, ipiv, piv->size() * sizeof(int), 
	    cudaMemcpyDeviceToHost );

    const int info =
      MAGMA<T>::gessm('f', m, n, k, ib, ipiv, l, ldl, a, lda, a, lda);
    if(info){
      fprintf(stdout, "TaskGESSM ERROR (%d)\n", info);
      fflush(stdout);
    }
#else
    int* h_ipiv = (int*)calloc( k, sizeof(int) );
    cudaMemcpy( h_ipiv, ipiv, k * sizeof(int), 
	    cudaMemcpyDeviceToHost );

    static double zone  =  1.0;
    static double mzone = -1.0;
//    static int                ione  =  1;
    cublasStatus_t status;

    int i, sb;
    int tmp, tmp2;

    for(i = 0; i < k; i += ib) {
        sb = std::min(ib, k-i);
        tmp  = i+1;
        tmp2 = i+sb;

//	magmablas<T>::laswp(n, a, lda, i+1, i+sb, ipiv, 1);

        /*
         * Compute block row of U.
         */
	status = CUBLAS<T>::trsm (
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
	  printf("DGESSM::cublasDtrsm() == %d\n", status);

        if (i+sb < m) {
	    /*
	    * Update trailing submatrix.
	    */
	    status = CUBLAS<T>::gemm(
		convertToOp(CblasNoTrans),
		convertToOp(CblasNoTrans),
		m-(i+sb), n, sb,
		&mzone,
		&l[ldl*i+(i+sb)], ldl,
		&a[i], lda,
		&zone, &a[i+sb], lda
	    );
	    if (status != CUBLAS_STATUS_SUCCESS)
	      printf("DGESSM::cublasDgemm() == %d\n", status);
	}
    }
#endif
  }
};

#if 0

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
struct TaskBodyGPU<TaskTSTRF<T> > {
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
    fprintf( stdout, "TaskGPU DTSTRF A(%lu,%lu) a=%p lda=%d "
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
      fprintf( stdout, "TaskTSTRF ERROR (%d)\n", info );
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

template<typename T>
struct TaskBodyGPU<TaskSSSSM<T> > {
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
    const int ib = CONFIG_IB; // from PLASMA

#if 1
    fprintf( stdout, "TaskGPU DSSSSM A1(%lu,%lu) a1=%p lda1=%d "
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
#if 0
    int* hipiv = (int*)calloc( piv->size(), sizeof(int) );
    cudaMemcpy( hipiv, ipiv, piv->size() * sizeof(int), 
	    cudaMemcpyDeviceToHost );

    const int info = MAGMA<T>::ssssm('f', m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, l1, ldl1, l2, ldl2, hipiv);
    if(info){
      fprintf(stdout, "TaskSSSSM ERROR (%d)\n", info );
      fflush(stdout);
    }
#endif
#if 1
    static T zone  = 1.0;
    static T mzone =-1.0;
    cublasStatus_t status;

    int* h_ipiv = (int*)calloc( piv->size(), sizeof(int) );
    cudaMemcpy( h_ipiv, ipiv, piv->size() * sizeof(int), 
	    cudaMemcpyDeviceToHost );

    int i, ii, sb;
    int im, ip;
    ip = 0;

    for(ii = 0; ii < k; ii += ib) {
        sb = std::min(k-ii, ib);

        for(i = 0; i < sb; i++) {
            im = h_ipiv[ip]-1;

            if (im != (ii+i)) {
                im = im - m1;
		status = CUBLAS<T>::swap
		  (
		   kaapi_cuda_cublas_handle(),
		   n1,
		   &a1[ii+i], lda1,
		   &a2[im], lda2
		  );
            }
            ip = ip + 1;
        }

	status = CUBLAS<T>::trsm
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
	  printf("TaskSSSSM::trsm() == %d\n", status);

	status = CUBLAS<T>::gemm
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
#endif
  }
};

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
