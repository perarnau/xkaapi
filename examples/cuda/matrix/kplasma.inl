
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
  
  static int tstrf_gpu(cudaStream_t stream, int m, int n, int ib, int nb,
                       value_type *hU, int ldhu, value_type *dU, int lddu,
                       value_type *hA, int ldha, value_type *dA, int ldda,
                       value_type *hL, int ldhl, value_type *dL, int lddl,
                       int *ipiv,
                       value_type *hwork, int ldhwork, value_type *dwork, int lddwork);
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
      status = CUBLAS<value_type>::trsm(
//                                         convertToSideMode(CblasRight),
                                         convertToSideMode(CblasLeft),
                                         convertToFillMode(CblasLower),
//                                         convertToOp(CblasTrans),
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
//                                          convertToOp(CblasNoTrans),
                                          convertToOp(CblasNoTrans),
                                          convertToOp(CblasTrans),
//                                          n, m-(i+sb), sb,
                                          m-(i+sb), n, sb,
                                          &mzone,
                                          &l[ldl*i+(i+sb)], ldl,
                                          &a[i], lda,
//                                          &l[ldl*i+(i+sb)], ldl,
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
           n1,
           &a1[ii+i], lda1,
//           &a1[ii+i], 1,
//           &a2[im], 1
           &a2[im], lda2
           );
        }
        ip = ip + 1;
      }
      
      status = CUBLAS<value_type>::trsm
      (
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
      
      status = CUBLAS<value_type>::gemm
      (
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
  
  static int tstrf_gpu(cudaStream_t stream, int m, int n, int ib, int nb,
                       value_type *hU, int ldhu, value_type *dU, int lddu,
                       value_type *hA, int ldha, value_type *dA, int ldda,
                       value_type *hL, int ldhl, value_type *dL, int lddl,
                       int *ipiv,
                       value_type *hwork, int ldhwork, value_type *dwork, int lddwork)
  {
#define UT(i,j) (dUT + (i)*ib*lddu + (j)*ib )
#define AT(i,j) (dAT + (i)*ib*ldda + (j)*ib )
#define L(i)    (dL  + (i)*ib*lddl          )
#define L2(i)   (dL2 + (i)*ib*lddl          )
#define hU(i,j) (hU  + (j)*ib*ldhu + (i)*ib )
#define hA(i,j) (hA  + (j)*ib*ldha + (i)*ib )
#define hL(i)   (hL  + (i)*ib*ldhl          )
#define hL2(i)  (hL2 + (i)*ib*ldhl          )
    
    value_type c_one     = 1.0;
    value_type c_neg_one = -1.0;
    
    int iinfo = 0;
    int info = 0;
    int maxm, mindim;
    int i, j, im, s, ip, ii, sb, p = 1;
    value_type *dAT, *dUT;
    value_type *dAp, *dUp;
#if 1
    value_type *dL2 = dL + ib;
    value_type *hL2 = hL + ib;
    p = 2;
#endif
    ip = 0;
    
    /* Function Body */
    mindim = std::min(m, n);
    s      = mindim / ib;
    
    /* Use hybrid blocked code. */
    maxm = ((m + 31)/32)*32;
    
    dUT = dU; dAT = dA;
    dAp = dwork;
    dUp = dAp + ib*lddwork;
    
    ip = 0;
    for( i=0; i<s; i++ )
    {
      ii = i * ib;
      sb = std::min(mindim-ii, ib);
      
      if ( i>0 ){
        // download i-th panel
        magmablas<value_type>::transpose( stream, dUp, lddu, UT(0, i), lddu, sb, ii );
        magmablas<value_type>::transpose( stream, dAp, ldda, AT(0, i), ldda, sb, m  );
        
        cublasGetMatrix( ii, sb, sizeof(value_type), dUp, lddu, hU(0, i), ldhu);
        cublasGetMatrix( m,  sb, sizeof(value_type), dAp, ldda, hA(0, i), ldha);
        
        // make sure that gpu queue is empty
        //cuCtxSynchronize();
        
#if 1
        CUBLAS<value_type>::trsm( CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
                                 n-(ii+sb), ib,
                                 &c_one, L2(i-1),      lddl,
                                 UT(i-1, i+1), lddu);
#else
        CUBLAS<value_type>::trsm( CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
                                 n-(ii+sb), ib,
                                 &c_one, L(i-1),       lddl,
                                 UT(i-1, i+1), lddu);
#endif
        CUBLAS<value_type>::gemm( CUBLAS_OP_N, CUBLAS_OP_N,
                                 n-(ii+sb), m, ib,
                                 &c_neg_one, UT(i-1, i+1), lddu,
                                 AT(0,   i-1), ldda,
                                 &c_one,     AT(0,   i+1), ldda );
      }
      
      // do the cpu part
#if 1
      PLASMA<value_type>::tstrf(m, sb, ib, nb,
                                (value_type*)hU(i, i), ldhu,
                                (value_type*)hA(0, i), ldha,
                                (value_type*)hL(i),    ldhl,
                                ipiv+ii,
                                (value_type*)hwork, ldhwork,
                                &info);
#endif
      
      if ( (info == 0) && (iinfo > 0) )
        info = iinfo + ii;
      
      // Need to swap betw U and A
#if 1
      // 	magmablas_dswapblk( 'R', n-(ii+sb),
      // 			    UT(i, i+1), lddu,
      // 			    AT(0, i+1), ldda,
      // 			    1, sb, ipiv+ii, 1, nb );
      magmablas<value_type>::swapblk( stream, n-(ii+sb),
                                     UT(i, i+1), lddu,
                                     AT(0, i+1), ldda,
                                     1, sb, ipiv+ii, 1, nb );
      
      for(j=0; j<ib; j++) {
        im = ipiv[ip]-1;
        if ( im == j ) {
          ipiv[ip] += ii;
        }
        ip++;
      }
#else
      for(j=0; j<ib; j++) {
        im = ipiv[ip]-1;
        if ( im != (j) ) {
          im = im - nb;
          //assert( (im>=0) && (im<m) );
          CUBLAS<value_type>::swap(  n-(ii+sb), UT(i, i+1)+j*lddu, 1, AT(0, i+1)+im*ldda, 1 );
          //magmablas_dswap( n-(ii+sb), UT(i, i+1)+j*lddu, 1, AT(0, i+1)+im*ldda, 1 );
        } else {
          ipiv[ip] += ii;
        }
        ip++;
      }
#endif
      
#if 1
      CORE_dlacpy( PlasmaUpperLower, sb, sb,
                  (value_type*)hL(i), ldhl,
                  (value_type*)hL2(i), ldhl );
      CORE_dtrtri( PlasmaLower, PlasmaUnit, sb,
                  (value_type*)hL2(i), ldhl, &info );
      if (info != 0) {
        fprintf(stderr, "ERROR, trtri returned with info = %d\n", info);
      }
#endif
      // upload i-th panel
      cublasSetMatrix( sb,   sb, sizeof(value_type), hU(i, i), ldhu, dUp,  lddu );
      cublasSetMatrix( m,    sb, sizeof(value_type), hA(0, i), ldha, dAp,  ldda );
      cublasSetMatrix( p*ib, sb, sizeof(value_type), hL(i),    ldhl, L(i), lddl );
      magmablas<value_type>::transpose( stream, UT(i, i), lddu, dUp, lddu, sb, sb);
      magmablas<value_type>::transpose( stream, AT(0, i), ldda, dAp, ldda, m,  sb);
      
      // make sure that gpu queue is empty
      //cuCtxSynchronize();
      
      // do the small non-parallel computations
      if ( s > (i+1) ) {
#if 1
        CUBLAS<value_type>::trsm( CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
                                 sb, sb,
                                 &c_one, L2(i),      lddl,
                                 UT(i, i+1), lddu);
#else
        CUBLAS<value_type>::trsm( CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
                                 sb, sb,
                                 &c_one, L(i),      lddl,
                                 UT(i, i+1), lddu);
#endif
        CUBLAS<value_type>::gemm( CUBLAS_OP_N, CUBLAS_OP_N,
                                 sb, m, sb,
                                 &c_neg_one, UT(i, i+1), lddu,
                                 AT(0, i  ), ldda,
                                 &c_one,     AT(0, i+1), ldda );
      }
      else {
#if 1
        CUBLAS<value_type>::trsm( CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
                                 n-mindim, sb,
                                 &c_one, L2(i),      lddl,
                                 UT(i, i+1), lddu);
#else
        CUBLAS<value_type>::trsm( CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
                                 n-mindim, sb,
                                 &c_one, L(i),      lddl,
                                 UT(i, i+1), lddu);
#endif
        CUBLAS<value_type>::gemm( CUBLAS_OP_N, CUBLAS_OP_N,
                                 n-mindim, m, sb,
                                 &c_neg_one, UT(i, i+1), lddu,
                                 AT(0, i  ), ldda,
                                 &c_one,     AT(0, i+1), ldda );
      }
    }
    
    return info;
#undef UT
#undef AT
#undef L
#undef L2
#undef hU
#undef hA
#undef hL
#undef hL2
  }
};

template<>
struct KPLASMA<float> {
  typedef float value_type;
  
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
        printf("TaskSSSSM::trsm() == %d\n", status);
      
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
  struct TaskGESSM: public ka::Task<5>::Signature
  <
  CBLAS_ORDER,			/* row / col */
  ka::R<ka::range1d<int> >,  /* pivot */
  uintptr_t,  /* pivot (GPU) */
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
  ka::W<ka::range2d<T> >,	      /* L */
  ka::W<ka::range1d<int> >,   /* pivot */
  uintptr_t	    /* WORK */
  >{};
  
  template<typename T>
  struct TaskSSSSM: public ka::Task<6>::Signature
  <
  CBLAS_ORDER,			/* row / col */
  ka::RW<ka::range2d<T> >,	    /* A1 */
  ka::RW<ka::range2d<T> >,	    /* A2 */
  ka::R<ka::range2d<T> >,	    /* L1 */
  ka::R<ka::range2d<T> >,	    /* L2 */
  uintptr_t  /* pivot (GPU) */
  >{};
};

template<typename T>
struct TaskBodyCPU<kplasma::TaskGESSM<T> > {
  void operator()(
                  CBLAS_ORDER order,
                  ka::range1d_r<int> piv,
                  uintptr_t piv_ptr,
                  ka::range2d_r<T> L,
                  ka::range2d_rw<T> A
                  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m = A->dim(1);
    const int n = A->dim(0);
    const int k = A->dim(1); /* TODO check */
    const int lda = A->lda();
    const int ldl = L->lda();
    T* const a = A->ptr();
    T* const l = (T*)L->ptr();
    int* const ipiv = (int*)piv->ptr();

    /* TODO problem here */
#if defined(CONFIG_VERBOSE)
    fprintf( stdout, "TaskCPU TaskGESSM A(%lu,%lu) a=%p lda=%d  L(%lu,%lu) l=%p ldl=%d k=%d\n",
            A->dim(0), A->dim(1), (void*)a, A->lda(),
            L->dim(0), L->dim(1), (void*)l, L->lda(),
            k
            );
    fflush(stdout);
#endif
    
    const int ib = CONFIG_IB_CPU;
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
                  ka::range2d_w<T> L,
                  ka::range1d_w<int> piv,
                  uintptr_t WORK
                  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m = A->dim(0);
    const int n = A->dim(1);
    const int lda = A->lda();
    const int ldl = L->lda();
    const int ldu = U->lda();
    const int ldw = m;
    T* const a = A->ptr();
    T* const l = L->ptr();
    T* const u = U->ptr();
    T* const work = (T*)WORK;
    int* const ipiv = (int*)piv->ptr();
    
#if defined(CONFIG_VERBOSE)
    fprintf( stdout, "TaskCPU TaskTSTRF U(%lu,%lu,%d) A(%lu,%lu,%d) L(%lu,%lu,%d)\n",
            U->dim(0), U->dim(1), U->lda(),
            A->dim(0), A->dim(1), A->lda(),
            L->dim(0), L->dim(1), L->lda() );
    fflush(stdout);
#endif
    const int ib = CONFIG_IB_CPU;
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
                  uintptr_t piv_ptr
                  )
  {
#if defined(CONFIG_USE_PLASMA)
    const int m1 = A1->dim(1);
    const int n1 = A1->dim(0);
    const int m2 = A2->dim(1);
    const int n2 = A2->dim(0);
    const int k = L1->dim(1);
    const int lda1 = A1->lda();
    const int lda2 = A2->lda();
    const int ldl1 = L1->lda();
    const int ldl2 = L2->lda();
    T* const a1 = A1->ptr();
    T* const a2 = A2->ptr();
    T* const l1 = (T*)L1->ptr();
    T* const l2 = (T*)L2->ptr();
    int* const ipiv = (int*)piv_ptr;
    
#if defined(CONFIG_VERBOSE)
    fprintf( stdout, "TaskCPU TaskSSSSM A1(%lu,%lu,%d) A2(%lu,%lu,%d) L1(%lu,%lu,%d) L2(%lu,%lu,%d), k=%d\n",
            A1->dim(0), A1->dim(1), A1->lda(),
            A2->dim(0), A2->dim(1), A2->lda(),
            L1->dim(0), L1->dim(1), L1->lda(),
            L2->dim(0), L2->dim(1), L2->lda(),
            k );
#endif
    const int ib = CONFIG_IB_CPU;
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
                  uintptr_t piv_ptr,                  
                  ka::range2d_r<T> L,
                  ka::range2d_rw<T> A
                  )
  {
    const int m = A->dim(1);
    const int n = A->dim(0);
    const int k = A->dim(1); /* TODO check */
    const int lda = A->lda();
    const int ldl = L->lda();
    T* const a = A->ptr();
    T* const l = (T*)L->ptr();
    int* const ipiv = (int*)piv_ptr;
    const int ib = CONFIG_IB_GPU;
    
#if defined(CONFIG_VERBOSE)
    fprintf( stdout, "TaskGPU TaskGESSM A(%lu,%lu) a=%p lda=%d  L(%lu,%lu) l=%p ldl=%d k=%d\n",
            A->dim(0), A->dim(1), (void*)a, A->lda(),
            L->dim(0), L->dim(1), (void*)l, L->lda(),
            k
            );
    fflush(stdout);
#endif
    KPLASMA<T>::gessm_gpu((cudaStream_t)stream.stream, m, n, k, ib, ipiv, l, ldl, a, lda);
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
                  uintptr_t piv_ptr                  
                  )
  {
    int m1 = A1->dim(1);
    int n1 = A1->dim(0);
    int m2 = A2->dim(1);
    int n2 = A2->dim(0);
    int k = L1->dim(1);
    int lda1 = A1->lda();
    int lda2 = A2->lda();
    int ldl1 = L1->lda();
    int ldl2 = L2->lda();
    T* const a1 = A1->ptr();
    T* const a2 = A2->ptr();
    T* const l1 = (T*)L1->ptr();
    T* const l2 = (T*)L2->ptr();
    int* const ipiv = (int*)piv_ptr;
    const int ib = CONFIG_IB_GPU;
    
#if defined(CONFIG_VERBOSE)
    fprintf( stdout, "TaskGPU TaskSSSSM A1(%lu,%lu,%d) A2(%lu,%lu,%d) L1(%lu,%lu,%d) L2(%lu,%lu,%d), k=%d\n",
            A1->dim(0), A1->dim(1), A1->lda(),
            A2->dim(0), A2->dim(1), A2->lda(),
            L1->dim(0), L1->dim(1), L1->lda(),
            L2->dim(0), L2->dim(1), L2->lda(),
            k );
#endif
    
    KPLASMA<T>::ssssm_gpu((cudaStream_t)stream.stream, m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, l1, ldl1, l2, ldl2, ipiv);
  }
};

#if 0
template<typename T>
struct TaskBodyGPU<kplasma::TaskTSTRF<T> > {
  void operator()(
                  ka::gpuStream stream,
                  CBLAS_ORDER order,
                  int nb,
                  ka::range2d_rw<T> U,
                  ka::range2d_rw<T> A,
                  ka::range2d_w<T> L,
                  uintptr_t piv,
                  uintptr_t WORK
                  )
  {
    int m = A->dim(0);
    int n = A->dim(1);
    int lda = A->lda();
    int ldl = L->lda();
    int ldu = U->lda();
    int ldw = m;
    T* const a = A->ptr();
    T* const l = L->ptr();
    T* const u = U->ptr();
    T* const work = (T*)WORK;
    int* const ipiv = (int*)piv;
    const int ib = CONFIG_IB_GPU;
    
#if defined(CONFIG_VERBOSE)
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
    cublasGetMatrix( L->dim(0), L->dim(1), sizeof(T), u, ldu, hu, ldu );
    cublasGetMatrix( A->dim(0), A->dim(1), sizeof(T), a, lda, ha, lda );
    memset( hl, 0, L->dim(0)*L->dim(1)*sizeof(T) );
    
    int info =
    KPLASMA<T>::tstrf_gpu( (cudaStream_t)stream.stream, m, n, ib, (int)L->dim(1), hu, ldu, u, ldu, ha, lda, a, lda, hl, ldl, l, ldl, ipiv, work, ldw, work, lda);
    if(info){
      fprintf( stdout, "TaskTSTRF ERROR (%d)\n", info );
      fflush(stdout);
    }
    
    free(ha);
    free(hl);
    free(hu);
    free(hwork);
  }
};
#endif

#endif /* CONFIG_USE_CUDA */

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
    const int ib = CONFIG_IB_CPU;
    
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
    const int ib = CONFIG_IB_CPU;
    
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
    const int ib = CONFIG_IB_CPU;
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
    const int ib = CONFIG_IB_CPU;
    
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
