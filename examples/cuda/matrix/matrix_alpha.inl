

template<typename T> 
struct TaskBodyAlpha<TaskGEMM<T> > {
  void operator()
  (
    ka::pointer_w<float> alpha,
    CBLAS_ORDER		   order, 
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    T a,
    ka::range2d_r<T> Aik,
    ka::range2d_r<T> Akj,
    T b,
    ka::range2d_rw<T> Aij
  )
  {
      *alpha = 12.f;
  }
};

template<typename T> 
struct TaskBodyAlpha<TaskPOTRF<T> > {
  void operator()( 
    ka::pointer_w<float> alpha,
    CBLAS_ORDER order, CBLAS_UPLO uplo, ka::range2d_rw<T> A 
  )
  {
      *alpha = 0.f;
  }
};


template<typename T> 
struct TaskBodyAlpha<TaskTRSM<T> > {
  void operator()( 
    ka::pointer_w<float> alpha,
    CBLAS_ORDER		   order, 
    CBLAS_SIDE             side,
    CBLAS_UPLO             uplo,
    CBLAS_TRANSPOSE        transA,
    CBLAS_DIAG             diag,
    T                 a,
    ka::range2d_r <T> A, 
    ka::range2d_rw<T> C
  )
  {
      *alpha = 5.f;
  }
};


template<typename T> 
struct TaskBodyAlpha<TaskSYRK<T> > {
  void operator()(
    ka::pointer_w<float> alpha,
    CBLAS_ORDER		   order, 
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    T a,
    ka::range2d_r <T>  A, 
    T beta,
    ka::range2d_rw<T> C 
  )
  {
      *alpha = 5.f;
  }
};

template<typename T> 
struct TaskBodyAlpha<TaskGETRF<T> > {
  void operator()( 
    ka::pointer_w<float> alpha,
    CBLAS_ORDER order, 
    ka::range2d_rw<T> A, 
    ka::range1d_w<int> piv
  )
  {
      *alpha = 5.f;
  }
};

template<>
struct TaskBodyAlpha<TaskPlasmaDGESSM> {
  void operator()( 
    ka::pointer_w<float> alpha,
    CBLAS_ORDER order, 
    ka::range1d_r<int> piv,
    ka::range2d_r<double> L, 
    ka::range2d_rw<double> A
  )
  {
      *alpha = 6.f;
  }
};


template<>
struct TaskBodyAlpha<TaskPlasmaDTSTRF> {
  void operator()( 
    ka::pointer_w<float> alpha,
    CBLAS_ORDER order, 
    int nb,
    ka::range2d_rw<double> U, 
    ka::range2d_rw<double> A,
    ka::range2d_rw<double> L,
    ka::range1d_w<int> piv,
    ka::range2d_rw<double> WORK
  )
  {
      *alpha = 6.f;
  }
};


template<>
struct TaskBodyAlpha<TaskPlasmaDSSSSM> {
  void operator()( 
    ka::pointer_w<float> alpha,
    CBLAS_ORDER order, 
    ka::range2d_rw<double> A1, 
    ka::range2d_rw<double> A2,
    ka::range2d_r<double> L1,
    ka::range2d_r<double> L2,
    ka::range1d_r<int> piv
  )
  {
      *alpha = 6.f;
  }
};
