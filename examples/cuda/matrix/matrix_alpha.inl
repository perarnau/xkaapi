

template<>
struct TaskBodyAlpha<TaskDGEMM> {
  void operator()
  (
    ka::pointer_w<float> alpha,
    CBLAS_ORDER		   order, 
    CBLAS_TRANSPOSE transA,
    CBLAS_TRANSPOSE transB,
    double_type a,
    ka::range2d_r<double_type> Aik,
    ka::range2d_r<double_type> Akj,
    double_type b,
    ka::range2d_rw<double_type> Aij
  )
  {
      *alpha = 12.f;
  }
};

template<>
struct TaskBodyAlpha<TaskDPOTRF> {
  void operator()( 
    ka::pointer_w<float> alpha,
    CBLAS_ORDER order, CBLAS_UPLO uplo, ka::range2d_rw<double_type> A 
  )
  {
      *alpha = 0.f;
  }
};


template<>
struct TaskBodyAlpha<TaskDTRSM> {
  void operator()( 
    ka::pointer_w<float> alpha,
    CBLAS_ORDER		   order, 
    CBLAS_SIDE             side,
    CBLAS_UPLO             uplo,
    CBLAS_TRANSPOSE        transA,
    CBLAS_DIAG             diag,
    double_type                 a,
    ka::range2d_r <double_type> A, 
    ka::range2d_rw<double_type> C
  )
  {
      *alpha = 12.f;
  }
};


template<>
struct TaskBodyAlpha<TaskDSYRK> {
  void operator()(
    ka::pointer_w<float> alpha,
    CBLAS_ORDER		   order, 
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    double_type a,
    ka::range2d_r <double_type>  A, 
    double_type beta,
    ka::range2d_rw<double_type> C 
  )
  {
      *alpha = 12.f;
  }
};

template<>
struct TaskBodyAlpha<TaskDGETRF> {
  void operator()( 
    ka::pointer_w<float> alpha,
    CBLAS_ORDER order, 
    ka::range2d_rw<double_type> A, 
    ka::range1d_w<int> piv
  )
  {
      *alpha = 0.f;
  }
};

template<>
struct TaskBodyAlpha<TaskPlasmaDGESSM> {
  void operator()( 
    ka::pointer_w<float> alpha,
    CBLAS_ORDER order, 
    ka::range1d_r<int> piv,
    ka::range2d_r<double_type> L, 
    ka::range2d_rw<double_type> A
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
    ka::range2d_rw<double_type> U, 
    ka::range2d_rw<double_type> A,
    ka::range2d_rw<double_type> L,
    ka::range1d_w<int> piv,
    ka::range2d_rw<double_type> WORK
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
    ka::range2d_rw<double_type> A1, 
    ka::range2d_rw<double_type> A2,
    ka::range2d_r<double_type> L1,
    ka::range2d_r<double_type> L2,
    ka::range1d_r<int> piv
  )
  {
      *alpha = 6.f;
  }
};
