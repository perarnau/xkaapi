
extern "C" {
#include <lapacke.h>
}

template<class T>
struct LAPACKE {
  typedef T value_type;
  static lapack_int lacpy( int matrix_order, char uplo, lapack_int m,
                             lapack_int n, const value_type* a, lapack_int lda, value_type* b,
                             lapack_int ldb );
  static lapack_int larnv( lapack_int idist, lapack_int* iseed, lapack_int n,
                             value_type* x );
  static value_type lamch_work( char cmach );

  static value_type lange_work( int matrix_order, char norm, lapack_int m,
                                  lapack_int n, const value_type* a, lapack_int lda,
                                  value_type* work );

  static lapack_int laswp_work( int matrix_order, lapack_int n, value_type* a,
                                  lapack_int lda, lapack_int k1, lapack_int k2,
                                  const lapack_int* ipiv, lapack_int incx );

  static lapack_int lacpy_work( int matrix_order, char uplo, lapack_int m,
                                lapack_int n, const value_type* a, lapack_int lda,
                                value_type* b, lapack_int ldb );

  static lapack_int getf2_nopiv( int matrix_order, lapack_int m, lapack_int n,
			     value_type * a, lapack_int lda );

  static lapack_int geqrt( int matrix_order, lapack_int m, lapack_int n,
			    lapack_int nb, value_type* a, lapack_int lda,
			    value_type* t, lapack_int ldt, value_type* work );

  static lapack_int getrs( int matrix_order, char trans, lapack_int n,
			     lapack_int nrhs, const value_type* a, lapack_int lda,
			     const lapack_int* ipiv, value_type* b, lapack_int ldb );

  static lapack_int getrf_work( int matrix_order, lapack_int m, lapack_int n,
                                value_type* a, lapack_int lda, lapack_int* ipiv );
};

template<>
struct LAPACKE<double> {
  typedef double value_type;
  static lapack_int lacpy( int matrix_order, char uplo, lapack_int m,
                             lapack_int n, const value_type* a, lapack_int lda, value_type* b,
                             lapack_int ldb )
  { return LAPACKE_dlacpy( matrix_order, uplo, m, n, a, lda, b, ldb ); }
                           
  static lapack_int larnv( lapack_int idist, lapack_int* iseed, lapack_int n,
                             value_type* x )
  { return LAPACKE_dlarnv( idist, iseed, n, x ); }

  static value_type lamch_work( char cmach )
  { return LAPACKE_dlamch_work( cmach ); }

  static value_type lange_work( int matrix_order, char norm, lapack_int m,
                                  lapack_int n, const value_type* a, lapack_int lda,
                                  value_type* work )
  { return LAPACKE_dlange_work( matrix_order, norm, m, n, a, lda, work ); }

  static lapack_int laswp_work( int matrix_order, lapack_int n, value_type* a,
                                  lapack_int lda, lapack_int k1, lapack_int k2,
                                  const lapack_int* ipiv, lapack_int incx )
  { return LAPACKE_dlaswp_work( matrix_order, n, a, lda, k1, k2, ipiv, incx ); }

  static lapack_int lacpy_work( int matrix_order, char uplo, lapack_int m,
                                lapack_int n, const value_type* a, lapack_int lda,
                                value_type* b, lapack_int ldb )
  { return LAPACKE_dlacpy_work( matrix_order, uplo, m, n, a, lda, b, ldb ); }

  static lapack_int geqrt( int matrix_order, lapack_int m, lapack_int n,
			    lapack_int nb, value_type* a, lapack_int lda,
			    value_type* t, lapack_int ldt, value_type* work )
  {
    return LAPACKE_dgeqrt_work( matrix_order, m, n, nb, a, lda, t, ldt, work);
  }

  static lapack_int getf2_nopiv( int matrix_order, lapack_int m, lapack_int n,
			     value_type * a, lapack_int lda )
  {
    double c_one = 1.f, c_zero = 0.f;
    static int c__1 = 1;

    int a_dim1, a_offset, i__1, i__2, i__3;
    double z__1;
    static int i__, j;
    static double sfmin;

    a_dim1 = lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Compute machine safe minimum */
    sfmin = LAPACKE<value_type>::lamch_work('S');

    i__1 = std::min(m,n);
    for (j = 1; j <= i__1; ++j) 
      {
	/* Test for singularity. */
	i__2 = j + j * a_dim1;
	if (!(a[i__2] == c_zero)) {

	  /* Compute elements J+1:M of J-th column. */
	  if (j < m) {
	    if (abs(a[j + j * a_dim1]) >= sfmin) 
	      {
		i__2 = m - j;
		z__1 = c_one / a[j + j * a_dim1];
		CBLAS<value_type>::scal(i__2, z__1, &a[j + 1 + j * a_dim1], c__1);
	      } 
	    else 
	      {
		i__2 = m - j;
		for (i__ = 1; i__ <= i__2; ++i__) {
		  i__3 = j + i__ + j * a_dim1;
		  a[i__3] = a[j + i__ + j * a_dim1] / a[j + j*a_dim1];
		}
	      }
	  }
	  
	} 
	
	if (j < std::min(m,n)) {  
	  /* Update trailing submatrix. */
	  i__2 = m - j;
	  i__3 = n - j;
	  z__1 = -1.f; 
	  CBLAS<value_type>::ger( 
	      ((matrix_order == LAPACK_COL_MAJOR) ? CblasColMajor : CblasRowMajor),
	      i__2, i__3, z__1, &a[j + 1 + j * a_dim1], c__1,
		 &a[j + (j+1) * a_dim1], lda, &a[j + 1 + (j+1) * a_dim1], lda);
	}
      }

    return 0;
  }

  static lapack_int getrs( int matrix_order, char trans, lapack_int n,
			     lapack_int nrhs, const value_type* a, lapack_int lda,
			     const lapack_int* ipiv, value_type* b, lapack_int ldb )
  {
    return LAPACKE_dgetrs(matrix_order, trans, n, nrhs, a, lda, ipiv, b, ldb);
  }

  static lapack_int getrf_work( int matrix_order, lapack_int m, lapack_int n,
                                value_type* a, lapack_int lda, lapack_int* ipiv )
  {
    return LAPACKE_dgetrf_work(matrix_order, m, n, a, lda, ipiv);
  }
};

template<>
struct LAPACKE<float> {
  typedef float value_type;
  static lapack_int lacpy( int matrix_order, char uplo, lapack_int m,
                             lapack_int n, const value_type* a, lapack_int lda, value_type* b,
                             lapack_int ldb )
  { return LAPACKE_slacpy( matrix_order, uplo, m, n, a, lda, b, ldb ); }
                           
  static lapack_int larnv( lapack_int idist, lapack_int* iseed, lapack_int n,
                             value_type* x )
  { return LAPACKE_slarnv( idist, iseed, n, x ); }

  static value_type lamch_work( char cmach )
  { return LAPACKE_slamch_work( cmach ); }

  static value_type lange_work( int matrix_order, char norm, lapack_int m,
                                  lapack_int n, const value_type* a, lapack_int lda,
                                  value_type* work )
  { return LAPACKE_slange_work( matrix_order, norm, m, n, a, lda, work ); }

  static lapack_int laswp_work( int matrix_order, lapack_int n, value_type* a,
                                  lapack_int lda, lapack_int k1, lapack_int k2,
                                  const lapack_int* ipiv, lapack_int incx )
  { return LAPACKE_slaswp_work( matrix_order, n, a, lda, k1, k2, ipiv, incx ); }

  static lapack_int lacpy_work( int matrix_order, char uplo, lapack_int m,
                                lapack_int n, const value_type* a, lapack_int lda,
                                value_type* b, lapack_int ldb )
  { return LAPACKE_slacpy_work( matrix_order, uplo, m, n, a, lda, b, ldb ); }

  static lapack_int geqrt( int matrix_order, lapack_int m, lapack_int n,
			    lapack_int nb, value_type* a, lapack_int lda,
			    value_type* t, lapack_int ldt, value_type* work )
  {
    return LAPACKE_sgeqrt_work( matrix_order, m, n, nb, a, lda, t, ldt, work);
  }

  static lapack_int getrs( int matrix_order, char trans, lapack_int n,
			     lapack_int nrhs, const value_type* a, lapack_int lda,
			     const lapack_int* ipiv, value_type* b, lapack_int ldb )
  {
    return LAPACKE_sgetrs(matrix_order, trans, n, nrhs, a, lda, ipiv, b, ldb);
  }

  static lapack_int getrf_work( int matrix_order, lapack_int m, lapack_int n,
                                value_type* a, lapack_int lda, lapack_int* ipiv )
  {
    return LAPACKE_sgetrf_work(matrix_order, m, n, a, lda, ipiv);
  }
};


