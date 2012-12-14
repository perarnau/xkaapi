/*
 ** xkaapi
 **
 ** Copyright 2009, 2010, 2011, 2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** Joao.Lima@imag.fr / joao.lima@inf.ufrgs.br
 **
 ** This software is a computer program whose purpose is to execute
 ** multithreaded computation with data flow synchronization between
 ** threads.
 **
 ** This software is governed by the CeCILL-C license under French law
 ** and abiding by the rules of distribution of free software.  You can
 ** use, modify and/ or redistribute the software under the terms of
 ** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
 ** following URL "http://www.cecill.info".
 **
 ** As a counterpart to the access to the source code and rights to
 ** copy, modify and redistribute granted by the license, users are
 ** provided only with a limited warranty and the software's author,
 ** the holder of the economic rights, and the successive licensors
 ** have only limited liability.
 **
 ** In this respect, the user's attention is drawn to the risks
 ** associated with loading, using, modifying and/or developing or
 ** reproducing the software by the user in light of its specific
 ** status of free software, that may mean that it is complicated to
 ** manipulate, and that also therefore means that it is reserved for
 ** developers and experienced professionals having in-depth computer
 ** knowledge. Users are therefore encouraged to load and test the
 ** software's suitability as regards their requirements in conditions
 ** enabling the security of their systems and/or data to be ensured
 ** and, more generally, to use and operate it in the same conditions
 ** as regards security.
 **
 ** The fact that you are presently reading this means that you have
 ** had knowledge of the CeCILL-C license and that you accept its
 ** terms.
 **
 */

#if CONFIG_USE_MAGMA

/* export functions */

typedef int magma_int_t;

#if defined(__cplusplus)
extern "C" {
#endif

magma_int_t magma_spotrf_gpu
(char, magma_int_t, float*, magma_int_t, magma_int_t*);

magma_int_t magma_dpotrf_gpu
(char, magma_int_t, double*, magma_int_t, magma_int_t*);

magma_int_t magma_sgetrf_gpu
(
 magma_int_t m, magma_int_t n, float *A,
 magma_int_t lda, magma_int_t *ipiv,
 magma_int_t *info
);

magma_int_t magma_dgetrf_gpu
(
 magma_int_t m, magma_int_t n, double *A,
 magma_int_t lda, magma_int_t *ipiv,
 magma_int_t *info
);

magma_int_t magma_sgetrf_nopiv_gpu
(
 magma_int_t m, magma_int_t n, float *A,
 magma_int_t lda, 
 magma_int_t *info
);

magma_int_t magma_dgetrf_nopiv_gpu
(
 magma_int_t m, magma_int_t n, double *A,
 magma_int_t lda, 
 magma_int_t *info
);

void magmablas_sgemm(char tA, char tB,
		     magma_int_t m, magma_int_t n, magma_int_t k, 
		     float alpha,
		     const float *A, magma_int_t lda, 
		     const float *B, magma_int_t ldb, 
		     float beta,
		     float *C, magma_int_t ldc);

void magmablas_dgemm(char tA, char tB,
		     magma_int_t m, magma_int_t n, magma_int_t k, 
		     double alpha,
		     const double *A, magma_int_t lda, 
		     const double *B, magma_int_t ldb, 
		     double beta,
		     double *C, magma_int_t ldc);

magma_int_t
magma_dgessm_gpu( char storev, magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t ib, 
                  magma_int_t *ipiv, 
                  double *dL1, magma_int_t lddl1, 
                  double *dL,  magma_int_t lddl, 
                  double *dA,  magma_int_t ldda, 
                  magma_int_t *info);

magma_int_t magma_sgessm_gpu( char storev, magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t ib, 
                  magma_int_t *ipiv, 
                  float *dL1, magma_int_t lddl1, 
                  float *dL,  magma_int_t lddl, 
                  float *dA,  magma_int_t ldda, 
                  magma_int_t *info);

magma_int_t
magma_dtstrf_gpu( char storev, magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb,
                  double *hU, magma_int_t ldhu, double *dU, magma_int_t lddu, 
                  double *hA, magma_int_t ldha, double *dA, magma_int_t ldda, 
                  double *hL, magma_int_t ldhl, double *dL, magma_int_t lddl,
                  magma_int_t *ipiv, 
                  double *hwork, magma_int_t ldhwork, double *dwork, magma_int_t lddwork,
                  magma_int_t *info);

magma_int_t magma_ststrf_gpu( char storev, magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb,
                  float *hU, magma_int_t ldhu, float *dU, magma_int_t lddu, 
                  float *hA, magma_int_t ldha, float *dA, magma_int_t ldda, 
                  float *hL, magma_int_t ldhl, float *dL, magma_int_t lddl,
                  magma_int_t *ipiv, 
                  float *hwork, magma_int_t ldhwork, float *dwork, magma_int_t lddwork,
                  magma_int_t *info);

magma_int_t
magma_dssssm_gpu(char storev, magma_int_t m1, magma_int_t n1, 
                 magma_int_t m2, magma_int_t n2, magma_int_t k, magma_int_t ib, 
                 double *dA1, magma_int_t ldda1, 
                 double *dA2, magma_int_t ldda2, 
                 double *dL1, magma_int_t lddl1, 
                 double *dL2, magma_int_t lddl2,
                 magma_int_t *IPIV, magma_int_t *info);

magma_int_t magma_sssssm_gpu(char storev, magma_int_t m1, magma_int_t n1, 
                 magma_int_t m2, magma_int_t n2, magma_int_t k, magma_int_t ib, 
                 float *dA1, magma_int_t ldda1, 
                 float *dA2, magma_int_t ldda2, 
                 float *dL1, magma_int_t lddl1, 
                 float *dL2, magma_int_t lddl2,
                 magma_int_t *IPIV, magma_int_t *info);

void   magmablas_dlaswp( magma_int_t N, 
             double *dAT, magma_int_t lda, 
             magma_int_t i1,  magma_int_t i2, 
             magma_int_t *ipiv, magma_int_t inci );


magma_int_t magma_dgeqrf_gpu( magma_int_t m, magma_int_t n, 
                              double *dA,  magma_int_t ldda, 
                              double *tau, double *dT, 
                              magma_int_t *info);

magma_int_t magma_sgeqrf_gpu( magma_int_t m, magma_int_t n, 
                  float *dA,   magma_int_t ldda,
                  float *tau, float *dT, 
                  magma_int_t *info );

magma_int_t magma_dormqr_gpu( char side, char trans, 
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              double *a,    magma_int_t lda, double *tau, 
                              double *c,    magma_int_t ldc,
                              double *work, magma_int_t lwork, 
                              double *td,   magma_int_t nb, magma_int_t *info);

magma_int_t magma_sormqr_gpu(char side, char trans,
                 magma_int_t m, magma_int_t n, magma_int_t k,
                 float *dA,    magma_int_t ldda, 
                 float *tau,
                 float *dC,    magma_int_t lddc,
                 float *hwork, magma_int_t lwork,
                 float *dT,    magma_int_t nb, 
                 magma_int_t *info);

#if defined(__cplusplus)
}
#endif

/* create structures */

/* for MAGMA */
template<class T>
struct MAGMA {
  typedef T value_type;
  static int potrf(char uplo, int n, value_type *A, int lda);

  static int getrf(int m, int n, value_type* A, int lda, int* piv);

  static int getrf_nopiv(int m, int n, value_type* A, int lda);

  static int ormqr(char side, char trans, 
		    int m, int n, int k,
		    value_type *a,    int lda, value_type *tau, 
		    value_type *c,    int ldc,
		    value_type *work, int lwork, 
		    value_type *td,   int nb);

  static int gessm(char storev, int m, int n, int k, int ib, 
		    int *ipiv, 
		    value_type *dL1, int lddl1, 
		    value_type *dL,  int lddl, 
		    value_type *dA,  int ldda);

  static int tstrf(char storev, int m, int n, int ib, int nb,
		    value_type *hU, int ldhu, value_type *dU, int lddu, 
		    value_type *hA, int ldha, value_type *dA, int ldda, 
		    value_type *hL, int ldhl, value_type *dL, int lddl,
		    int *ipiv, 
		    value_type *hwork, int ldhwork, value_type *dwork, int lddwork);

  static int ssssm(char storev, int m1, int n1, 
		   int m2, int n2, int k, int ib, 
		   value_type *dA1, int ldda1, 
		   value_type *dA2, int ldda2, 
		   value_type *dL1, int lddl1, 
		   value_type *dL2, int lddl2,
		   int *IPIV);
};
             
template<>
struct MAGMA<double> {
  typedef double value_type;
  static int potrf(char uplo, int n, value_type *A, int lda)
  {
    int info;
    magma_dpotrf_gpu( uplo, n, A, lda, &info );
    return info;
  }

  static int getrf(int m, int n, value_type* A, int lda, int* piv )
  {
    int info;
    magma_dgetrf_gpu( m, n, A, lda, piv, &info );
    return info;
  }

  static int getrf_nopiv(int m, int n, value_type* A, int lda )
  {
    int info;
    magma_dgetrf_nopiv_gpu( m, n, A, lda, &info );
    return info;
  }

  static int ormqr(char side, char trans, 
		    int m, int n, int k,
		    value_type *a,    int lda, value_type *tau, 
		    value_type *c,    int ldc,
		    value_type *work, int lwork, 
		    value_type *td,   int nb)
  {
    int info;
    magma_dormqr_gpu(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, td, nb, &info);
    return info;
  }

  static int gessm(char storev, int m, int n, int k, int ib, 
		    int *ipiv, 
		    value_type *dL1, int lddl1, 
		    value_type *dL,  int lddl, 
		    value_type *dA,  int ldda)
  {
    int info;
    magma_dgessm_gpu(storev, m, n, k, ib, ipiv, dL1, lddl1, dL, lddl, dA, ldda, &info);
    return info;
  }

  static int tstrf(char storev, int m, int n, int ib, int nb,
		    value_type *hU, int ldhu, value_type *dU, int lddu, 
		    value_type *hA, int ldha, value_type *dA, int ldda, 
		    value_type *hL, int ldhl, value_type *dL, int lddl,
		    int *ipiv, 
		    value_type *hwork, int ldhwork, value_type *dwork, int lddwork)
  {
    int info;
    magma_dtstrf_gpu(storev, m, n, ib, nb, hU, ldhu, dU, lddu, hA, ldha, dA, ldda, hL, ldhl, dL, lddl, ipiv, hwork, ldhwork, dwork, lddwork, &info);
    return info;
  }

  static int ssssm(char storev, int m1, int n1, 
		   int m2, int n2, int k, int ib, 
		   value_type *dA1, int ldda1, 
		   value_type *dA2, int ldda2, 
		   value_type *dL1, int lddl1, 
		   value_type *dL2, int lddl2,
		   int *IPIV)
  {
    int info;
    magma_dssssm_gpu(storev, m1, n1, m2, n2, k, ib, dA1, ldda1, dA2, ldda2, dL1, lddl1, dL2, lddl2, IPIV, &info);
    return info;
  }
};

template<>
struct MAGMA<float> {
  typedef float value_type;
  static int potrf(char uplo, int n, value_type *A, int lda)
  {
    int info;
    magma_spotrf_gpu( uplo, n, A, lda, &info );
    return info;
  }

  static int getrf(int m, int n, value_type* A, int lda, int* piv )
  {
    int info;
    magma_sgetrf_gpu( m, n, A, lda, piv, &info );
    return info;
  }

  static int getrf_nopiv(int m, int n, value_type* A, int lda )
  {
    int info;
    magma_sgetrf_nopiv_gpu( m, n, A, lda, &info );
    return info;
  }

  static int ormqr(char side, char trans, 
		    int m, int n, int k,
		    value_type *a,    int lda, value_type *tau, 
		    value_type *c,    int ldc,
		    value_type *work, int lwork, 
		    value_type *td,   int nb)
  {
    int info;
    magma_sormqr_gpu(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, td, nb, &info);
    return info;
  }

  static int gessm(char storev, int m, int n, int k, int ib, 
		    int *ipiv, 
		    value_type *dL1, int lddl1, 
		    value_type *dL,  int lddl, 
		    value_type *dA,  int ldda)
  {
    int info;
    magma_sgessm_gpu(storev, m, n, k, ib, ipiv, dL1, lddl1, dL, lddl, dA, ldda, &info);
    return info;
  }

  static int tstrf(char storev, int m, int n, int ib, int nb,
		    value_type *hU, int ldhu, value_type *dU, int lddu, 
		    value_type *hA, int ldha, value_type *dA, int ldda, 
		    value_type *hL, int ldhl, value_type *dL, int lddl,
		    int *ipiv, 
		    value_type *hwork, int ldhwork, value_type *dwork, int lddwork)
  {
    int info;
    magma_ststrf_gpu(storev, m, n, ib, nb, hU, ldhu, dU, lddu, hA, ldha, dA, ldda, hL, ldhl, dL, lddl, ipiv, hwork, ldhwork, dwork, lddwork, &info);
    return info;
  }

  static int ssssm(char storev, int m1, int n1, 
		   int m2, int n2, int k, int ib, 
		   value_type *dA1, int ldda1, 
		   value_type *dA2, int ldda2, 
		   value_type *dL1, int lddl1, 
		   value_type *dL2, int lddl2,
		   int *IPIV)
  {
    int info;
    magma_sssssm_gpu(storev, m1, n1, m2, n2, k, ib, dA1, ldda1, dA2, ldda2, dL1, lddl1, dL2, lddl2, IPIV, &info);
    return info;
  }
};

#endif // CONFIG_USE_MAGMA

