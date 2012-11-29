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

#if defined(CONFIG_USE_PLASMA)

extern "C" {
#include "plasma.h"
#include "core_blas.h"
}

template<class T>
struct PLASMA {
  typedef T value_type;
  static int geqrt(int m, int n, int ib,
                 value_type* a, int lda,
                 value_type* t, int ldt,
                 value_type* tau, value_type* work);

  static int ormqr(int side, int trans,
		   int m, int n, int k, int ib,
		   value_type *v, int ldv,
		   value_type *t, int ldt,
		   value_type *c, int ldc,
		   value_type *work, int ldwork);

  static int tsqrt(int m, int n, int ib,
		   value_type *a1, int lda1,
		   value_type *a2, int lda2,
		   value_type *t, int ldt,
		   value_type *tau, value_type *work);

  static int tsmqr(int side, int trans,
		   int m1, int n1, int m2, int n2, int k, int ib,
		   value_type *a1, int lda1,
		   value_type *a2, int lda2,
		   value_type *v, int ldv,
		   value_type *t, int ldt,
		   value_type *work, int ldwork);

  static int gessm(int m, int n, int k, int ib,
		   int *ipiv,
		   value_type *l, int ldl,
		   value_type *a, int lda);

  static int tstrf(int m, int n, int ib, int nb,
		   value_type *u, int ldu,
		   value_type *a, int lda,
		   value_type *l, int ldl,
		   int *ipiv, value_type *work,
		   int ldwork, int *info);

  static int ssssm(int m1, int n1, int m2, int n2, int k, int ib,
		   value_type *a1, int lda1,
		   value_type *a2, int lda2,
		   value_type *l1, int ldl1,
		   value_type *l2, int ldl2,
		   int *ipiv);

  static int getrf(int m, int n, int ib,
		      value_type *a, int lda,
		      int *ipiv, int *info);
  
  static void plrnt(int m, int n, value_type *A, int lda,
          int bigM, int m0, int n0, unsigned long long int seed);
  
  static void plgsy(value_type bump, int m, int n, value_type *A, int lda,
                    int bigM, int m0, int n0, unsigned long long int seed );
};

static inline int
convertToSidePlasma( int side ) 
{
    switch (side) {
	case CblasRight:
            return PlasmaRight;
	case CblasLeft:
        default:
         return PlasmaLeft;
    }        
}

static inline int
convertToTransPlasma( int trans ) 
{
    switch(trans) {
        case CblasNoTrans:
            return PlasmaNoTrans;
        case CblasTrans:
            return PlasmaTrans;
        case CblasConjTrans:
            return PlasmaConjTrans;                        
        default:
            return PlasmaNoTrans;
    }
}

template<>
struct PLASMA<double> {
  typedef double value_type;
  static int geqrt(int m, int n, int ib,
                 value_type* a, int lda,
                 value_type* t, int ldt,
                 value_type* tau, value_type* work)
  {
    return CORE_dgeqrt(m, n, ib, a, lda, t, ldt, tau, work);
  }

  static int ormqr(int side, int trans,
		   int m, int n, int k, int ib,
		   value_type *v, int ldv,
		   value_type *t, int ldt,
		   value_type *c, int ldc,
		   value_type *work, int ldwork)
  {
    return CORE_dormqr( convertToSidePlasma(side), convertToTransPlasma(trans),
	    m, n, k, ib,
	    v, ldv, t, ldt, c, ldc, work, ldwork);
  }

  static int tsqrt(int m, int n, int ib,
		   value_type *a1, int lda1,
		   value_type *a2, int lda2,
		   value_type *t, int ldt,
		   value_type *tau, value_type *work)
  {
    return CORE_dtsqrt(m, n, ib, a1, lda1, a2, lda2, t, ldt, tau, work);
  }

  static int tsmqr(int side, int trans,
		   int m1, int n1, int m2, int n2, int k, int ib,
		   value_type *a1, int lda1,
		   value_type *a2, int lda2,
		   value_type *v, int ldv,
		   value_type *t, int ldt,
		   value_type *work, int ldwork)
  {
    return CORE_dtsmqr( convertToSidePlasma(side), convertToTransPlasma(trans),
	m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, v, ldv, t, ldt, work, ldwork);
  }

  static int gessm(int m, int n, int k, int ib,
		   int *ipiv,
		   value_type *l, int ldl,
		   value_type *a, int lda)
  {
    return CORE_dgessm(m, n, k, ib, ipiv, l, ldl, a, lda);
  }

  static int tstrf(int m, int n, int ib, int nb,
		   value_type *u, int ldu,
		   value_type *a, int lda,
		   value_type *l, int ldl,
		   int *ipiv, value_type *work,
		   int ldwork, int *info)
  {
    return CORE_dtstrf(m, n, ib, nb, u, ldu, a, lda, l, ldl, ipiv, work, ldwork, info);
  }

  static int ssssm(int m1, int n1, int m2, int n2, int k, int ib,
		   value_type *a1, int lda1,
		   value_type *a2, int lda2,
		   value_type *l1, int ldl1,
		   value_type *l2, int ldl2,
		   int *ipiv)
  {
    return CORE_dssssm(m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, l1, ldl1, l2, ldl2, ipiv);
  }

  static int getrf(int m, int n, int ib,
		      value_type *a, int lda,
		      int *ipiv, int *info)
  {
    return CORE_dgetrf_incpiv(m, n, ib, a, lda, ipiv, info);
  }
  
  static void plrnt(int m, int n, value_type *A, int lda,
                    int bigM, int m0, int n0, unsigned long long int seed)
  {
    CORE_dplrnt(m, n, A, lda, bigM, m0, n0, seed);
  }
  
  static void plgsy(value_type bump, int m, int n, value_type *A, int lda,
                    int bigM, int m0, int n0, unsigned long long int seed)
  {
    CORE_dplgsy(bump, m, n, A, lda, bigM, m0, n0, seed);
  }
};

template<>
struct PLASMA<float> {
  typedef float value_type;
  static int geqrt(int m, int n, int ib,
                 value_type* a, int lda,
                 value_type* t, int ldt,
                 value_type* tau, value_type* work)
  {
    return CORE_sgeqrt(m, n, ib, a, lda, t, ldt, tau, work);
  }

  static int ormqr(int side, int trans,
		   int m, int n, int k, int ib,
		   value_type *v, int ldv,
		   value_type *t, int ldt,
		   value_type *c, int ldc,
		   value_type *work, int ldwork)
  {
    return CORE_sormqr( convertToSidePlasma(side), convertToTransPlasma(trans),
	    m, n, k, ib,
	    v, ldv, t, ldt, c, ldc, work, ldwork);
  }

  static int tsqrt(int m, int n, int ib,
		   value_type *a1, int lda1,
		   value_type *a2, int lda2,
		   value_type *t, int ldt,
		   value_type *tau, value_type *work)
  {
    return CORE_stsqrt(m, n, ib, a1, lda1, a2, lda2, t, ldt, tau, work);
  }

  static int tsmqr(int side, int trans,
		   int m1, int n1, int m2, int n2, int k, int ib,
		   value_type *a1, int lda1,
		   value_type *a2, int lda2,
		   value_type *v, int ldv,
		   value_type *t, int ldt,
		   value_type *work, int ldwork)
  {
    return CORE_stsmqr( convertToSidePlasma(side), convertToTransPlasma(trans),
	m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, v, ldv, t, ldt, work, ldwork);
  }

  static int gessm(int m, int n, int k, int ib,
		   int *ipiv,
		   value_type *l, int ldl,
		   value_type *a, int lda)
  {
    return CORE_sgessm(m, n, k, ib, ipiv, l, ldl, a, lda);
  }

  static int tstrf(int m, int n, int ib, int nb,
		   value_type *u, int ldu,
		   value_type *a, int lda,
		   value_type *l, int ldl,
		   int *ipiv, value_type *work,
		   int ldwork, int *info)
  {
    return CORE_ststrf(m, n, ib, nb, u, ldu, a, lda, l, ldl, ipiv, work, ldwork, info);
  }

  static int ssssm(int m1, int n1, int m2, int n2, int k, int ib,
		   value_type *a1, int lda1,
		   value_type *a2, int lda2,
		   value_type *l1, int ldl1,
		   value_type *l2, int ldl2,
		   int *ipiv)
  {
    return CORE_sssssm(m1, n1, m2, n2, k, ib, a1, lda1, a2, lda2, l1, ldl1, l2, ldl2, ipiv);
  }

  static int getrf(int m, int n, int ib,
		      value_type *a, int lda,
		      int *ipiv, int *info)
  {
    return CORE_sgetrf_incpiv(m, n, ib, a, lda, ipiv, info);
  }
  
  static void plrnt(int m, int n, value_type *A, int lda,
                    int bigM, int m0, int n0, unsigned long long int seed)
  {
    CORE_splrnt(m, n, A, lda, bigM, m0, n0, seed);
  }
  
  static void plgsy(value_type bump, int m, int n, value_type *A, int lda,
                    int bigM, int m0, int n0, unsigned long long int seed)
  {
    CORE_splgsy(bump, m, n, A, lda, bigM, m0, n0, seed);
  }
};

#endif /* CONFIG_USE_PLASMA */
