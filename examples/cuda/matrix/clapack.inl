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

extern "C" {
#include <clapack.h>
}

template<class T>
struct CLAPACK {
  typedef T value_type;
  static int getrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   value_type *A, const int lda, int *ipiv);
  static int potrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, value_type *A, const int lda);
};

template<>
struct CLAPACK<double> {
  typedef double value_type;
  static int getrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   value_type *A, const int lda, int *ipiv)
  { return clapack_dgetrf(Order, M, N, A, lda, ipiv); }
                   
  static int potrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, value_type *A, const int lda)
  { return clapack_dpotrf(Order, Uplo, N, A, lda); }
};

template<>
struct CLAPACK<float> {
  typedef float value_type;
  static int getrf(const enum CBLAS_ORDER Order, const int M, const int N,
                   value_type *A, const int lda, int *ipiv)
  { return clapack_sgetrf(Order, M, N, A, lda, ipiv); }
                   
  static int potrf(const enum ATLAS_ORDER Order, const enum ATLAS_UPLO Uplo,
                   const int N, value_type *A, const int lda)
  { return clapack_spotrf(Order, Uplo, N, A, lda); }
};

