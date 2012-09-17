
#include <cuda_runtime_api.h>

#if defined(__cplusplus)
extern "C" {
#endif

void magmablas_dlaswp(cudaStream_t stream, int n, double *dAT, int lda, 
                  int i1, int i2, int *ipiv, int inci);

void magmablas_dtranspose(cudaStream_t stream, double *odata, int ldo, 
                     double *idata, int ldi, 
                     int m, int n );

void magmablas_dswapblk(cudaStream_t stream, int n, 
                    double *dA1T, int lda1,
                    double *dA2T, int lda2,
                    int i1, int i2, int *ipiv, int inci, int offset);

#if defined(__cplusplus)
}
#endif

template<class T>
struct magmablas {
  typedef T value_type;

  static void laswp(cudaStream_t stream, int n, value_type* dAT, int lda, 
		    int i1, int i2, int *ipiv, int inci);

  static void transpose(cudaStream_t stream, value_type *odata, int ldo, 
		       value_type *idata, int ldi, 
		       int m, int n );

  static void swapblk(cudaStream_t stream, int n, 
		      value_type* dA1T, int lda1,
		      value_type* dA2T, int lda2,
		      int i1, int i2, int *ipiv, int inci, int offset);

};

template<>
struct magmablas<double> {
  typedef double value_type;

  static void laswp(cudaStream_t stream, int n, value_type* dAT, int lda, 
		    int i1, int i2, int *ipiv, int inci)
  {
    magmablas_dlaswp(stream, n, dAT, lda, i1, i2, ipiv, inci);
  }

  static void transpose(cudaStream_t stream, value_type *odata, int ldo, 
		       value_type *idata, int ldi, 
		       int m, int n )
  {
    magmablas_dtranspose(stream, odata, ldo, idata, ldi, m, n);
  }

  static void swapblk(cudaStream_t stream, int n, 
		      value_type* dA1T, int lda1,
		      value_type* dA2T, int lda2,
		      int i1, int i2, int *ipiv, int inci, int offset)
  {
    magmablas_dswapblk(stream, n, dA1T, lda1, dA2T, lda2, i1, i2, ipiv, inci, offset);
  }
};
