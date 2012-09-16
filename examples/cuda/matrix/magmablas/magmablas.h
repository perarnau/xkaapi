
#include <cuda_runtime_api.h>

#if defined(__cplusplus)
extern "C" {
#endif

void 
magmablas_dlaswp(cudaStream_t stream, int n, double *dAT, int lda, 
                  int i1, int i2, int *ipiv, int inci);

#if defined(__cplusplus)
}
#endif

template<class T>
struct magmablas {
  typedef T value_type;

  static void laswp(cudaStream_t stream, int n, value_type* dAT, int lda, 
		    int i1, int i2, int *ipiv, int inci);
};

template<>
struct magmablas<double> {
  typedef double value_type;

  static void laswp(cudaStream_t stream, int n, value_type* dAT, int lda, 
		    int i1, int i2, int *ipiv, int inci)
  {
    magmablas_dlaswp(stream, n, dAT, lda, i1, i2, ipiv, inci);
  }
};
