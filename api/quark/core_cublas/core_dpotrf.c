
#include <lapacke.h>

#include "core_cublas.h"

#include "control/common.h"

void CORE_dpotrf_cpu(int uplo, int N, double *A, int LDA, int *INFO)
{
#if 0
  fprintf(stdout, "%s: uplo=%d N=%d A=%p LDA=%d\n", __FUNCTION__,
          uplo, N, A, LDA);
  fflush(stdout);
#endif
  *INFO = LAPACKE_dpotrf_work(
                              LAPACK_COL_MAJOR,
                              lapack_const(uplo),
                              N, A, LDA );
}

int CORE_dpotrf_parallel_cpu(int uplo, int N, double *A, int LDA, int *INFO)
{
  if(N > 500) /* from control/context.c */
  {
    int NB;
    int status;
    plasma_context_t *plasma;
    PLASMA_sequence *sequence = NULL;
    PLASMA_request request = PLASMA_REQUEST_INITIALIZER;
    PLASMA_desc descA;
    
    plasma = plasma_context_self();
    if (plasma == NULL) {
      plasma = plasma_context_create();
      plasma_context_insert(plasma, pthread_self());
      plasma->quark = QUARK_Setup(1);
      plasma->world_size = 1;
      plasma_barrier_init(plasma);
      plasma->scheduling = PLASMA_DYNAMIC_SCHEDULING;
    }
    
    plasma->nb = 500;
    status = PLASMA_dpotrf(uplo, N, A, LDA);
    
    *INFO = status;
  }
  else
  {
    CORE_dpotrf_cpu(uplo, N, A, LDA, INFO);
  }
  
  return 0;
}

void CORE_dpotrf_quark_cpu(Quark *quark)
{
  int uplo;
  int n;
  double *A;
  int lda;
  PLASMA_sequence *sequence;
  PLASMA_request *request;
  int iinfo;
  
  int info;
  
  quark_unpack_args_7(quark, uplo, n, A, lda, sequence, request, iinfo);

  CORE_dpotrf_parallel_cpu(uplo, n, A, lda, &info);
  
  if (sequence->status == PLASMA_SUCCESS && info != 0)
    plasma_sequence_flush(quark, sequence, request, iinfo+info);
}
