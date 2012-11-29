/*
 ** xkaapi
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** Joao.Lima@imagf.r / joao.lima@inf.ufrgs.br
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

#include <stdio.h>

#include "cublas_v2.h"

#include "kaapi_impl.h"
#include "kaapi_cuda_cublas.h"
#include "kaapi_cuda_proc.h"

int kaapi_cuda_cublas_init(kaapi_cuda_proc_t * proc)
{
  const cublasStatus_t status = cublasCreate(&proc->ctx.handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stdout, "%s: cublasCreate CUBLAS ERROR %d\n", __FUNCTION__,
	    status);
    fflush(stdout);
    abort();
  }
  //cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
  cublasSetPointerMode(proc->ctx.handle, CUBLAS_POINTER_MODE_HOST);

  return 0;
}

void kaapi_cuda_cublas_set_stream(void)
{
  kaapi_processor_t *const self_proc = kaapi_get_current_processor();
  const cublasStatus_t status =
      cublasSetStream(self_proc->cuda_proc.ctx.handle,
		      kaapi_cuda_kernel_stream());
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stdout, "%s: CUBLAS ERROR %u\n", __FUNCTION__, status);
    fflush(stdout);
    abort();
  }
}

void kaapi_cuda_cublas_finalize(kaapi_cuda_proc_t * proc)
{
  cublasDestroy(proc->ctx.handle);
}

cublasHandle_t kaapi_cuda_cublas_handle(void)
{
  return (kaapi_get_current_processor()->cuda_proc.ctx.handle);
}
