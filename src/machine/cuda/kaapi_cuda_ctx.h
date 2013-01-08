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

#ifndef KAAPI_CUDA_CTX_H_INCLUDED
#define KAAPI_CUDA_CTX_H_INCLUDED

#include <cuda_runtime_api.h>

#include "kaapi_impl.h"
#include "kaapi_cuda_proc.h"

#if defined(KAAPI_USE_CUPTI)
/* Need thread-safe context for traces */
void kaapi_cuda_ctx_set_(const int dev);

void kaapi_cuda_ctx_exit_(const int dev);
#else
/* Need thread-safe context for traces */
static inline void kaapi_cuda_ctx_set_(const int dev)
{
}

static inline void kaapi_cuda_ctx_exit_(const int dev)
{
}
#endif

static inline void kaapi_cuda_ctx_set(const int dev)
{
#if defined(KAAPI_USE_CUPTI)
  kaapi_cuda_ctx_set_(dev);
#endif
  const cudaError_t res = cudaSetDevice(dev);
  if (res != cudaSuccess) {
    fprintf(stderr, "%s: ERROR %d\n", __FUNCTION__, res);
    fflush(stderr);
  }
}

static inline void kaapi_cuda_ctx_exit(const int dev)
{
#if defined(KAAPI_USE_CUPTI)
  kaapi_cuda_ctx_exit_(dev);
#endif
}


static inline void kaapi_cuda_ctx_push(void)
{
  /* TODO future usage. */
  kaapi_cuda_ctx_set_(kaapi_cuda_self_device());
}

static inline void kaapi_cuda_ctx_pop(void)
{
  /* TODO future usage */
  kaapi_cuda_ctx_exit_(kaapi_cuda_self_device());
}

#endif				/* ! KAAPI_CUDA_CTX_H_INCLUDED */
