/*
** kaapi_cuda_func.c
** xkaapi
** 
** Created on Jul 2010
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <cuda.h>
#include "kaapi_cuda_func.h"
#include "kaapi_cuda_error.h"


/* exported */

int kaapi_cuda_func_init(kaapi_cuda_func_t* fn)
{
  fn->off = 0;
  return 0;
}


int kaapi_cuda_func_load_ptx
(kaapi_cuda_func_t* fn, const char* mpath, const char* fname)
{
  CUresult res;

  res = cuModuleLoad(&fn->mod, mpath);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuModuleLoad", res);
    goto on_error0;
  }

  res = cuModuleGetFunction(&fn->fu, fn->mod, fname);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuModuleGetFunction", res);
    goto on_error1;
  }

  return 0;

 on_error1:
  cuModuleUnload(fn->mod);
 on_error0:
  return -1;
}


int kaapi_cuda_func_unload_ptx(kaapi_cuda_func_t* fn)
{
  cuModuleUnload(fn->mod);
  return 0;
}


int kaapi_cuda_func_push_uint(kaapi_cuda_func_t* fn, unsigned int value)
{
  CUresult res;

#define ALIGN_UP(__offset, __alignment) \
  __offset = ((__offset) + ((__alignment) - 1)) & ~((__alignment) - 1)

  ALIGN_UP(fn->off, __alignof(value));

  res = cuParamSeti(fn->fu, fn->off, value);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuParamSeti", res);
    return -1;
  }

  fn->off += sizeof(unsigned int);

  return 0;
}


int kaapi_cuda_func_push_float(kaapi_cuda_func_t* fn, float value)
{
  CUresult res;

  ALIGN_UP(fn->off, __alignof(value));

  res = cuParamSetf(fn->fu, fn->off, value);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuParamSetf", res);
    return -1;
  }

  fn->off += sizeof(float);

  return 0;
}


int kaapi_cuda_func_push_ptr(kaapi_cuda_func_t* fn, CUdeviceptr devptr)
{
  CUresult res;

  ALIGN_UP(fn->off, __alignof(devptr));

  res = cuParamSetv(fn->fu, fn->off, &devptr, sizeof(devptr));
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuParamSetv", res);
    return -1;
  }

  fn->off += sizeof(void*);

  return 0;
}


int kaapi_cuda_func_call_async
(
 kaapi_cuda_func_t* fn,
 CUstream stream,
 const kaapi_cuda_dim2_t* bdim,
 const kaapi_cuda_dim3_t* tdim
)
{
  CUresult res;

  /* finalize param memory */
  cuParamSetSize(fn->fu, fn->off);

  /* set the number of threads */
  res = cuFuncSetBlockShape(fn->fu, tdim->x, tdim->y, tdim->z);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuFuncSetBlockShape", res);
    goto on_error;
  }

  /* async kernel launch */
  res = cuLaunchGridAsync(fn->fu, bdim->x, bdim->y, stream);
  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuLaunchGridAsync", res);
    goto on_error;
  }

  return 0;

 on_error:
  return -1;
}


int kaapi_cuda_func_wait(kaapi_cuda_func_t* fn, CUstream stream)
{
  const CUresult res = cuStreamSynchronize(stream);

  if (res != CUDA_SUCCESS)
  {
    kaapi_cuda_error("cuStreamSynchronize", res);
    return -1;
  }

  return 0;

}
