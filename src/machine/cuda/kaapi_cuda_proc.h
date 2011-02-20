/*
** kaapi_cuda_proc.h
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
#ifndef KAAPI_CUDA_PROC_H_INCLUDED
# define KAAPI_CUDA_PROC_H_INCLUDED


#include <pthread.h>
#include <sys/types.h>
#include <cuda.h>


typedef struct kaapi_cuda_proc
{
  CUdevice dev;
  CUstream stream;

  CUcontext ctx;
  pthread_mutex_t ctx_lock;

  int is_initialized;

  unsigned int kasid_user;

} kaapi_cuda_proc_t;


int kaapi_cuda_proc_initialize(kaapi_cuda_proc_t*, unsigned int);
int kaapi_cuda_proc_cleanup(kaapi_cuda_proc_t*);
size_t kaapi_cuda_get_proc_count(void);
unsigned int kaapi_cuda_get_first_kid(void);


#endif /* ! KAAPI_CUDA_PROC_H_INCLUDED */
