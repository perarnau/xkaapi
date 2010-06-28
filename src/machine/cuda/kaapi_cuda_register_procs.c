/*
** kaapi_cuda.c
** 
** Created on Jun 23
** Copyright 2009 INRIA.
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


#include <cuda.h>
#include <stdlib.h>
#include "kaapi_impl.h"
#include "kaapi_cuda.h"
#include "../common/kaapi_procinfo.h"


/* exported */

int kaapi_cuda_register_procs(kaapi_procinfo_list_t* kpl)
{
  const char* const gpuset_str = getenv("KAAPI_GPUSET");
  int devcount;
  int err;

  if (gpuset_str == NULL)
    return 0;

  if (cuInit(0) != CUDA_SUCCESS)
    return -1;

  if (cuDeviceGetCount(&devcount) != CUDA_SUCCESS)
    return -1;

  err = kaapi_procinfo_list_parse_string
    (kpl, gpuset_str, KAAPI_PROC_TYPE_GPU, (unsigned int)devcount);
  if (err)
    return -1;

  return 0;
}
