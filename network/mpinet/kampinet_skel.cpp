/*
** xkaapi
** 
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
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
#include "kaapi_impl.h"
#include "kampinet_channel.h"
#include "kampinet_device.h"
#include <mpi.h>

namespace MPINET {


// --------------------------------------------------------------------
int Device::skel()
{
  MPI_Status status;
  union {
    Net::RWInstruction rw;
    Net::AMInstruction am;
  } header;
  void*          buffer_am = 0;
  kaapi_uint64_t sz_buffer_am = 0;

  Net::Service_fnc service = 0;
  int err;
  
  while (1)
  {
    err = MPI_Recv(&header, 2*sizeof(kaapi_uint64_t), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    kaapi_assert( (err == MPI_SUCCESS) );
    kaapi_assert( (status.MPI_ERROR == MPI_SUCCESS) );

    /* switch tag: 1 -> AM, 3 -> RW */
    switch (status.MPI_TAG) {
      case 1: /* AM */
        if (sz_buffer_am < header.am.size)
        {
          if (buffer_am !=0) free(buffer_am);
          buffer_am = malloc( header.am.size );
          sz_buffer_am = header.am.size;
        }
        err = MPI_Recv(buffer_am, header.am.size, MPI_BYTE, status.MPI_SOURCE, 3, MPI_COMM_WORLD, &status);
        kaapi_assert( (err == MPI_SUCCESS) );
        kaapi_assert( (status.MPI_ERROR == MPI_SUCCESS) );
        
        /* call the service handler */
        service = (Net::Service_fnc)header.am.handler;
        (*service)(0, status.MPI_SOURCE, buffer_am, sz_buffer_am );
        break;

      case 2: /* RW */
        err = MPI_Recv((void*)header.rw.dptr, header.rw.size, MPI_BYTE, status.MPI_SOURCE, 3, MPI_COMM_WORLD, &status);
        kaapi_assert(err == MPI_SUCCESS);
        kaapi_assert(status.MPI_ERROR == MPI_SUCCESS);
        break;

      default:
        kaapi_assert_m(false, "bad tag");
        break;
    }
  }
  return 0;
}


} // -namespace
