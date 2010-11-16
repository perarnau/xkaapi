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
#include "kampinet_channel.h"
#include <mpi.h>

namespace MPINET {

// -----------------------------------------------------------------------
int OutChannel::initialize() throw()
{
  return 0;
}


// -----------------------------------------------------------------------
int OutChannel::terminate() throw()
{
  return 0;
}


// -----------------------------------------------------------------------
void OutChannel::flush( Net::Instruction* first, Net::Instruction* last )
{
  /* */
  Net::Instruction* curr = first;
  int err;
  while (curr != last)
  {
    switch (curr->type) {
      case Net::Instruction::INST_VOID:
        break;

      case Net::Instruction::INST_NOP:
        if (curr->i_cbk.cbk !=0)
          (*curr->i_cbk.cbk)(0, this, curr->i_cbk.arg);
        break;

      case Net::Instruction::INST_WB:
        if (curr->i_cbk.cbk !=0)
          (*curr->i_cbk.cbk)(0, this, curr->i_cbk.arg);
        break;

      case Net::Instruction::INST_AM:
        /* send header: first 2 fields */
        err = MPI_Send( &curr->i_am.handler, 2*sizeof(kaapi_uint64_t), MPI_BYTE, _dest, 1, MPI_COMM_WORLD );
        kaapi_assert(err == MPI_SUCCESS);

        /* send message: pointed by the third field */
        /* here use asynchronous send if available... */
        err = MPI_Send( curr->i_am.lptr, curr->i_am.size, MPI_BYTE, _dest, 3, MPI_COMM_WORLD );
        kaapi_assert(err == MPI_SUCCESS);
        if (curr->i_cbk.cbk !=0)
          (*curr->i_cbk.cbk)(err, this, curr->i_cbk.arg);
      break;

      case Net::Instruction::INST_RWDMA:
        /* send header: first 2 fields */
        err = MPI_Send( &curr->i_rw.dptr, 2*sizeof(kaapi_uint64_t), MPI_BYTE, _dest, 2, MPI_COMM_WORLD );
        kaapi_assert(err == MPI_SUCCESS);

        /* send message: pointed by the third field */
        /* here use remote write op if available... */
        err = MPI_Send( curr->i_rw.lptr, curr->i_rw.size, MPI_BYTE, _dest, 3, MPI_COMM_WORLD );
        kaapi_assert(err == MPI_SUCCESS);

        if (curr->i_cbk.cbk !=0)
          (*curr->i_cbk.cbk)(err, this, curr->i_cbk.arg);
      break;
    };
    ++curr;
  }
  return;
}

} // -namespace
