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


// #define TRACE_THIS_FILE
namespace MPINET {

// -----------------------------------------------------------------------
int OutChannel::initialize(MPI_Comm comm, int dest) throw()
{
  ka::OutChannel::initialize();
  _comm = comm;
  _dest = dest;
  return 0;
}


// -----------------------------------------------------------------------
int OutChannel::terminate() throw()
{
  return 0;
}


// -----------------------------------------------------------------------
void OutChannel::flush( ka::Instruction* first, ka::Instruction* last )
{
  /* */
//  ka::logfile() << "In " << __PRETTY_FUNCTION__ << " " << last-first << std::endl;
  ka::Instruction* curr = first;
  int err;
  while (curr != last)
  {
    switch (curr->type) {
      case ka::Instruction::INST_VOID:
        break;

      case ka::Instruction::INST_NOP:
        if (curr->i_cbk.cbk !=0)
          (*curr->i_cbk.cbk)(0, this, curr->i_cbk.arg);
        break;

      case ka::Instruction::INST_WB:
        if (curr->i_cbk.cbk !=0)
          (*curr->i_cbk.cbk)(0, this, curr->i_cbk.arg);
        break;

      case ka::Instruction::INST_AM:
#if defined(TRACE_THIS_FILE)
printf("%i::Send AM message to %i, handler=%p, size=%i\n", ka::System::local_gid, _dest, (void*)curr->i_am.handler, (int)curr->i_am.size);
fflush(stdout);
#endif
        /* send header: first 2 fields */
        err = MPI_Send( &curr->i_am.handler, 2*sizeof(uint64_t), MPI_BYTE, _dest, 1, _comm );
        kaapi_assert(err == MPI_SUCCESS);

        /* send message: pointed by the third field */
        /* here use asynchronous send if available... */
        if (curr->i_am.size >0)
        {
#if defined(TRACE_THIS_FILE)
printf("%i::Send AM message data to %i, pointer=%p, size=%i\n", ka::System::local_gid, _dest, *(void**)curr->i_am.lptr, (int)curr->i_am.size);
fflush(stdout);
#endif
          err = MPI_Send( (void*)curr->i_am.lptr, (int)curr->i_am.size, MPI_BYTE, _dest, 3, _comm );
          kaapi_assert(err == MPI_SUCCESS);
        }
        if (curr->i_cbk.cbk !=0)
          (*curr->i_cbk.cbk)(err, this, curr->i_cbk.arg);
      break;

      case ka::Instruction::INST_RWDMA:
#if defined(TRACE_THIS_FILE)
printf("%i::Send rDMA message to %i, @=%p, size=%i\n", ka::System::local_gid, _dest, (void*)curr->i_rw.dptr, (int)curr->i_rw.size);
fflush(stdout);
#endif
        /* send header: first 2 fields */
        err = MPI_Send( &curr->i_rw.dptr, 2*sizeof(uint64_t), MPI_BYTE, _dest, 2, _comm );
        kaapi_assert(err == MPI_SUCCESS);

        /* send message: pointed by the third field */
        /* here use remote write op if available... */
        if (curr->i_rw.size >0)
        {
          err = MPI_Send( (void*)curr->i_rw.lptr, (int)curr->i_rw.size, MPI_BYTE, _dest, 3, _comm );
          kaapi_assert(err == MPI_SUCCESS);
        }

        if (curr->i_cbk.cbk !=0)
          (*curr->i_cbk.cbk)(err, this, curr->i_cbk.arg);
      break;
    };
    ++curr;
  }
//  ka::logfile() << "In " << __PRETTY_FUNCTION__ << std::endl;
  return;
}

} // -namespace
