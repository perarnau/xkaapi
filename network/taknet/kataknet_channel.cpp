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
#include "kataknet_channel.h"
#include <iostream>
namespace TAKNET {

// -----------------------------------------------------------------------
int OutChannel::initialize( int dest) throw()
{
  ka::OutChannel::initialize();
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
  int err;
  ka::Instruction* curr = first;  

  /* */
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
      {
        /* call cbk with writer barrier */
        if (curr->i_cbk.cbk !=0)
          (*curr->i_cbk.cbk)(0, this, curr->i_cbk.arg);
      } break;

      case ka::Instruction::INST_AM:
      {
        /* send header: first 2 fields */
        Header header;
        header.opcode   = 'A';
        header.sid      = curr->i_am.handler;
        header.size     = curr->i_am.size;
        if (curr->i_am.size ==0)
        {
          err = taktuk_send( 1+_dest, TAKTUK_TARGET_ANY, &header, sizeof(header));
        } else {
          iovec iov[2];
          iov[0].iov_base = &header;
          iov[0].iov_len  = sizeof(Header);
          iov[1].iov_base = (void*)curr->i_am.lptr;
          iov[1].iov_len  = curr->i_am.size;
          err = taktuk_sendv( 1+_dest, TAKTUK_TARGET_ANY, iov, 2);
        }
        std::cout << "[taknet] send AM fnc @:" << (void*)curr->i_am.handler
                  << ", data size:" << header.size << std::endl;
        /* debug mode: assert during communication */
        kaapi_assert_debug(err == TAKNET_OK);

        if (curr->i_cbk.cbk !=0)
          (*curr->i_cbk.cbk)(err, this, curr->i_cbk.arg);
      } break;

      case ka::Instruction::INST_RWDMA:
      {
        iovec iov[2];
        Header header;
        header.opcode   = 'D';
        header.size     = curr->i_rw.size;
        header.ptr      = (uint64_t)curr->i_rw.dptr;
        std::cout << "[taknet] send RDMA header @:" << (void*)curr->i_rw.dptr
                  << ", data size:" << curr->i_rw.size << std::endl;
        err = taktuk_send( 1+_dest, TAKTUK_TARGET_ANY, &header, sizeof(header));
        kaapi_assert_debug(err == TAKNET_OK);
        if (err != TAKNET_OK)
        {
          if (curr->i_cbk.cbk !=0)
            (*curr->i_cbk.cbk)(err, this, curr->i_cbk.arg);
          break;
        }
        std::cout << "[taknet] send RDMA data size: " << curr->i_rw.size << std::endl;
        err = taktuk_send( 1+_dest, TAKTUK_TARGET_ANY, curr->i_rw.lptr, curr->i_rw.size);
        /* debug mode: assert during communication */
        kaapi_assert_debug(err == TAKNET_OK);

        if (curr->i_cbk.cbk !=0)
          (*curr->i_cbk.cbk)(err, this, curr->i_cbk.arg);
      } break;
    };
    ++curr;
  }

  return;
}

} // -namespace
