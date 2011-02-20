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
#include "kagasnet_channel.h"
#include "kagasnet_device.h"

namespace GASNET {

// --------------------------------------------------------------------
void Device::kaapi_gasnet_service_call(gasnet_token_t token, void *buffer_am, size_t sz_buffer_am, 
  gasnet_handlerarg_t handlerH, gasnet_handlerarg_t handlerL)
{
  gasnet_node_t src;
  int err = gasnet_AMGetMsgSource( token, &src );
  kaapi_assert( err == GASNET_OK );
  uintptr_t handler = (uintptr_t)handlerH;
  handler = handler << (uintptr_t)32UL;
  handler = handler | (uintptr_t)handlerL;
  ka::Service_fnc service = (ka::Service_fnc)handler;
#if 0
  std::cout << ka::Network::object.local_gid() << "::[Device::kaapi_gasnet_service_call] recv message from:" << src << ", fnc:(" 
            << handlerH << "," << handlerL << ")=" << (void*)handler 
            << ", buffer size:" << sz_buffer_am
            << std::endl << std::flush;    
#endif
  service(0, src, buffer_am, sz_buffer_am );
}


// --------------------------------------------------------------------
void* Device::skeleton( void* arg )
{
  Device* device = (Device*)arg;
//  device->skel();
  device->_state.write( Device::S_FINISHED );
  return 0;
}


// --------------------------------------------------------------------
int Device::skel()
{
  int err;
  while (!_ack_term)
  {
    err = gasnet_AMPoll();    
    kaapi_assert( (err == GASNET_OK) );
  }
  return 0;
}


} // -namespace
