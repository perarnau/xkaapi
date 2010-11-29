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
#ifndef _GASNET_DEVICE_H_
#define _GASNET_DEVICE_H_

#include "kaapi++"
#include "kanet_device.h"
#define GASNET_PAR 1
#include <gasnet.h>

namespace GASNET {

class OutChannel;

enum {
  kaapi_gasnet_service_call_id       = 201,
};

/* Macro to check return codes and terminate with useful message. */
#define GASNET_SAFE(fncall) do {                                     \
    int _retval;                                                     \
    if ((_retval = fncall) != GASNET_OK) {                           \
      fprintf(stderr, "ERROR calling: %s\n"                          \
                      " at: %s:%i\n"                                 \
                      " error: %s (%s)\n",                           \
              #fncall, __FILE__, __LINE__,                           \
              gasnet_ErrorName(_retval), gasnet_ErrorDesc(_retval)); \
      fflush(stderr);                                                \
      gasnet_exit(_retval);                                          \
    }                                                                \
  } while(0)

// --------------------------------------------------------------------
/** \name Channel
    \ingroup Net
    A Device object should be implemented in order to be able to create Channel
    object over a specific transport protocol (tcp, myrinet, mpi, ib, taktuk,...)
    A device is dynamicly linked to the process in order to select the right device
    online : more over, if the network is not used, then the code is not loaded.
*/
class Device : public ka::Device {
public:
  /** Constructor 
  */
  Device();

  /** Destructor 
  */
  ~Device();

  /**
  */
  int initialize();

  /**
  */
  int commit();

  /** 
  */
  int terminate();

  /** 
  */
  int abort();

  /** 
  */
  ka::OutChannel* open_channel( const char* url );
  
  /**
  */
  int close_channel( ka::OutChannel* channel);

  /** 
  */
  const char* get_urlconnect( ) const;

protected:
  /** Infinite loop to read incomming message */
  int skel();

  /* */
  static void* skeleton( void* arg );

  /* */
  static void kaapi_gasnet_service_call(gasnet_token_t token, void *buffer_am, size_t sz_buffer_am, gasnet_handlerarg_t handler);
  
protected:
  int                       _wcom_rank;  ///< my rank
  int                       _wcom_size;  ///< number of nodes
  ka::atomic_t<32>          _state;      ///< state of the device
  volatile int              _ack_term;   ///< On node 0
  pthread_t                 _tid;        ///< Thread that wait incomming message
  
  enum DeviceState {
    S_CREATE,
    S_INIT,
    S_TERM,
    S_FINISHED,
    S_ERROR
  };
}; // -- end class device


} // end namespace 
#endif 
