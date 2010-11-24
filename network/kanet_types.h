/* KAAPI internal interface */
// =========================================================================
// (c) INRIA, projet MOAIS, 2006
// Author: T. Gautier, X. Besseron, J.N. Quintin
// Status: ok
//
//
// =========================================================================
#ifndef _NETWORK_TYPES_H_
#define _NETWORK_TYPES_H_

#include <iosfwd>
#include "ka_types.h"
#include <list>
#include <set>

#include <sys/uio.h>

#if defined(__linux__)
#include <netinet/in.h>
#elif defined(__APPLE__)
#include <arpa/inet.h>
#elif defined(KAAPI_USE_SUNOS)
#include <sys/types.h>
#include <netinet/in.h>
#include <inttypes.h>
#endif

namespace ka {
class Channel;
class OutChannel;


// ----------------------------------------------------------------------
/** \name Protocol definition
    \author T. Gautier 
    \brief number to code protocol
    \ingroup Net
*/
enum ProtocolNumber {
  PSN_AM  = 1,
  PSN_DMA = 2,
  PSN_WS  = 3
};


// -------------------------------------------------------------------------
/** \name IOVectEntry
    \author T. Gautier 
    \brief Data structure for IOVect
    \ingroup Net
    This type should match the struct iovec type
*/
struct IOVectEntry : public iovec {
  /** Constructor of new iovect of given length
  */
  IOVectEntry( void* d =0, size_t l =0)
  {
    iov_base = (char*)d;
    iov_len  = l;
  }
};


// -------------------------------------------------------------------------
typedef void (*Callback_fnc)(int errocode, Channel* ch, void* userarg);


// -------------------------------------------------------------------------
typedef void (*Service_fnc)(int errocode, GlobalId source, void* buffer, size_t size);


// -------------------------------------------------------------------------
/*
 * inline definitions
 */
} // namespace Net

#endif
