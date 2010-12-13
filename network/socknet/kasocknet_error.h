/* KAAPI internal interface */
// ==========================================================================
// (c) INRIA, projet MOAIS, 2006
// Author: T. Gautier
// - based on socket class from Inuktitut2 & 3, author : C. Martin and T. Gautier
// ==========================================================================
#ifndef _KAAPI_SN_NETERROR_H
#define _KAAPI_SN_NETERROR_H

#include "ka_error.h"

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <netdb.h>
#include <netinet/in.h>

namespace SOCKNET {

/* projection into SOCKNET */
using ka::IOError;
using ka::RuntimeError;
using ka::ComFailure;
using ka::InvalidArgumentError;
using ka::OutOfRangeError;
using ka::BadAlloc;

/* fwd declaration */
class SocketClient;

/* fwd declaration */
class SocketServer;

/** Base class for SOCKNET exception 
*/
class SockError : public ka::PosixError {
public:
  SockError ( int err, const char* msg ="SOCKNET Error", const std::string& host = "" )
   : ka::PosixError(msg, err), _host(host)
  {}
  ~SockError() throw()
  { }
  const char * what( ) const
  { return hstrerror(_code);}
protected:
  std::string _host;
};


} /* namespace */
#endif
