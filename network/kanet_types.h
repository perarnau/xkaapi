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

#include <stdexcept>
#include "ka_types.h"     // use only atomic

namespace ka {
class Channel;
class OutChannel;

// -------------------------------------------------------------------------
/** Compatibility with kaapi type kaapi_globalid_t
*/
typedef uint32_t GlobalId;

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
typedef uint8_t ServiceId;


// -------------------------------------------------------------------------
struct SegmentInfo {
  SegmentInfo() : segaddr(0), segsize((size_t)-1) {}
  SegmentInfo(uintptr_t addr, size_t size) : segaddr(addr), segsize(size) {}
  uintptr_t   segaddr;    /* base address allocation */
  size_t      segsize;    /* size of address space */
};


// --------------------------------------------------------------------
/** IOComFailure: 
*/
class ComFailure: public std::runtime_error {
public:
  /// Error Code
  enum  Code {
    /// No error return value
    OK  = 0,
    ///
    FAIL,
    ///
    ABORT,
    ///
    TRUNCATED,
    ///
    NOT_FILLED,
    ///
    FATAL,
    ///
    NOROUTE,
    ///
    NULL_SERVICE,
    ///
    PANIC,
    ///
    BAD_REQUEST,
    ///
    NOT_IMPLEMENTED,
    ///
    BAD_ADDRESS,
    ///
    NO_CONNECTION,
    ///
    BAD_PROTOCOL,
    /// operation has reach a timedout before finishing
    TIMEOUT,
    /// Socket is closed or was closed during operation
    ERR_PIPE,
    /// Host unreachable
    HOST_UNREACH,
    /// Host is not found 
    HOST_NOTFOUND,
    /// Host is found but has no associated address
    HOST_NOTADDRESS,
    /// Asynchronous post, used by asynchronous I/O
    ASYNC_POST,
    /// 1 + last error code
    LAST_CODE
  };
public:
  /// Cstor
  ComFailure( Code e );

  /// Cstor
  ComFailure(const char* msg, Code e=OK);

  /// Return a string about the exception code
  const char* what() const throw();

private:
  Code _code;
};




// --------------------------------------------------------------------
/** Bad url
*/
class BadURL: public ComFailure {
public:
  BadURL( );
};


// --------------------------------------------------------------------
/** AlreadyBind
*/
class AlreadyBind: public ComFailure {
public:
  AlreadyBind( const std::string& msg = "object is already bind");
};


// --------------------------------------------------------------------
/** NoFound
*/
class NoFound: public ComFailure {
public:
  NoFound( const std::string& msg = "name no found");
};


// -------------------------------------------------------------------------
/*
 * inline definitions
 */
} // namespace Net

#endif
