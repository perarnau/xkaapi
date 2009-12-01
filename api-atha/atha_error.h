/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** xavier.besseron@imag.fr
** vincent.danjean@imag.fr
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
#ifndef _ATHA_ERROR_H_
#define _ATHA_ERROR_H_

#include "kaapi.h"
#include <iosfwd>
#include <sstream>
#include <string>
#include <exception>

extern "C" {
#include <errno.h>
#include <signal.h>
#include <string.h>   // - strerror
}

namespace atha {

class Exception;

/** \defgroup Exception
    \ingroup atha
    \brief Definition of exceptions in KAAPI
*/
//@{

  /** Function call by KAAPI_ASSERT before throwing a Kaapi exception
      The main goal of this function is to be able to capture and exception
      during debuging the program. All exception throwed by Kaapi make a call
      to this function.
      \see KAAPI_ASSERT
  */ 
  void __Exception_throw( const Exception& err );

  /** Function call by macro to throw a Kaapi exception
      \see KAAPI_ASSERT
  */ 
  template<class T>
  void Exception_throw( const T& ex ) throw(T);

// --------------------------------------------------------------------
/** \brief Base class for the KAAPI's exception hierarchy
    Base class for hierarchy of error classes.
    Match the ANSI/C++ proposal for the hierarchy.
*/
class Exception {
public:
  /** Default constructor 
   */
  Exception();
  
  /** Default constructor 
   */
  Exception(int code);

  /** Destructor 
   */
  virtual ~Exception();

  /** Output message to stream
      @param o an output stream
   */
  virtual std::ostream& print( std::ostream& o ) const;

  /** Return an human readable string for the error message
   */ 
  virtual const char* what () const;

  /** Return error code if any
  */
  int  code() const;

protected:
  int _code;   // - some error code, 0 is always OK
};

/** Function to throw an exception.
    The purpose of this method is to be able to set a breakpoint on throw 
    \param ex a reference to an Exception object
    \exception the err object
*/
template<class T>
void Exception_throw( const T& ex ) throw(T)
{ 
  __Exception_throw(ex);
  throw ex;
}


// --------------------------------------------------------------------
/** RestartException
  RestartException. An object of this class is throwed by Community::initialize 
  if a restart mode of execution is selected.
*/
class RestartException: public Exception {
public:
  /**
  */
  RestartException(); 
protected:
};


// --------------------------------------------------------------------
/** ServerException
  ServerException. An object of this class is throwed by Community::initialize 
  if the process run only as a checkpoint server and do not participate 
  to the computation.
*/
class ServerException: public Exception {
public:
  /**
  */
  ServerException(); 
protected:
};


// --------------------------------------------------------------------
/** RuntimeError
RuntimeError. An object of this class is throwed if
an un-previsible runtime error occurs.
*/
class RuntimeError: public Exception {
public:
  /**
  */
  RuntimeError(const std::string& msg = "", int code =0); 

  /** Return textual message of the exeception
  */
  const char* what() const;
  
protected:
  std::string _msg;
};


// --------------------------------------------------------------------
/** Posix. 
  To provide an envelop as exception for certain posix error
*/
class PosixError: public RuntimeError {
public:
  /** 
  */
  PosixError( const std::string& msg, int code );

  /** Return textual message of the exeception 
  */
  const char* what () const;
protected:
};


// --------------------------------------------------------------------
/** RangeError. An object of this class is throwed if
    an un-previsible runtime error occurs while checking a 
    domain for a value.
*/
class RangeError: public RuntimeError {
public:
  RangeError(const std::string& msg = "");
};


// --------------------------------------------------------------------
/** Logic. An object of this class is throwed for
    error that could be, in principle, detected at compilation.
*/
class LogicError: public Exception {
public:
  LogicError(const std::string& msg, int code =0); 
  const char* what () const;
protected:
  std::string _msg;
};


// --------------------------------------------------------------------
/** InvalidArgumentError. An object of this class is throwed if
    the argument of a call to a function is invalid.
*/
class InvalidArgumentError: public LogicError {
public:
  InvalidArgumentError(const std::string& msg, int code =0 );
};


// --------------------------------------------------------------------
/** LengthError. 
*/
class LengthError: public LogicError {
public:
  LengthError(const std::string& msg = "");
};


// --------------------------------------------------------------------
/** OutOfRangeError. 
*/
class OutOfRangeError: public LogicError {
public:
  OutOfRangeError(const std::string& msg = "");
};


// --------------------------------------------------------------------
/** NoMoreResource. 
*/
class NoMoreResource: public LogicError {
public:
  NoMoreResource(const std::string& msg = "");
};                                         


// --------------------------------------------------------------------
/** BadAlloc. 
*/
class BadAlloc: public NoMoreResource {
public:
  BadAlloc(const std::string& msg = "");
};


// --------------------------------------------------------------------
/** IO. Base class for input/ouput communication capability errors. 
*/
class IOError: public RuntimeError{
public:
  IOError( const std::string& msg = "", int code =0 );
};                                         

// --------------------------------------------------------------------
/** End of File exception
*/
class EndOfFile: public IOError{
public:
  EndOfFile();
};                                         


// --------------------------------------------------------------------
/** IOComFailure: 
*/
class ComFailure: public IOError {
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
  const char* what() const;
};


// --------------------------------------------------------------------
/** Bad url
*/
class BadURL: public IOError {
public:
  BadURL( );
};


// --------------------------------------------------------------------
/** AlreadyBind
*/
class AlreadyBind: public InvalidArgumentError {
public:
  AlreadyBind( const std::string& msg = "object is already bind");
};


// --------------------------------------------------------------------
/** NoFound
*/
class NoFound: public InvalidArgumentError {
public:
  NoFound( const std::string& msg = "name no found");
};

//@}

// --------------------------------------------------------------------
/** Signal handler for SIGSEGV, SIGBUS, SIGFPE, SIGABRT, SIGILL
 *  Print a backtrace
 */
void backtrace_sighandler(int sig, siginfo_t *info, void *secret);

} // - namespace atha...



#endif
