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
#include "ka_error.h"
#include "ka_debug.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#if not defined(KAAPI_USE_ARCH_PPC)
#include <execinfo.h>
#endif
#include <cxxabi.h>
#include <cstring>

namespace ka {

// --------------------------------------------------------------------
void print_backtrace_c()
{
#if not defined(KAAPI_USE_ARCH_PPC)
  const unsigned int MAX_DEPTH = 100;
  void *trace[MAX_DEPTH];
  unsigned int trace_size;
  char **trace_strings;

  trace_size = backtrace(trace, MAX_DEPTH);
  trace_strings = backtrace_symbols(trace, trace_size);
  for (unsigned int i=0; i<trace_size; ++i)
  {  
    logfile() << std::setfill(' ') << std::right << std::setw(3) << i << ": "
        << trace_strings[i] << std::endl;
  }
  free(trace_strings); // malloc()ed by backtrace_symbols
#endif
}
  
// --------------------------------------------------------------------
void print_backtrace_cpp()
{
#if not defined(KAAPI_USE_ARCH_PPC)
  const unsigned int MAX_DEPTH = 100;
  void *trace[MAX_DEPTH];
  unsigned int trace_size=0;
  char **trace_strings;

  trace_size = backtrace(trace, MAX_DEPTH);
  trace_strings = backtrace_symbols(trace, trace_size);

  logfile() << "Call stack: \n" << std::endl;

  for (unsigned int i=0; i<trace_size; ++i)
  { 
    size_t function_size = 200;
    char *function = static_cast<char*>(malloc(function_size));
    char *begin = 0, *end = 0;

    // find the parentheses and address offset surrounding the mangled name
    for (char *j = trace_strings[i]; *j; ++j) {
      if (*j == '(') 
      {
        begin = j;
      }
      else if (*j == '+') 
      {
        end = j;
      }
    }
    
    if (begin && end) 
    { // found mangled name 
      *begin = '\0';
      begin++;
      *end = '\0';
      
      // found our mangled name, now in [begin, end)
      int status;
      char *ret = abi::__cxa_demangle(begin, function, &function_size, &status);
      if (ret) {
        // return value may be a realloc() of the input
        function = ret;
      }
      else 
      {
        // demangling failed, just pretend it's a C function with no args
        std::strncpy(function, begin, function_size);
        std::strncat(function, "()", function_size);
        function[function_size-1] = ' ';
      }
      logfile() << std::setfill(' ') << std::right << std::setw(3) << i << ": "
          << function << " in " << trace_strings[i] << std::endl;
    }
    else
    {
      // didn't find the mangled name, just print the whole line
      logfile() << std::setfill(' ') << std::right << std::setw(3) << i << ": "
          << trace_strings[i] << std::endl;
    }
    free(function);     
  }  
#endif
}
  
// --------------------------------------------------------------------
void backtrace_sighandler(int sig, siginfo_t *info, void *secret) 
{
  switch( sig )
  {
    case SIGILL: 
      logfile() << "Got signal SIGILL: Illegal Instruction" << std::endl;
      break;
    case SIGABRT: 
      logfile() << "Got signal SIGABRT: Abort" << std::endl;
      break;
    case SIGFPE: 
      logfile() << "Got signal SIGFPE: Floating point exception" << std::endl;
      break;
    case SIGSEGV: 
      logfile() << "Got signal SIGSEGV: Invalid memory reference" << std::endl;
      break;
    case SIGBUS: 
      logfile() << "Got signal SIGBUS: Bus error (bad memory access)" << std::endl;
      break;
    default: 
      logfile() << "Got signal " << sig << std::endl;
      break;    
  }
  print_backtrace_cpp();
  raise( sig );
}
  
// --------------------------------------------------------------------
void System_abort( const std::string& msg )
{
  logfile() << msg << std::endl;
  logfile() << "Abort" << std::endl;
  std::cerr << "Abort" << std::endl;
  abort(); // signal SIGABRT, exit and create a core (if allowed)
}

// --------------------------------------------------------------------
Exception::Exception() 
 : _code(0) 
{}


// --------------------------------------------------------------------
Exception::Exception(int code) 
 : _code(code) 
{}


// --------------------------------------------------------------------
Exception::~Exception() 
{}


// --------------------------------------------------------------------
std::ostream& Exception::print( std::ostream& o ) const
{ 
  return o << this->what(); 
}


// --------------------------------------------------------------------
const char* Exception::what () const
{ return "Unknown exception"; }


// --------------------------------------------------------------------
int Exception::code() const
{ return _code; }


// --------------------------------------------------------------------
RestartException::RestartException()
 : Exception(0)
{}

// --------------------------------------------------------------------
ServerException::ServerException()
 : Exception(0)
{}

// --------------------------------------------------------------------
RuntimeError::RuntimeError(const std::string& msg, int code) 
 : Exception(code), _msg(msg)
{}


// --------------------------------------------------------------------
const char* RuntimeError::what () const
{
  static char tmp[1024];
  sprintf(tmp,"%s, code=%i", _msg.c_str(), _code);
  return tmp;
}

// --------------------------------------------------------------------
PosixError::PosixError( const std::string& msg, int code )
 : RuntimeError(msg,code) {}

// --------------------------------------------------------------------
const char* PosixError::what () const
{
  static char tmp[1024];
  sprintf(tmp,"[Msg=%s, Posix code=%i, Posix msg='%s']", 
  _msg.c_str(), _code, strerror(_code) );
  return tmp;
}


// --------------------------------------------------------------------
#if defined(KAAPI_DEBUG)
void __Exception_throw(  const Exception& err ) 
{ 
 // std::cout << "[atha::throw] called:" << err.what() << std::endl;
}
#else
void __Exception_throw(  const Exception& ) 
{ 
}
#endif


// --------------------------------------------------------------------
RangeError::RangeError(const std::string& msg)
 : RuntimeError(msg)
{}

// --------------------------------------------------------------------
LogicError::LogicError(const std::string& msg, int code) 
 : Exception(code), _msg(msg) 
{}

// --------------------------------------------------------------------
const char* LogicError::what () const
{ return _msg.c_str(); }


// --------------------------------------------------------------------
InvalidArgumentError::InvalidArgumentError(const std::string& msg, int code)
 : LogicError(msg, code) 
{}

// --------------------------------------------------------------------
LengthError::LengthError(const std::string& msg)
 : LogicError(msg) 
{}

// --------------------------------------------------------------------
OutOfRangeError::OutOfRangeError(const std::string& msg)
 : LogicError(msg) 
{}

// --------------------------------------------------------------------
NoMoreResource::NoMoreResource(const std::string& msg)
 : LogicError(msg) 
{}

// --------------------------------------------------------------------
BadAlloc::BadAlloc(const std::string& )
 : NoMoreResource() 
{}


// --------------------------------------------------------------------
IOError::IOError( const std::string& msg, int code) 
 : RuntimeError(msg,code) 
{}

// --------------------------------------------------------------------
EndOfFile::EndOfFile()
 : IOError("End of file", 0)
{}

// --------------------------------------------------------------------
ComFailure::ComFailure( Code e )
 : IOError("ComFailure ::", e)
{ }


// --------------------------------------------------------------------
ComFailure::ComFailure(const char* msg, Code e)
 : IOError(msg,e)
{}


//-------------------------------------------------------------------------
static const char*  ComErrorErrorMsg []= {
  "OK", 
  "FAIL ",
  "ABORT ",
  "TRUNCATED ",
  "NOT_FILLED ",
  "FATAL ",
  "NOROUTE ",
  "NULL_SERVICE ",
  "BAD OR NULL REQUEST  ",
  "PANIC ",
  "NOT_IMPLEMENTED ",
  "BAD_ADDRESS ",
  "NO_CONNECTION ",
  "BAD_PROTOCOL ",
  "TIMEOUT "
  "ERR_PIPE ",
  "HOST_UNREACH ",
  "HOST_NOTFOUND ",
  "HOST_NOTADDRESS "
};


//-------------------------------------------------------------------------
const char* ComFailure::what() const
{
  if ( (Code(_code)>=0) && (Code(_code) < LAST_CODE) ) 
    return ComErrorErrorMsg[_code];
  else  
    return "UNDEFINED COMFAILURE ERROR CODE ";  
}


//-------------------------------------------------------------------------
BadURL::BadURL( )
 : IOError("Bad url of object") 
{}


//-------------------------------------------------------------------------
AlreadyBind::AlreadyBind( const std::string& msg)
 : InvalidArgumentError( msg ) 
{}

//-------------------------------------------------------------------------
NoFound::NoFound( const std::string& msg)
 : InvalidArgumentError( msg ) 
{}


} // - namespace
