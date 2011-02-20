/* KAAPI internal interface */
// =========================================================================
// (c) INRIA, projet MOAIS, 2006
// Author: T. Gautier, X. Besseron, J.N. Quintin
// Status: ok
//
//
#include <iosfwd>
#include <list>
#include <set>

#include "kanet_types.h"


namespace ka {


// --------------------------------------------------------------------
ComFailure::ComFailure( Code e )
 : std::runtime_error("ComFailure ::"), _code(e)
{ }


// --------------------------------------------------------------------
ComFailure::ComFailure(const char* msg, Code e)
 : std::runtime_error(msg), _code(e)
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
const char* ComFailure::what() const throw()
{
  if ( (_code>=0) && (_code < LAST_CODE) ) 
    return ComErrorErrorMsg[_code];
  else  
    return std::runtime_error::what();
}

//-------------------------------------------------------------------------
BadURL::BadURL( )
 : ComFailure("Bad url of object",BAD_ADDRESS) 
{}


//-------------------------------------------------------------------------
AlreadyBind::AlreadyBind( const std::string& msg)
 : ComFailure(msg.c_str(),BAD_ADDRESS) 
{}


//-------------------------------------------------------------------------
NoFound::NoFound( const std::string& msg)
 : ComFailure(msg.c_str(),HOST_NOTFOUND) 
{}

} // namespace