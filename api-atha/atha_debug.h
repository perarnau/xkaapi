/* KAAPI public interface */
// =========================================================================
// (c) INRIA, projet MOAIS, 2006-2009
// Author: T. Gautier, X. Besseron, V. Danjean
//
//
//
// =========================================================================
#ifndef _ATHA_DEBUG_H_
#define _ATHA_DEBUG_H_

#ifdef KAAPI_COMPILE_SOURCE
   // Only include kaapi_config.h when building kaapi itself
   // (to know if KAAPI_DEBUG is defined or not)
#  include "kaapi_config.h"
#endif
#include <string>

/** \namespace atha
    \brief The atha namespace contains definitions to port the library onto an operating system
*/
namespace atha {

#ifndef KAAPI_DEBUG_MEM
#  define _kaapi_malloc( s) ::malloc(s)
#  define _kaapi_calloc( s1, s2) ::calloc(s1,s2)
#  define _kaapi_free( p ) ::free(p)
#else
#  define _kaapi_malloc( s) __kaapi_malloc(s)
#  define _kaapi_calloc( s1, s2) __kaapi_calloc(s1,s2)
#  define _kaapi_free( p ) __kaapi_free(p)
  extern void* __kaapi_malloc( size_t s);
  extern void* __kaapi_calloc( size_t s1, size_t s2);
  extern void  __kaapi_free( void* p );
#endif

// --------------------------------------------------------------------
  /** \brief Abort the process with a display the given message
      \ingroup atha
  */
  extern void System_abort( const std::string& msg );

  /** Log file for this process
      \ingroup atha
      If not set_logfile, then return the std::cout stream.
  */
  extern std::ostream& logfile();
  
  /**
  */
  extern void initialize_logfile();

  /**
  */
  extern void lock_logfile();

  /**
  */
  extern void unlock_logfile();
  
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

// -----------------------------------------------------------------------
/*! \def KAAPI_XVALTOSTR(msg)
    \ingroup atha
    \brief Helper for macro KAAPI_VALTOSTR
*/
#define KAAPI_XVALTOSTR(msg)     \
  #msg

/*! \def KAAPI_VALTOSTR(msg)
    \ingroup atha
    \brief Macro to convert value to a string
*/
#define KAAPI_VALTOSTR(msg)     \
  KAAPI_XVALTOSTR(msg)

/*! \def KAAPI_SUFX(msg)
    \ingroup atha
    \brief Append file name and line number to the argument string
*/
#define KAAPI_SUFX(msg)  \
  msg"\tFile:" __FILE__ ", Line:"KAAPI_VALTOSTR(__LINE__)

/*! \def KAAPI_ASSERT(cond,msg)
    \ingroup atha
    \brief Assertion which display message before calling System_abort if cond is failed
*/
#define KAAPI_ASSERT_M(cond, msg)       \
    if (!(cond)) {       \
       ::std::ostringstream str;\
       str << msg << "\tFile:" << __FILE__ << ", Line:" << KAAPI_VALTOSTR(__LINE__);\
       atha::System_abort( str.str() );\
    }

/*! \def KAAPI_ASSERT(cond, msg)
    \ingroup atha
    \brief Assertion which throw exception ex if cond is not equal to true
*/
#define KAAPI_ASSERT(cond, ex) \
    if (!(cond)) {       \
       ::std::ostringstream str;\
       str << ex.what() << "\tFile:" << __FILE__ << ", Line:" << KAAPI_VALTOSTR(__LINE__);\
       atha::System_abort( str.str() );\
    }


/*! \def KAAPI_LOG(cond, msg)
    \ingroup atha
    \brief Log message iff the condition cond is true
*/
#if defined(KAAPI_DEBUG)
#define KAAPI_CPPLOG(cond, msg) \
  if ((cond)) {\
    Util::logfile() << msg << ::std::endl;\
  }

/*! \def KAAPI_LOG_INST(cond, inst)
    \ingroup atha
    \brief Dot instruction iff the condition cond is true
*/
#define KAAPI_CPPLOG_INST(cond, inst) \
  if ((cond)) {\
    inst;\
  }
#else
#define KAAPI_CPPLOG(cond, msg) 
#define KAAPI_CPPLOG_INST(cond, inst) 
#endif

/*! \def KAAPI_DEBUG
    \ingroup atha
    \brief To compile library with enabling some code for debugging
*/

/*! \def KAAPI_ASSERT_DEBUG(cond, ex)
    \ingroup atha
    \brief Equivalent to the macro KAAPI_ASSERT iff KAAPI_DEBUG is defined else do nothing
*/

/*! \def KAAPI_ASSERT_DEBUG_M(cond, msg)
    \ingroup atha
    \brief Equivalent to the macro KAAPI_ASSERT_M iff KAAPI_DEBUG is defined
*/
#ifndef KAAPI_DEBUG
#  define KAAPI_PRINT(expr)       
#  define KAAPI_ASSERT_DEBUG(cond, ex) 
#  define KAAPI_ASSERT_DEBUG_M(cond,msg)

#else /* KAAPI_DEBUG */

#  define KAAPI_PRINT(expr)  \
  expr;       
#  define KAAPI_ASSERT_DEBUG(cond, ex) \
  KAAPI_ASSERT(cond,ex)
#  define KAAPI_ASSERT_DEBUG_M(cond,msg) \
  KAAPI_ASSERT_M(cond,msg)
#endif /* KAAPI_DEBUG */

} // namespace

#endif
