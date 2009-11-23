/* KAAPI public interface */
// =========================================================================
// (c) INRIA, projet MOAIS, 2006-2009
// Author: T. Gautier
//
//
//
// =========================================================================
#ifndef _ATHA_INIT_H
#define _ATHA_INIT_H

#include "atha_types.h"
#include "atha_properties.h"
#include "atha_parser.h"
#include "atha_component.h"

namespace atha {

class Init : public KaapiComponent {
public:
  /** The component object for the util package
  */
  static Init component;

  /** Global variable = true iff verbose mode if on for Util library level
      Default value is false.
  */
  static bool verboseon;

  /** Global variable = true iff trace logging is enable
      Default value is false.
  */
  static bool enable_trace;

  /** Global variable that defines the architecture for this processor
  */
  static Architecture local_archi;

  /** The global id of this process
  */
  static kaapi_uint32_t local_gid;

  /** Should be defined for internal purpose only...
  */
  static bool on_term;
  static bool on_thread;
  
  /** Default constructor: assign the most prioritary number (0) for this component
  */
  Init();

  /** Initialize the default attributs for Mutex, Condition and Thread
      Should be called before any use of Util functions or methods.
      \return the error code
  */
  int initialize() throw();

  /** Terminate the use of the Util library
      \return the error code
  */
  int terminate() throw();

  /** Add the options for this Kaapi' Component
  */
  void add_options( Parser* parser, Properties* global_prop );

  /** Declare dependencies with other component
  */
  void declare_dependencies();

};


} // namespace Init
#endif
