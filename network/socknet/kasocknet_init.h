/* KAAPI internal interface */
// ==========================================================================
// Copyright(c)'2005 by MOAIS Team
// see the copyright file for more information
// Author: T. Gautier
// ==========================================================================
#ifndef _SOCKNET_INIT_H_
#define _SOCKNET_INIT_H_
#include "utils_component.h"

namespace SOCKNET {

// ---------------------------------------------------------------------------
/** Declaration of the component SOCKNET
*/
class Init : public Util::KaapiComponent {
public:
  static Init component;

  /** Global variable = true iff verbose mode if on for Util library level
      Default value is false.
  */
  static bool verboseon;

  /** Global variable size of the send buffer for each socket
  */
  static size_t sendbuffersize;

  /** Global variable size of the receive buffer for each socket
  */
  static size_t recvbuffersize;

  /** List of local AF_INET interface that are up and not loopback 
      first: name
      second: network address
  */
  static std::map<std::string, std::string> local_interfaces; 

  /** Default constructor: assign the priority to 20
  */
  Init();

  /** Initialization of the network library and return the default network
  */
  int initialize() throw();

  /** Terminaison of the network library 
  */
  int terminate() throw();

  /** Add the options for this Kaapi' Component
  */
  void add_options( Util::Parser* parser, Util::Properties* global_prop );

  /** Declare dependencies with other component
  */
  void declare_dependencies();
};


} // namespace

#endif
