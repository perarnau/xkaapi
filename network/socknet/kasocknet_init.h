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

// --------------------------------------------------------------------
// --------------------------------------------------------------------
/** Design Pattern: factory of device over socknet
*/
class DeviceFactory : public ka::DeviceFactory {
public:
  /**
  */
  DeviceFactory();
  
  /** Virtual dstor 
  */
  ~DeviceFactory();
  
  /** Return the name of the type of created network 
  */
  const char* get_name() const;
  
  /** Return a new device
  */
  ka::Device* create(int* argc, char*** argv);
};


} // namespace

/* exported name for the shared library */
extern "C" ka::DeviceFactory* socknet_factory();

#endif
