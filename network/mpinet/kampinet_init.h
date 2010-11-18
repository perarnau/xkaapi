/* KAAPI internal interface */
// ==========================================================================
// Copyright(c)'2005 by MOAIS Team
// see the copyright file for more information
// Author: T. Gautier
// ==========================================================================
#ifndef _SOCKNET_INIT_H_
#define _SOCKNET_INIT_H_

#include "kampinet_device.h"

namespace MPINET {

// --------------------------------------------------------------------
/** Design Pattern: factory of device over mpinet
*/
class DeviceFactory : public ka::DeviceFactory {
public:
  /** Virtual dstor 
  */
  ~DeviceFactory();
  
  /** Return the name of the type of created network 
  */
  const char* get_name() const;
  
  /** Return a new device
  */
  ka::Device* create();
};


} // namespace

/* exported name for the shared library */
extern "C" ka::DeviceFactory* mpinet_factory();

#endif
