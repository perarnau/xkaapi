// ==========================================================================
// Copyright(c)'2005 by MOAIS Team
// see the copyright file for more information
// Author: T. Gautier
// ==========================================================================
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>

#include "kampinet_init.h" 


// --------------------------------------------------------------------
extern "C" ka::DeviceFactory* mpinet_factory()
{
  static MPINET::DeviceFactory theFactoryObject;
  return &theFactoryObject;
}

namespace MPINET {

// --------------------------------------------------------------------
DeviceFactory::~DeviceFactory()
{ }


// --------------------------------------------------------------------
const char* DeviceFactory::get_name() const
{ return "mpinet"; }


// --------------------------------------------------------------------
ka::Device* DeviceFactory::create( )
{ 
  Device* dev = new MPINET::Device();
  if (dev->initialize() !=0) 
  {
    delete dev;
    return 0;
  }
  return dev;
}

} // namespace
