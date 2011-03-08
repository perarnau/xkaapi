/*
** xkaapi
** 
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
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
#include "config.h"
#include "kanet_device.h" 
#include <errno.h>
#include <string.h>
#include <map>
#if defined(__pic__) && defined(__GNUC__)
#include <dlfcn.h>
#endif
#include <iostream>
#include <sstream>

namespace ka {

// --------------------------------------------------------------------
struct string_less_compare
{
  bool operator()(const char* s1, const char* s2) const
  { return strcmp(s1, s2) < 0; }
};

static std::map<const char*,DeviceFactory*, string_less_compare> all_devicefact;


// --------------------------------------------------------------------
DeviceFactory::~DeviceFactory()
{ }


// --------------------------------------------------------------------
void DeviceFactory::destroy( Device* dev )
{ 
  if (dev !=0) {
    delete dev;
  }
}


// --------------------------------------------------------------------
int DeviceFactory::register_factory( const char* name, DeviceFactory* df )
{
  if (all_devicefact.find(name) != all_devicefact.end()) return EEXIST;
  all_devicefact.insert( std::make_pair(name, df) );
  return 0;
}


// --------------------------------------------------------------------
DeviceFactory* DeviceFactory::resolve_factory( const char* name )
{
  std::map<const char*,DeviceFactory*,string_less_compare>::const_iterator iterator = all_devicefact.find(name);
  if (iterator != all_devicefact.end()) 
    return iterator->second;

#if defined(__pic__) && defined(__GNUC__)
  /* else try to load the shared library */
  std::ostringstream sh_path;
  sh_path << "libkadev_" << name;
#if defined(__APPLE__)
  sh_path << ".0.dylib";
#elif defined(__linux__)
  sh_path << ".so";  
#else
#  warning "File extension for shared library should be defined"
#endif
  const char* c_sh_path = strdup(sh_path.str().c_str());
  void* handle = dlopen(c_sh_path, RTLD_NOW);
  if (handle ==0) 
  {
#if defined(KAAPI_DEBUG)
    std::cerr << "Cannot find shared library with name: '" << c_sh_path << "'"
              << ", error:" << dlerror() << std::endl;
#endif 
    return 0;
  }
  
  /* find the initializer C method and call it */
  std::ostringstream sh_initializator;
  sh_initializator << name << "_factory";
  void* device_init = dlsym( handle, sh_initializator.str().c_str() );
  if (device_init ==0)
  {
    std::cerr << "Device library " << c_sh_path << " does not contain initializator: '" << sh_initializator.str() << std::endl;
    return 0;
  }
  ka::DeviceFactory* (*fnc)() = (ka::DeviceFactory* (*)())device_init;
  ka::DeviceFactory* df = (*fnc)();
  register_factory(name, df );
  return df;
#else // if pic
  return 0;
#endif
}


// --------------------------------------------------------------------
Device::Device( const char* name ) 
{
  memset(_name, 0, sizeof(_name));
  if (name != 0)
    strncpy(_name, name, 31);  
}


// --------------------------------------------------------------------
Device::~Device()
{ 
}

} // - namespace Net...
