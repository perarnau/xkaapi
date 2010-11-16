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
#ifndef _MPINET_DEVICE_H_
#define _MPINET_DEVICE_H_

#include "../kanet_device.h"
#include <map>

namespace MPINET {

class OutChannel;

// --------------------------------------------------------------------
/** Design Pattern: factory of device over mpinet
*/
class DeviceFactory : public Net::DeviceFactory {
public:
  /** Virtual dstor 
  */
  ~DeviceFactory();
  
  /** Return the name of the type of created network 
  */
  const char* get_name() const;
  
  /** Return a new device
  */
  Net::Device* create();

  /** Destroy a created network 
  */
  void destroy( Net::Device* );
};



// --------------------------------------------------------------------
/** \name Channel
    \ingroup Net
    A Device object should be implemented in order to be able to create Channel
    object over a specific transport protocol (tcp, myrinet, mpi, ib, taktuk,...)
    A device is dynamicly linked to the process in order to select the right device
    online : more over, if the network is not used, then the code is not loaded.
*/
class Device : public Net::Device {
public:
  /** Constructor 
  */
  Device();

  /** Destructor 
  */
  ~Device();

  /**
  */
  int initialize();

  /**
  */
  int commit();

  /** 
  */
  int terminate();

  /** 
  */
  int abort();

  /** 
  */
  Net::OutChannel* open_channel( const char* url );
  
  /**
  */
  int close_channel( Net::OutChannel* channel);

  /** 
  */
  const char* get_urlconnect( ) const;

protected:
  /** Infinite loop to read incomming message */
  int skel();

protected:
  int   _wcom_rank;                       ///< my rank
  int   _wcom_size;                       ///< communicator size
  std::map<int,OutChannel*> _openchannel; ///< route to node rank i
}; // -- end class device


} // end namespace 
#endif 
