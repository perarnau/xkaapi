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
#ifndef _NETWORK_DEVICE_H_
#define _NETWORK_DEVICE_H_

#include "kanet_types.h"

namespace ka {

// --------------------------------------------------------------------
/** Fwd decl
*/
class OutChannel;
class Device;


// --------------------------------------------------------------------
/** Design Pattern: factory of network
    The factory is based on using shared library loaded and unloaded at
    runtime.
    The name of the device directly defines the name of the shared library.
    
    The device implementation is loaded upon the resolve factory call:
    - either a factory was registered to the device factory
    - either the factory was not registered and the runtime will try to
    load the shared library with name "libkadev_<device name>.so".
    
*/
class DeviceFactory {
public:
  /** Virtual dstor 
  */
  virtual ~DeviceFactory();
  
  /** Return the name of the type of created device 
  */
  virtual const char* get_name() const = 0;
  
  /** Load .so and return a new device 
  */
  virtual Device* create(int* argc, char*** argv) =0;

  /** Destroy a created device 
  */
  virtual void destroy( Device* );

  /** Register device factory
     \retval 0 in case of successful call
     \retval EEXIST if the name is already bind to a factory
  */
  static int register_factory( const char* name, DeviceFactory* df );

  /** Resolve device factory 
  */
  static DeviceFactory* resolve_factory( const char* name );
protected:
};


// --------------------------------------------------------------------
/** \name Channel
    \ingroup Net
    A Device object should be implemented in order to be able to create Channel
    object over a specific transport protocol (tcp, myrinet, mpi, ib, taktuk,...)
    A device is dynamicly linked to the process in order to select the right device
    online : more over, if the network is not used, then the code is not loaded.
*/
class Device {
public:
  /** Constructor 
  */
  Device( const char* name );

  /** Destructor 
  */
  virtual ~Device();

  /** \name Device initialilization and termination
  */
  //@{
  /** Initialization of the device
      \exception InvalidArgumentError bad parameters passed in properties
      \exception RuntimeError kind of exception is thrown in case of error      
  */
  virtual int initialize(int* argc, char*** argv) = 0;

  /** Second stage of initialization of a device
      Terminate the initialization process, accept incomming message and local node
      may send message to known hosts.
  */
  virtual int commit() = 0;

  /** Terminate the device
    This function is automatically called at the normal termination of the network.
  */
  virtual int terminate() = 0;

  /** Abort the device
    This function is automatically called in case of abortion of the process
  */
  virtual int abort() = 0;
  //@}

  // -----------------------------------------------------------------------
  //! \name Management of Device
  // -----------------------------------------------------------------------
  //@{
  /** Allocate memory for rdma operation through this device.
      \param size the size in byte of the request memory region
      \return a pointer to a memory region that contains at least size bytes.
      \return 0 iff no memory of that size can be allocated
      \see deallocate
  */
  virtual void* allocate( size_t size ) =0;
  
  /** Deallocate a memory region allocated by 'allocate'
      \param addr to the memory to deallocate
      \see allocate
  */
  virtual void deallocate( void* addr ) =0;

  /** Translate an address of a logical address space to the virtual address space
      \param the address in the logical address space
      \param size the size in byte of the request memory region
      \return a pointer to a memory region that contains at least size bytes.
      \return 0 iff no memory of that size can be allocated
      \see deallocate
  */
  virtual void* bind( uintptr_t addr, size_t size ) = 0;


  /** Return the segment info for the RDMA memory on gid 
  */  
  virtual SegmentInfo get_seginfo( GlobalId gid ) const =0;
  //@}

  //@{
  /** Open a channel to a given url
      \param gid of the node on which to open channel
      \return channel to the node or 0 if route cannot be open.
      \see close_channel
  */
  virtual OutChannel* open_channel( const char* url ) =0;
  
  /** Delete the channel
      The method is called to close the channel after it all references to it have been
      released on the network level.
      \param channel the channel to delete
      \see open_channel
  */
  virtual int close_channel(OutChannel* channel) =0;

  /** Poll for incomming message.
  */
  virtual void poll() = 0;

  /** Global barrier
  */
  virtual void barrier() = 0;
  //@}

  // -----------------------------------------------------------------------
  //! \name Misc
  // -----------------------------------------------------------------------
  /** Return the url to be broadcast if other node want to connect to this node
      using this specific device.
  */
  virtual const char* get_urlconnect( ) const = 0;
  
  /** Return the name of the device
  */
  const char* get_name() const;

  /** Infinite loop to read incomming message */
  virtual int skel() =0;

protected:
  char  _name[32];             ///< C-name of the device
}; // -- end class device



inline const char* Device::get_name() const
{ return _name; }

} // end namespace 
#endif 
