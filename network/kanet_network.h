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
#ifndef _KANETWORK_NETWORK_H_
#define _KANETWORK_NETWORK_H_

#include "ka_error.h"
#include "kanet_types.h"
#include "kanet_device.h"
#include "kanet_channel.h"
#include <map>
#include <vector>

namespace ka {

/* Fwd decl
*/
class Device;
class InstructionStream;
class Channel;



// --------------------------------------------------------------------
/** Network is the class for network adapter interface.
    Network object has only local vision of the topology. A network object only 
    known its neighbors. Numbering if local. For instance, let us consider two 
    Network objects connected by a channel I : 
                 I
      NO1 <-------------> NO2
           3           1
    * On the object NO1, NO2 has label 3 and NO1 has, by convention label 0.
    * On the object NO2, NO1 has label 1 and NO2 has, by convention label 0.
    * The peer node for I on node NO1 is 3.
    * The peer node for I on node NO2 is 1.
    * The channel I has an identifier and several channels between NO1 and NO2
    may exists. The channel is associated to a ka::Device that manage connexion
    between node.
    On top of Network and local numbering of nodes, distributed algorithms have
    to be used to provide more convenient numbering. 


    A node can communicate to an other node using a OutChannel object. 
    Using the get_XX_route class of method, the caller can have access to such an
    OutChannel object.
*/
class Network {
public:
  /** Constructor 
  */
  Network();

  /** Destructor 
  */
  virtual ~Network();

  /** \name Network initialilization and termination
  */
  //@{
  /** First stage of initialization of a network
      Remark: All recognized arguments are deleted from the command line
      \warning On return the network is not necessarily initilized. 
      \warning A call to commit must be performed to terminate the initilialization
      This function is called should to initialize a secondary network.
      It is automatically called at initilization of the network layer for
      each secondary network.
      At the end of the first stage of initialisation, all local initializations 
      are done, but communication may failed to reach other node.
      \warning this method should be called from derived class to initialize data members
      \exception InvalidArgumentError bad parameters passed in properties
      \exception RuntimeError kind of exception is thrown in case of error      
  */
  void initialize()
    throw (RuntimeError);

  /** Second stage of initialization of a network
      Terminate the initialization process, accept incomming message and local node
      may send message to known hosts.
      \warning this method should be called from derived class to initialize data members
      \exception InvalidArgumentError bad parameters passed in properties
      \exception RuntimeError kind of exception is thrown in case of error      
  */
  void commit( )
    throw (RuntimeError);

  /** Terminate a network
    This function is automatically called at termination of the network 
    layer for each network.
  */
  void terminate () throw (RuntimeError);
  //@}


  // -----------------------------------------------------------------------
  //! \name Routing
  // -----------------------------------------------------------------------
  //@{
  /** Add a new node
      If the same route exists to this node then it is not added.
      A same channel may declared as a route to several nodes.
      This method is not reentrant and the Network object may be locked before the invocation.
      \param ni the pointer to the node info structure
      \param channel the new route to add
  */
  void add_node(GlobalId gid, const char* url);

  /** Add a route to node
      If the same route exists to this node then it is not added.
      A same channel may declared as a route to several nodes.
      This method is not reentrant and the Network object may be locked before the invocation.
      \param ni the pointer to the node info structure
      \param channel the new route to add
  */
  void add_route(GlobalId gid, OutChannel* channel);
  
  /** Delete the route to the node info using the channel
      This method is not reentrant and the Network object may be locked before the invocation.
      \param channel the route to del
  */
  void del_route(GlobalId gid, OutChannel* channel);
  
  /** Return the route (channel) to send data to local node identifier using a specific device
      This method is reentrant and the Network object is locked during the invocation.
      \param gid the remote 
      \return a pointer to the default route to reach the node or 0 if route does not exist
  */
  OutChannel* get_default_local_route( GlobalId gid );

  /** Set a channel as a default channel to the destination to the local node identifier
      This method is not reentrant and the Network object may be locked before the invocation.
      \param gid the node
      \param ch the default route to define for the node
  */
  void set_default_local_route( GlobalId gid, OutChannel* ch );

  /** Print routing routes information
  */
  std::ostream& print_route( std::ostream& o );
  
  /** Print info about nodes
  */
  std::ostream& print_info( std::ostream& o );

  /** Dump on std::cout (debug)
  */
  void dump_info();
  //@}

  // -----------------------------------------------------------------------
  //! \name Misc
  // -----------------------------------------------------------------------
  /** Return the url to be broadcast if other node want to connect to this node
  */
  const char* get_urlconnect( ) const
       throw (IOError);

  /** Return the device of the address
  */
  Device* get_device_from_url( const std::string& url_peer ) throw ();
  
  /** The network object
  */
  static Network object;

//protected:
  typedef std::set<const char*>  ListUrls; 
  
  std::map<GlobalId,ListUrls*>   _gid2urls;       ///< global to list of its url
  std::map<GlobalId,OutChannel*> _gid2channel;    ///< global to default channel lookup table
  std::map<std::string,Device*>  _name2device;    ///< map name->device
  std::vector<Device*>           _all_devices;    ///< all devices
  Device*                        _default_device; ///< default device

  /* somes counters */
public:  
  friend class IODaemon;
  friend class IOInstructionStream;
}; // -- end class network


} // end namespace 
#endif 
