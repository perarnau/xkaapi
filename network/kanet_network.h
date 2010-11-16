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

namespace Net {

/* Fwd decl
*/
class Device;
class IOInstructionStream;
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
    may exists. The channel is associated to a Net::Device that manage connexion
    between node.
    On top of Network and local numbering of nodes, distributed algorithms have
    to be used to provide more convenient numbering. 
    
    Default properties:
    * net.blocsize = <integer>;      // the size in bytes for bloc allocation
*/
class Network {
public:
  /** Constructor 
  */
  Network();

  /** Destructor 
  */
  ~Network();

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


  // ----------------------------------------------------------------------
  //! \name Properties and name management
  // ----------------------------------------------------------------------
  //@{
  /** Get the name of the network.
      This method equivalent to calling 'get_property("name")'.
  */
  const std::string& get_name( ) const;

  /** Set the name of the network. 
      This method equivalent to calling 'set_property("name")'.
  */
  const std::string& set_name(const std::string& name );
  //@}


  // -----------------------------------------------------------------------
  //! \name Routing
  // -----------------------------------------------------------------------
  //@{
  /** Typedef
  */
  typedef ::Net::NodeInfo NodeInfo;
  
  /** iterator over known nodes
  */
  typedef std::vector<NodeInfo*>::iterator iterator_node;

  /** iterator over known nodes
  */
  typedef std::vector<NodeInfo*>::const_iterator const_iterator_node;
  
  /** iterator over declared routes
  */
  typedef std::vector<OutChannel*>::iterator iterator_route;

  /** iterator over declared routes
  */
  typedef std::vector<OutChannel*>::const_iterator const_iterator_route;

  /** Return the node information from its local identifier
      The method allocates a new local id for the given node if it is not known.
      This method is not reentrant and the Network object may be locked before the invocation.
      \param lid the local identifier of the node
      \return the pointer to the node info structure if the node is known or 0
  */
  NodeInfo* localid_to_nodeinfo(ka_uint16_t lid);
  
  /** Return the node information from its global identifier
      This method is not reentrant and the Network object may be locked before the invocation.
      \param gid the global id of the node
      \return the pointer to the node info structure if the node is known or 0
  */
  NodeInfo* globalid_to_nodeinfo(GlobalId gid);

  /** Return the node information from its global identifier
   *  This method lock the Network object.
   *  If not found, try it will ask to the nameserver.
   *  \param gid the global id of the node
   *  \return the pointer to the node info structure if the node is known or 0
   */
  Network::NodeInfo* resolve_nodeinfo(GlobalId gid);
  
  /** Return the cluster object with clustername
      If the object does not exist allocate it and register it.
      This method is not reentrant and the Network object may be locked before the invocation.
      \param clustername the name of the cluster
      \return the node info of the new added node with a local id
  */
  Cluster* resolve_cluster( const std::string& clusterpath );

  /** Register the cluster object with its clusterpath
      This method is not reentrant and the Network object may be locked before the invocation.
      \param clusterpath the name of the cluster
      \return the node info of the new added node with a local id
  */
  void register_cluster( Cluster* cluster, const std::string& clusterpath );

  /** Add a new node with its global identifier and url to connect
      If the global id as already been declared, the return value is the previously delcared node.
      This method is not reentrant and the Network object may be locked before the invocation.
      \param gid the global identifier of the new node
      \param url_peer the url of the added node
      \return the node info of the new added node with a local id
  */
  NodeInfo* add_node(GlobalId gid, const std::string& url_peer, const std::string& cluster);

  /** Del a node from the routing tables
      The invocation call supress all known route to this node.
      The channels associated to the routes are not close neither deallocated.
      The node info data structure should be deallocated using delete.
      This method is not reentrant and the Network object may be locked before the invocation.
      \param ni the node info 
  */
  void del_node(NodeInfo* ni);

  /** Add a route to node
      If the same route exists to this node then it is not added.
      A same channel may declared as a route to several nodes.
      This method is not reentrant and the Network object may be locked before the invocation.
      \param ni the pointer to the node info structure
      \param channel the new route to add
  */
  void add_route(NodeInfo* ni, OutChannel* channel);
  
  /** Delete the route to the node info using the channel
      This method is not reentrant and the Network object may be locked before the invocation.
      \param channel the route to del
  */
  void del_route(NodeInfo* ni, OutChannel* channel);
  
  /** Delete all the routes to any node info using the channel
      This method is not reentrant and the Network object may be locked before the invocation.
      \param channel the route to del
  */
  void del_routes(OutChannel* channel);
  
  /** Return the route (channel) to send data to local node identifier
      This method is reentrant and the Network object is locked during the invocation.
      \param ni the node info
      \return a pointer to the default route to reach the node or 0 if route doesnot exist
  */
  OutChannel* get_default_local_route( NodeInfo* ni );

  /** Return the route (channel) to send data to local node identifier using a specific device
      This method is reentrant and the Network object is locked during the invocation.
      \param ni the node info
      \return a pointer to the default route to reach the node or 0 if route doesnot exist
  */
  OutChannel* get_default_local_route( Device* device, NodeInfo* ni );

  /** Set a channel as a default channel to the destination to the local node identifier
      This method is not reentrant and the Network object may be locked before the invocation.
      \param ni the node info
      \param ch the default route to define for the node
  */
  void set_default_local_route( NodeInfo* ni, OutChannel* ch );

  /** open a route to ni using url as connection address
      May return an existing route.
      This method is not reentrant and the Network object may be locked before the invocation.
  */
  OutChannel* open_route( NodeInfo* ni, const std::string& url );

  /** close a route channel to ni 
      This method is not reentrant and the Network object may be locked before the invocation.
  */
  void close_route( NodeInfo* ni, OutChannel* channel );

  /** close all routes channel to ni 
      This method is not reentrant and the Network object may be locked before the invocation.
  */
  void close_routes( NodeInfo* ni );

  /** return the number of known node including myself
      This method is reentrant.
  */
  size_t count_nodes();

  /** return the iterator over the first known node
      It is to the responsability of the caller to assure that the iteration 
      is protected against concurrent modification of the set of node by locking
      the network object.
  */
  iterator_node begin_node();

  /** return the iterator over the first known node
      It is to the responsability of the caller to assure that the iteration 
      is protected against concurrent modification of the set of node by locking
      the network object.
  */
  const_iterator_node begin_node() const;

  /** return the iterator over the past-the-last known node
      It is to the responsability of the caller to assure that the iteration 
      is protected against concurrent modification of the set of node by locking
      the network object.
  */
  iterator_node end_node();

  /** return the iterator over the past-the-last known node
      It is to the responsability of the caller to assure that the iteration 
      is protected against concurrent modification of the set of node by locking
      the network object.
  */
  const_iterator_node end_node() const;
  
  /** Print routing nodes information
  */
  std::ostream& print_node( std::ostream& o );  

  /** Print routing routes information
  */
  std::ostream& print_route( std::ostream& o );
  
  /** Dump on std::cout (debug)
  */
  void dump_info();
  //@}

  // -----------------------------------------------------------------------
  //! \name Misc
  // -----------------------------------------------------------------------
  /** Return the url to be broadcast if other node want to connect to this node
  */
  const std::string& get_urlconnect( ) const
       throw (IOError);

  /** Return the device name of the address
  */
  std::string get_devicename_from_url( const std::string& url_peer ) const throw ();

  /** Return the device of the address
  */
  Device* get_device_from_url( const std::string& url_peer ) const throw ();
  
  /** Return default device for a cluster name in a hierarchy
  */
  Device* get_device_for_this_cluster( const std::string& clustername ) const throw();
   
  /** Return the identifier of this type of Network used to identify network
  */
  ka_uint8_t get_id() const;

  /** Flush all channels to all devices
  */
  void flush_channels();

  /** Set a new upcall object to forward upcall from network
      \param new_upcall the new object used to forward upcall
      \return the previous upcall object
  */
  Upcall* set_upcall( Upcall* new_upcall );

  /** Set a new Callback object to forward callback from network
  \param new_callback the new object used to forward callback
  \return the previous callback object
   */
  Callback* set_callback( Callback* new_callback );

  /** Signal event to the upcall object associated with the network
      Override the definition of Upcall::signal
  */  
  void signal(ComFailure::Code error_no, InChannel* ch, CallStack* cc);

  /** Signal event to the upcall object associated with the network
      Override the definition of Upcall::signal
  */  
  void signal(ComFailure::Code error_no, OutChannel* ch, Header* header);

  /** Method call on the server side when disconnection appears
      Inherited from Util::Callback
  */
  void notify(ComFailure::Code error_no);

  /** Method call on the server side when disconnection appears
      Inherited from Upcall
  */
  void notify(ComFailure::Code error_no, InChannel* ch, CallStack* cc);

  /** Method call on the client side when disconnection appears
      Inherited from Callback
  */
  void notify(ComFailure::Code error_no, OutChannel* ch, Header* header);

  /** Attach an IOInstructionStream to the process of sending message
  */
  void attach_iostream( IOInstructionStream* ios );

  /** Deattach an IOInstructionStream from the process of sending message
  */
  void deattach_iostream( IOInstructionStream* ios );
    
protected:
  ka_uint8_t                           _nid;              ///< id of the network instance
  std::string                          _name;             ///< The name of the network
  Upcall*                              _upcall;           ///< object to forward upcall, default is the network object
  Callback*                            _callback;         ///< object to forward callback, default is the network object
  mutable ka_uint16_t                  _next_lid;         ///< next local id for a node
  mutable std::list<ka_uint16_t>       _free_lid_entries; ///< list of free entries in _known_nodes

  mutable std::vector<NodeInfo*>       _known_nodes;      ///< known nodes from local id
  mutable size_t                       _count_nodes;      ///< number of nodes
  mutable std::map<GlobalId,NodeInfo*> _gid2nodes;        ///< global to nodeinfo lookup table
  
  Device*                              _default_device;   ///< Default device

  /* somes counters */
public:  
  friend class IODaemon;
  friend class IOInstructionStream;
}; // -- end class network


} // end namespace 
#endif 
