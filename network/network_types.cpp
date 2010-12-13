// ====================================================================
// (c) INRIA, projet MOAIS, 2006
// Author: T. Gautier, X. Besseron
// Status: ok
//
//
// ====================================================================
#include "network_types.h"
#include "network_init.h"
#include "network_iochannel.h"
#include "network_network.h"
#include <algorithm>

namespace Net {

// -------------------------------------------------------------------------
NetworkObject::NetworkObject()
  : _manager(0)
{ }


// -------------------------------------------------------------------------
NetworkObject::~NetworkObject() 
{}


// -------------------------------------------------------------------------
void NetworkObject::initialize( )
{ 
  _manager = 0;
}


// -------------------------------------------------------------------------
void NetworkObject::set_network( Network* network ) throw()
{
  _manager = network;
}


// -------------------------------------------------------------------------
void NetworkObject::terminate()
{ }


// -------------------------------------------------------------------------
DeviceObject::DeviceObject()
  : _manager(0)
{ }


// -------------------------------------------------------------------------
DeviceObject::~DeviceObject() 
{}


// -------------------------------------------------------------------------
void DeviceObject::initialize( )
{ 
  _manager = 0;
}


// -------------------------------------------------------------------------
void DeviceObject::set_device( Device* device ) throw()
{
  _manager = device;
}


// -------------------------------------------------------------------------
void DeviceObject::terminate()
{ }


// -------------------------------------------------------------------------
void Callback::notify(ComFailure::Code error_no )
{ notify(error_no, 0, 0); }


// -------------------------------------------------------------------------
void Callback::signal(ComFailure::Code error_no, OutChannel* ch, Header* h)
{ 
  if ( _nocounter || (_count.decr_and_return() == 0) )
  {
    KAAPI_LOG(_notified, "Bad assertion: notify on '" << typeid(*this).name() << "'" << std::endl);
    KAAPI_ASSERT_DEBUG_M(_notified == false, "[ka::Callback::signal] bad assertion");
    _notified = true;
    notify(error_no, ch, h); 
  }
}


// -------------------------------------------------------------------------
void Callback::signaler(ComFailure::Code error_no, OutChannel* ch, Header* header, void* arg)
{ 
  if (arg ==0) return; 
  Callback* cbk = (Callback*)arg; 
//  std::cout << "In Callback::signaler, cbk=" << cbk<< std::endl;
  cbk->signal(error_no, ch, header); 
}  


// -------------------------------------------------------------------------
void Upcall::signal(ComFailure::Code error_no, InChannel* ch, CallStack* cc)
{ 
  notify(error_no, ch, cc); 
}


// -------------------------------------------------------------------------
void Upcall::notify(ComFailure::Code error_no )
{ notify(error_no, 0, 0); }


// --------------------------------------------------------------------
NodeInfo::~NodeInfo()
{
/*
    for( iterator_route ir = routes.begin() ; ir != routes.end(); ++ir )
        delete *ir;
    routes.clear();
*/
}

// --------------------------------------------------------------------
void NodeInfo::set_state( State s )
{ 
  if (s != node_state)
    state_time = Util::WallTimer::gettime();
  node_state = s; 
}

// --------------------------------------------------------------------
NodeInfo::State NodeInfo::get_state() const
{ return node_state; }

// --------------------------------------------------------------------
bool NodeInfo::add_address( const std::string& url_peer )
{
  std::map<std::string,OutChannel*>::iterator curr = routes.find(url_peer);
  if (curr != routes.end()) return false;
  routes.insert( std::make_pair(url_peer, (OutChannel*)0) );
  return true;
}

// --------------------------------------------------------------------
bool NodeInfo::add_route( const std::string& url_peer, OutChannel* channel )
{
  if (url_peer.empty()) return false;
  std::map<std::string,OutChannel*>::iterator curr = routes.find(url_peer);
  if (curr != routes.end()) return false;
  routes.insert( std::make_pair(url_peer, channel) );
  return true;
}


// --------------------------------------------------------------------
NodeInfo::iterator_route NodeInfo::begin_route()
{ return routes.begin(); }

// --------------------------------------------------------------------
NodeInfo::const_iterator_route NodeInfo::begin_route() const
{ return routes.begin(); }

// --------------------------------------------------------------------
NodeInfo::iterator_route NodeInfo::end_route()
{ return routes.end(); }

// --------------------------------------------------------------------
NodeInfo::const_iterator_route NodeInfo::end_route() const
{ return routes.end(); }


// ---------------------------------------------------------------------------
std::ostream& NodeInfo::print(std::ostream& o) const
{
  o << "[gid=" << global_id
    << ", lid=" << local_id
    << ", clustername=" << clustername
    << ", cluster=" << cluster
    << ", default_url=" << default_url
    << ", #routes=" << routes.size();
  const_iterator_route icurr = routes.begin();
  const_iterator_route iend  = routes.end();
  if (icurr != iend)
  {
    o << " ->(";
    while (icurr != iend)
    {
      o << "\n[peerurl:" << icurr->first;
      if (icurr->second !=0) 
      {
        icurr->second->print( o );
      }
      o << "] ";
      ++icurr;
    }
    o << ")";
  }
  o << ", default route @=" << default_route
    << "]";

  return o;
}



// ---------------------------------------------------------------------------
Cluster::Cluster() 
 : _name("default"), _parent(0), 
   _leadername(Util::GlobalId::NO_SITE), _leader(0), _level(0), _default_device(0)
{}

// ---------------------------------------------------------------------------
void Cluster::initialize()
{
  _lock.initialize();
  _recompute_exself = false;
}

// ---------------------------------------------------------------------------
void Cluster::terminate()
{
  _lock.terminate();
}

// ---------------------------------------------------------------------------
size_t Cluster::size() const
{ 
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  return _all_nodes.size(); 
}

// ---------------------------------------------------------------------------
void Cluster::set_leader( NodeInfo* ni )
{
  if (ni ==0) return;
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  if (_leader ==0) return;
  _leader = ni;
  _leadername = _leader->global_id;
}

// ---------------------------------------------------------------------------
void Cluster::set_leader( const Util::GlobalId& gid )
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  _leadername = gid;
  _leader = get_network()->globalid_to_nodeinfo( gid );
}


// ---------------------------------------------------------------------------
NodeInfo* Cluster::get_leader()
{ 
  if (_leader !=0) return _leader;
  _leader = get_network()->globalid_to_nodeinfo( _leadername );
  if (_leader !=0) {
    return _leader;
  }
  _leader = get_network()->get_leader_for_this_cluster( _name );
  if (_leader !=0)
    _leadername = _leader->global_id;
  return _leader;
}


// ---------------------------------------------------------------------------
void Cluster::elect_new_leader()
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  if (!_self_nodes.empty()) 
  {
    NodeInfo* min_id = 0;
    for (unsigned int i=0; i<_self_nodes.size(); ++i)
    {
      if ((min_id ==0) && (_self_nodes[i]->get_state() == NodeInfo::S_CREATED))
        min_id = _self_nodes[i];
      if ( (min_id !=0) && (_self_nodes[i]->get_state() == NodeInfo::S_CREATED) && (_self_nodes[i]->global_id < min_id->global_id)) 
        min_id = _self_nodes[i];
    }
    _leader = min_id;
    if (min_id !=0)
      ka::Init::network->set_leader_for_this_cluster( _leader->global_id, this->get_name() );
    return;
  }
  if (!_all_nodes.empty()) 
  {
    NodeInfo* min_id = 0;
    for (unsigned int i=0; i<_all_nodes.size(); ++i)
    {
      if ((min_id ==0) && (_all_nodes[i]->get_state() == NodeInfo::S_CREATED))
        min_id = _all_nodes[i];
      if ( (min_id !=0) && (_all_nodes[i]->get_state() == NodeInfo::S_CREATED) && (_all_nodes[i]->global_id < min_id->global_id)) 
        min_id = _all_nodes[i];
    }
    _leader = min_id;
    if (min_id !=0)
      ka::Init::network->set_leader_for_this_cluster( _leader->global_id, this->get_name() );
    return;
  }
}


// ---------------------------------------------------------------------------
NodeInfo* Cluster::get(unsigned int i)
{ 
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  if (i>=size()) return 0;
  return _all_nodes[i];
}

// ---------------------------------------------------------------------------
const NodeInfo* Cluster::get(unsigned int i) const
{ 
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  if (i>=size()) return 0;
  return _all_nodes[i];
}

// ---------------------------------------------------------------------------
size_t Cluster::self_size() const
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  return _self_nodes.size();
}

// ---------------------------------------------------------------------------
NodeInfo* Cluster::self_get(unsigned int i)
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  if (i>=self_size()) return 0;
  return _self_nodes[i];
}

// ---------------------------------------------------------------------------
const NodeInfo* Cluster::self_get(unsigned int i) const
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  if (i>=self_size()) return 0;
  return _self_nodes[i];
}


// ---------------------------------------------------------------------------
size_t Cluster::except_self_size() const
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK);
  return _exself_nodes.size(); 
}

// ---------------------------------------------------------------------------
NodeInfo* Cluster::except_self_get(unsigned int i)
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK);
  if (i>=_exself_nodes.size()) return 0;
  return _exself_nodes[i]; 
}

// ---------------------------------------------------------------------------
const NodeInfo* Cluster::except_self_get(unsigned int i) const
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  if (i>=_exself_nodes.size()) return 0;
  return _exself_nodes[i]; 
}

// ---------------------------------------------------------------------------
size_t Cluster::size_subcluster() const
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  return _vsubclusters.size();
}

// ---------------------------------------------------------------------------
Cluster* Cluster::get_subcluster(unsigned int i)
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  if (i>=size_subcluster()) return 0;
  return _vsubclusters[i];
}

// ---------------------------------------------------------------------------
const Cluster* Cluster::get_subcluster(unsigned int i) const
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  if (i>=size_subcluster()) return 0;
  return _vsubclusters[i];
}

// ---------------------------------------------------------------------------
const std::string& Cluster::get_name() const
{ 
  return _name; 
}

// ---------------------------------------------------------------------------
void Cluster::set_name( const std::string& name)
{
  _name = name;
}

// ---------------------------------------------------------------------------
const std::string& Cluster::get_path() const
{ 
  return _path;
}

// ---------------------------------------------------------------------------
Cluster* Cluster::get_parent( )
{ return _parent; }

// ---------------------------------------------------------------------------
void Cluster::set_parent( Cluster* p )
{ _parent = p; }

// ---------------------------------------------------------------------------
Cluster* Cluster::insert_node( 
  std::list<std::string>::const_iterator ibeg, 
  std::list<std::string>::const_iterator iend, 
  NodeInfo* ni 
) throw (Util::InvalidArgumentError)
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::WRITELOCK);
  const std::string& name = *ibeg;
  if (name != get_name())
    Exception_throw( Util::InvalidArgumentError("bad cluster name") );
  
  /* test if ni is leader for this cluster: */
  if (ka::Init::network->is_a_leader( ni->global_id, get_name()))
  {
    set_leader( ni );
    KAAPI_LOG( false, "[ka::Cluster::insert_node]: add a leader for cluster:" << get_name() << ", leader is:" << ni->global_id );
  }
  
  /* terminal case ? */
  Cluster* retval = 0;
  ++ibeg;
  if (ibeg == iend) 
  {
    if (std::find(_self_nodes.begin(), _self_nodes.end(), ni ) == _self_nodes.end())
    {
      _self_nodes.push_back( ni );
      _hself_nodes.insert( ni->global_id );
      _recompute_exself = true;
    }
    retval = this;
  }
  else {
    /* update _default_device if 0 */
    if (_default_device ==0) 
    {
      Device* device = get_network()->get_device_for_this_cluster( _name );
      if (device ==0) 
      {
        if (_parent !=0)
          _default_device = _parent->_default_device;
      }
      else
        _default_device = device;
    }
    
    /* else find the sub cluster *ibeg in _subcluster */
    const std::string& subclustername = *ibeg;
    std::map<std::string,Cluster*>::iterator curr = _subclusters.find(subclustername);
    //Util::logfile() << "Subclustername:" << subclustername << std::endl;
    Cluster* subcluster = 0;
    if (curr == _subclusters.end())
    {
      //Util::logfile() << "Create new subclustername:" << subclustername << std::endl;
      subcluster = new Cluster;
      subcluster->initialize();
      subcluster->set_network( get_network() );
      subcluster->set_name( subclustername );
      subcluster->_path = _path + '/' + subclustername;
      get_network()->register_cluster( subcluster, subcluster->_path );
      Device* device = get_network()->get_device_for_this_cluster( subclustername );
      subcluster->_default_device = ( device ==0 ? _default_device : device );
      subcluster->_leader = 0;
      if (subcluster->get_parent() !=0)
        Exception_throw( Util::InvalidArgumentError("sub cluster name as already a parent in other hierarchy") );
      subcluster->set_parent(this);
      subcluster->_level = _level+1;
      _subclusters.insert( std::make_pair(subclustername, subcluster) );
      _vsubclusters.push_back( subcluster );
    }
    else 
      subcluster = curr->second;
    
    /* recursive call */
    retval = subcluster->insert_node( ibeg, iend, ni );
  }

  /* insert locally in all nodes: do it at the end in order to do nothing in case of exception */
  if (std::find(_all_nodes.begin(), _all_nodes.end(), ni ) == _all_nodes.end())
  {
    _all_nodes.push_back( ni );
    _hall_nodes.insert( ni->global_id ) ;
    _recompute_exself = true;
  }
  
  return retval;
}



// --------------------------------------------------------------------
void Cluster::erase_node( NodeInfo* ni )
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::WRITELOCK);
  std::vector<NodeInfo*>::iterator icurr = std::find(_all_nodes.begin(), _all_nodes.end(), ni );
  if ( icurr != _all_nodes.end())
  { /* found in this cluster erase it and do recursive call */
    _recompute_exself = true;
    _all_nodes.erase( icurr );
    _hall_nodes.erase(  (*icurr)->global_id );
    icurr = std::find(_self_nodes.begin(), _self_nodes.end(), ni );
    if ( icurr != _self_nodes.end())
    {
      _self_nodes.erase( icurr );
      _hself_nodes.erase( _hself_nodes.find( ni->global_id ) ); /* should exist ! */
    }
    for (unsigned int i=0; i<_vsubclusters.size(); ++i)
      if (_vsubclusters[i] !=0) _vsubclusters[i]->erase_node( ni );
    if (_leader == ni) {
      Util::logfile() << "WARNING erasing the leader node info data structure. Please contact the authors" << std::endl;
    }
  }
}



// --------------------------------------------------------------------
void Cluster::compute_except_self_sets( )
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::WRITELOCK);
  if (!_recompute_exself) return;

  /* compute the except_self sets for this cluster and its subclusters */
  typedef std::set<Util::GlobalId> Set;
  for (unsigned int i=0; i<_vsubclusters.size(); ++i)
  {
    _vsubclusters[i]->_exself_nodes.clear();
    _vsubclusters[i]->_hexself_nodes.clear();
    _vsubclusters[i]->compute_except_self_sets();
  }

  for (unsigned int i=0; i<_vsubclusters.size(); ++i)
  {
    for (unsigned int j=0; j < _vsubclusters.size(); ++j)
    {
      if (i ==j) continue;
      for (unsigned int k=0; k < _vsubclusters[i]->_all_nodes.size(); ++k)
        _vsubclusters[j]->_exself_nodes.push_back( _vsubclusters[i]->_all_nodes[k] );
    }
  }
  _recompute_exself = false;
}


// --------------------------------------------------------------------
bool Cluster::is_incluster( const Util::GlobalId& gid ) const
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK );
  return _hall_nodes.find( gid ) != _hall_nodes.end();
}


// --------------------------------------------------------------------
struct FTAB {
  FTAB(int l):level(l){}
  int level;
};

std::ostream& operator<<( std::ostream& o, const FTAB& ft )
{
  for (int i=0; i<ft.level; ++i)
    o << '\t';
  return o;
}
FTAB ftab( int level )
{ return FTAB(level); }


void Cluster::print( std::ostream& o ) 
{
  Util::RWCriticalGuard( _lock, Util::RWCriticalGuard::READLOCK);
  NodeInfo* leader = get_network()->get_leader_for_this_cluster( get_name() );
  o << _level << ftab(_level) << "Cluster: '" << get_name() << "'"
    << ", path:'" << _path << "'"
    << ", device:" << (_default_device ==0 ? "no specific device" : _default_device->get_factory()->get_name())
    << ", leadername:" << _leadername
    << ", leader:" << (leader == 0 ? Util::GlobalId::NO_SITE : leader->global_id)
    << ", #nodes:" << size() 
    << ", #selfnodes:" << self_size() 
    << ", #subclusters:" << size_subcluster() 
    << std::endl;
  for (unsigned int i=0; i<self_size(); ++i)
  {
    const NodeInfo* ni = self_get(i);
    o << _level << ftab(_level) << "Node:" << ni->global_id << ", clusterpath:'" << (ni->cluster == 0 ? "undefined" : ni->cluster->_path);
    std::map<Device*,NodeInfo::Routes>::const_iterator ibeg = ni->per_dev_routes.begin();
    std::map<Device*,NodeInfo::Routes>::const_iterator iend = ni->per_dev_routes.end();
    while (ibeg != iend)
    {
      Device* device = ibeg->first;
      const NodeInfo::Routes& routes = ibeg->second;
      if (device !=0) o << " Device:" << device->get_factory()->get_name();
      else o << " Device: unknown (0)";
      NodeInfo::Routes::const_iterator ib = routes.begin();
      NodeInfo::Routes::const_iterator ie = routes.end();
      while (ib != ie)
      {
        o << " url:" << ib->first << ", @=" << ib->second;
        ++ib;
      }
      ++ibeg;
    }
    o << std::endl;
  }
  compute_except_self_sets();
  o << _level << ftab(_level) << "Except self nodes:" << std::endl;
  for (unsigned int i=0; i<except_self_size(); ++i)
  {
    NodeInfo* ni = except_self_get(i);
    o << _level << ftab(_level) << "Gid:" << ni->global_id << std::endl;
  }
  for (unsigned int i=0; i<size_subcluster(); ++i)
    get_subcluster(i)->print(o);
}

} // namespace Net
