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
#include "kaapi_impl.h"
#include "kanet_network.h"
#include "kanet_channel.h"
#include "kanet_device.h"
#include "ka_parser.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <vector>


namespace ka {

// --------------------------------------------------------------------
Network Network::object;

// --------------------------------------------------------------------
Network::Network( ) 
 : _gid2channel(), _name2device(), _all_devices(), _default_device(0)
{
}

// --------------------------------------------------------------------
Network::~Network() 
{ 
}

// ---------------------------------------------------------------------------
void Network::initialize( )
    throw (RuntimeError)
{
  _gid2channel.clear();
  _name2device.clear();
  _all_devices.clear();  

  /* initialize devices: use the environment variable KAAPI_NETWORK
     that should define a list of device to used.
   */
  const char* lnet = getenv("KAAPI_NETWORK");
  if (lnet)
  {
    std::string str_list_net = std::string(lnet);
    std::list<std::string> list_of_net;
    ka::Parser::String2lstStr( list_of_net, str_list_net );
    std::list<std::string>::const_iterator ibeg = list_of_net.begin();
    std::list<std::string>::const_iterator iend = list_of_net.end();
    while (ibeg != iend)
    {
      DeviceFactory* df = DeviceFactory::resolve_factory(ibeg->c_str());
#if defined(KAAPI_DEBUG)
      if (df ==0)
        std::cerr << "Unknown network name: '" << *ibeg << "'" << std::endl;
#endif
      if (df !=0)
      {
        Device* device = df->create();
        kaapi_assert( device != 0);
        _all_devices.push_back( device );
        _name2device.insert( std::make_pair(device->get_name(), device) );
      }
      ++ibeg;
    }
  }
  if (!_all_devices.empty())
    _default_device = *_all_devices.begin();
}

// ---------------------------------------------------------------------------
void Network::commit( )
    throw (RuntimeError)
{
  std::vector<Device*>::iterator ibeg = _all_devices.begin();
  std::vector<Device*>::iterator iend = _all_devices.end();
  while (ibeg != iend)
  {
    (*ibeg)->commit();
    ++ibeg;
  }
}

// ---------------------------------------------------------------------------
void Network::terminate () throw (RuntimeError)
{
  std::vector<Device*>::iterator ibeg = _all_devices.begin();
  std::vector<Device*>::iterator iend = _all_devices.end();
  while (ibeg != iend)
  {
    (*ibeg)->terminate();
    ++ibeg;
  }

  /* destroy all devices to network interface */
  ibeg = _all_devices.begin();
  iend = _all_devices.end();
  while (ibeg != iend)
  {
    const char* name = (*ibeg)->get_name();
    DeviceFactory* df = DeviceFactory::resolve_factory( name );
    kaapi_assert( df != 0);
    df->destroy( *ibeg );
    ++ibeg;
  }
  _all_devices.clear();
}


#ifdef MACOSX_EDITOR
#pragma mark --- routing table
#endif
// ---------------------------------------------------------------------------
void Network::add_node(GlobalId gid, const char* url)
{
  ListUrls* listurls;
  std::map<GlobalId,ListUrls*>::iterator curr = _gid2urls.find(gid);
  
  if (curr == _gid2urls.end())
  {
    listurls = new ListUrls;
    _gid2urls.insert( std::make_pair(gid, listurls ) );
  }
  else 
    listurls = curr->second;
  
  listurls->insert( strdup(url) );
}


// ---------------------------------------------------------------------------
void Network::add_route(GlobalId gid, OutChannel* channel)
{
  if (channel == 0) return;

  /* look if route exist to ni */
  std::map<GlobalId,OutChannel*>::iterator curr = _gid2channel.find( gid );
  kaapi_assert_m( curr == _gid2channel.end(), "route already exist" );

  _gid2channel.insert( std::make_pair(gid, channel) );
}

// ---------------------------------------------------------------------------
void Network::del_route(GlobalId gid, OutChannel* channel)
{
  if (channel == 0) return;

  std::map<GlobalId,OutChannel*>::iterator curr = _gid2channel.find( gid );
  if (curr == _gid2channel.end())

  _gid2channel.erase( curr );
}


// ---------------------------------------------------------------------------
OutChannel* Network::get_default_local_route( GlobalId gid ) 
{
  std::map<GlobalId,OutChannel*>::iterator curr = _gid2channel.find( gid );
  if (curr != _gid2channel.end()) 
    return curr->second;
  
  /* try to open a route to gid using its urls */
  std::map<GlobalId,ListUrls*>::iterator lur = _gid2urls.find(gid);
  if (lur == _gid2urls.end()) return 0;
  ListUrls* listurls = _gid2urls.find(gid)->second;
  if (listurls ==0) return 0;
  ListUrls::iterator urlbeg = listurls->begin();
  ListUrls::iterator urlend = listurls->end();
  while (urlbeg != urlend)
  {
    Device* device = get_device_from_url( *urlbeg );
    if (device ==0) {
      std::cout << "Cannot find device for url:" << *urlbeg << std::endl << std::endl;
    }
    else {
      OutChannel* channel = device->open_channel(*urlbeg);
      if (channel !=0)
      {
        channel->set_peer( gid, *urlbeg );
        _gid2channel.insert( std::make_pair( gid, channel ) );
        return channel;
      }
      ++urlbeg;
    }
  }
  return 0;
}



// ---------------------------------------------------------------------------
void Network::set_default_local_route( GlobalId gid, OutChannel* channel ) 
{
  if (channel ==0) return;

  /* look if channel is in the list of route */
  std::map<GlobalId,OutChannel*>::iterator curr = _gid2channel.find( gid );
  if (curr != _gid2channel.end()) 
  {
    curr->second = channel;
    return;
  }

  _gid2channel.insert( std::make_pair( gid, channel ) );
}


// --------------------------------------------------------------------
std::ostream& Network::print_route( std::ostream& o )
{
  std::map<GlobalId,OutChannel*>::iterator icurr = _gid2channel.begin();
  std::map<GlobalId,OutChannel*>::iterator iend  = _gid2channel.end();
  while (icurr != iend)
  {
    OutChannel* channel = icurr->second;
    o << "Gid: " << icurr->first 
      << "\n -> default @channel: "  << channel 
      << " peer node: "  << channel->get_peer_url()
      << ", device: " << channel->get_device()->get_name() << std::endl;

    ListUrls* listurls;
    std::map<GlobalId,ListUrls*>::iterator curr = _gid2urls.find(icurr->first);
    kaapi_assert( curr != _gid2urls.end() );
    listurls = curr->second;
    ListUrls::iterator urlbeg = listurls->begin();
    ListUrls::iterator urlend = listurls->end();
    while (urlbeg != urlend)
    {
      o << "\turl:  " << *urlbeg << std::endl;
      ++urlbeg;
    }
    
    ++icurr;
  }

  return o;
}

// --------------------------------------------------------------------
std::ostream& Network::print_info( std::ostream& o )
{
  std::map<GlobalId,ListUrls*>::iterator icurr = _gid2urls.begin();
  std::map<GlobalId,ListUrls*>::iterator iend  = _gid2urls.end();
  while (icurr != iend)
  {
    o << "Gid: " << icurr->first << std::endl;

    ListUrls* lu = icurr->second;
    ListUrls::iterator urlbeg = lu->begin();
    ListUrls::iterator urlend = lu->end();
    while (urlbeg != urlend)
    {
      o << "\turl:  " << *urlbeg << std::endl;
      ++urlbeg;
    }
    
    ++icurr;
  }

  return o;
}

// --------------------------------------------------------------------
void Network::dump_info()
{
  std::ostringstream buff1;
  print_info(buff1); buff1 << std::endl;
  
  std::ostringstream buff2;
  print_route(buff2); buff2 << std::endl;

  printf("%i::Node info:\n%s\n%i::Node route info:\n%s\n", 
    ka::System::local_gid, buff1.str().c_str(), 
    ka::System::local_gid, buff2.str().c_str() 
  );
  fflush(stdout);
}


#ifdef MACOSX_EDITOR
#pragma mark --- misc
#endif
// --------------------------------------------------------------------
size_t Network::size() const
{ 
  return _gid2urls.size();
}

// --------------------------------------------------------------------
const char* Network::get_urlconnect( ) const throw (IOError)
{
  if (_default_device ==0) ka::Exception_throw( IOError("no default device") );
  return _default_device->get_urlconnect();
}

// --------------------------------------------------------------------
Device* Network::get_device_from_url( const std::string& url_peer ) throw ()
{
  std::string::size_type sep = 0;
  sep = url_peer.find(":");
  if (sep == (std::string::size_type)-1) 
    return 0;
  
  /* try to find an existing device */
  const char* name_device = url_peer.substr(0, sep ).c_str();
  std::map<std::string, Device*>::const_iterator curr = _name2device.find(name_device);
  if (curr != _name2device.end())
  {
    return curr->second;
  }

  /* try to create a new device from a registered factory */
  DeviceFactory* df = DeviceFactory::resolve_factory( name_device );
  if (df !=0)
  {
    Device* device = df->create();
    kaapi_assert( device != 0);
    _all_devices.push_back( device );
    _name2device.insert( std::make_pair(device->get_name(), device) );
    return device;
  }
  return 0;
}

} // - namespace Net...
