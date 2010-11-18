// ==========================================================================
// Copyright(c)'2005 by MOAIS Team
// see the copyright file for more information
// Author: T. Gautier
// ==========================================================================
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>

#include "utils_init.h"
#include "network_network.h"
#include "network_init.h"
#include "socknet_init.h"
#include "socknet_network.h"

namespace SOCKNET {

// --------------------------------------------------------------------
Init Init::component;

// --------------------------------------------------------------------
bool Init::verboseon = false;

// --------------------------------------------------------------------
size_t Init::sendbuffersize = 0;

// --------------------------------------------------------------------
size_t Init::recvbuffersize = 0;

// --------------------------------------------------------------------
std::map<std::string, std::string> Init::local_interfaces; 

// --------------------------------------------------------------------
Init::Init()
 : Util::KaapiComponent("socknet", Util::KaapiComponentManager::SOCKNET_COMPONENT_PRIORITY)
{}

// --------------------------------------------------------------------
int Init::initialize() throw()
{
  KaapiComponent::initialize();

  static bool once = true;
  if (once) once = false; else return 0;

  try {
    Net::Device::register_factory( &SOCKNET::factory );

    /* set verbose mode */
    SOCKNET::Init::verboseon = Util::Parser::String2Bool( Util::KaapiComponentManager::prop["socknet.verboseon"] );
    KAAPI_LOG (SOCKNET::Init::verboseon, "[SOCKNET::Init::initialize]");

    SOCKNET::Init::sendbuffersize = 1024*Util::Parser::String2Long( Util::KaapiComponentManager::prop["socknet.sndbuf"] );
    KAAPI_ASSERT_M(SOCKNET::Init::sendbuffersize >0, "[SOCKNET::Init::initialize] invalid send buffer size");

    SOCKNET::Init::recvbuffersize = 1024*Util::Parser::String2Long( Util::KaapiComponentManager::prop["socknet.rcvbuf"] );
    KAAPI_ASSERT_M(SOCKNET::Init::recvbuffersize >0, "[SOCKNET::Init::initialize] invalid recv buffer size");

    /* compute the list of all local interfaces AF_INET & IFF_UP & !IFF_LOOPBACK */
    Init::local_interfaces.clear();
    struct ifaddrs* interface = 0;
    if (getifaddrs( &interface ) == 0)
    {
      struct ifaddrs* curr = interface;
      while (curr !=0)
      {
        if ((curr->ifa_flags & IFF_UP) && !(curr->ifa_flags & IFF_LOOPBACK))
        {
          if (curr->ifa_addr != 0 && curr->ifa_addr->sa_family == AF_INET)
          {
            const sockaddr_in* saddr = (const sockaddr_in*)curr->ifa_addr;
            std::string name = curr->ifa_name;
            std::string addr = inet_ntoa( saddr->sin_addr);
            KAAPI_LOG( SOCKNET::Init::verboseon, "[SOCKNET::Device::initialize] found interface:" << name << ", @:" << addr );
            Init::local_interfaces.insert( std::make_pair( name, addr ) );
          }
        }
        curr = curr->ifa_next;
      }
      freeifaddrs( interface );
    } else {
      KAAPI_LOG( SOCKNET::Init::verboseon, "[SOCKNET::Device::initialize] no inet interface is up!");
      Util::Exception_throw( Util::PosixError("[SOCKNET::Device::initialize]no inet interface is up!", EINVAL) );
    }


    /* initialize return network */
    if (!Util::Parser::String2Bool( Util::KaapiComponentManager::prop["socknet.start"]) )
      return 0;


  } catch (const Util::Exception& ex) {
    KAAPI_LOG(  SOCKNET::Init::verboseon, "[SOCKNET::Init::initialize] catch '" << ex.what() << "'");
    return -1;
  } catch (...) {
    KAAPI_LOG(  SOCKNET::Init::verboseon, "[SOCKNET::Init::initialize] catch unkown exception");
    return -1;
  }

  return 0;
}


// --------------------------------------------------------------------
static Util::Parser::Module* socknet_module = 0;
int Init::terminate() throw()
{
  static bool once = true;
  if (once) once = false; else return 0;

  try {
    if (socknet_module !=0) delete socknet_module;
    if (!Util::Parser::String2Bool( Util::KaapiComponentManager::prop["socknet.start"]) )
      return 0;
  } catch (const Util::Exception& ex) {
    KAAPI_LOG( SOCKNET::Init::verboseon, "[SOCKNET::Init::terminate] catch '" << ex.what() << "'");
    return -1;
  } catch (...) {
    return -1;
  }
  return 0;
}


// --------------------------------------------------------------------
void Init::add_options( Util::Parser* parser, Util::Properties* global_prop )
{
  if (socknet_module ==0) 
  {
    socknet_module = new Util::Parser::Module;
    KAAPI_ASSERT_M(socknet_module !=0, "[SOCKNET::Factory::get_module] bad assertion");
    /* module definition */
    socknet_module->set_name("socknet");
    socknet_module->add_option("verboseon", "false", "Enable/Disable the verbose mode");
    socknet_module->add_option("hostname", "", "The hostname on which the process is running. If not defined, then KAAPI try to get the information from the system.");
    socknet_module->add_option("fileurl", "", "File name where URL of the process will be stored. The url is a string that represent a Kaapi network address.");
    socknet_module->add_option("bindurl", "<no value>", "IP Address where to bind the socket server to accept connection");
    socknet_module->add_option("interface", "<no value>", "interface name (e.g. eth0) where to bind the socket server to accept connection if bindurl is not given");
    socknet_module->add_option("start", "false", "true if the network should be initialized");
    socknet_module->add_option("sndbuf", "128", "(KByte) Send buffer size for socket");
    socknet_module->add_option("rcvbuf", "1024", "(KByte) Send buffer size for socket");
    socknet_module->add_option("cachesize", "-1", "[NOT YET IMPLEMENTED] Maximal size of alive connexions at once, default value=-1 for infinite cache");
    socknet_module->add_option("connectionless", "false", "Set to 'true' in order to connect socket before sending each message and disconnect after sending");
  }
  parser->add_module( socknet_module, global_prop);
}


// --------------------------------------------------------------------
void Init::declare_dependencies()
{
  Util::Init::component.is_required_by( &Init::component );
  Net::Init::component.is_required_by ( &Init::component );
  Init::component.is_required_by();
}


} // namespace
