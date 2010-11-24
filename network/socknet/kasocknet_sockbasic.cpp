 // ==========================================================================
// (c) INRIA, projet MOAIS, 2006
// see the copyright file for more information
// Author: T. Gautier
// - based on socket class from Inuktitut2 & 3, author : C. Martin and T. Gautier
// ==========================================================================

#include "kasocknet_sockbasic.h"
#include "ka_debug.h"
#include <iostream>
#include <stdio.h>
#include <string.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
//for fcntl
#include <fcntl.h>
//for TCP_NODELAY
#include <netinet/in.h>
//for close
#include <unistd.h>
#include <signal.h>

#if defined(KAAPI_USE_LINUX) && defined(KAAPI_USE_ARCH_ITA)
typedef void (*sighandler_t)(int);
#endif

namespace SOCKNET {


// --------------------------------------------------------------------------
Address::Address()
{
  ::bzero(&_sockaddr,sizeof(_sockaddr));
#if defined(__APPLE__) || defined(__linux__)
  _sockaddr.sin_family = AF_INET;
#endif
  set_toany();
}


// --------------------------------------------------------------------------
Address::Address(const Address& addr)
{
  *this = addr;
}


// --------------------------------------------------------------------------
Address::~Address()
{
}


// --------------------------------------------------------------------------
Address& Address::operator=(const Address& addr)
{
  _sockaddr = addr._sockaddr;
  return *this;
}


// --------------------------------------------------------------------------
void Address::set_toany(  )
{
  _sockaddr.sin_addr.s_addr = htonl(INADDR_ANY);
  _sockaddr.sin_port = htons(0);
}


// --------------------------------------------------------------------------
ComFailure::Code Address::set_byname( const std::string& host )
{
//std::cout << "IN " << __PRETTY_FUNCTION__ << std::endl;
  hostent* hp;
redo:
  hp = gethostbyname(host.c_str());
  if (hp == 0) {
    bzero(&_sockaddr,sizeof(_sockaddr));
    switch (h_errno) {
      case TRY_AGAIN: goto redo;
      case HOST_NOT_FOUND: return ComFailure::HOST_NOTFOUND;
      case NO_RECOVERY: return ComFailure::FAIL;
      case NO_DATA: return ComFailure::HOST_NOTADDRESS;
    };
    return ComFailure::FAIL;
  }
  // copy address
  _sockaddr.sin_family = AF_INET;
  memcpy(&_sockaddr.sin_addr, hp->h_addr_list[0], sizeof(struct in_addr));
  return ComFailure::OK;
}


// --------------------------------------------------------------------------
std::string Address::get_hostname() const
{
  char tmp[1024];
  sprintf(tmp,"%s", inet_ntoa( _sockaddr.sin_addr ) );
  return std::string(tmp);
}


// --------------------------------------------------------------------------
std::string Address::to_string() const
{
  char tmp[1024];
  sprintf(tmp,"socknet:%s:%i", inet_ntoa( _sockaddr.sin_addr ) , ntohs(_sockaddr.sin_port) );
  return std::string(tmp);
}


// --------------------------------------------------------------------------
ComFailure::Code Address::from_string(const std::string& s)
{
  /* parse url: socknet:host:port, host : ip=<n1>.<n2>.<n3>.<n4> | host name */
  Address server;
  std::string host;
  int port = 0;
  typedef std::string::size_type csize_t;
  csize_t beg_port = 0;
  csize_t beg_host = s.find("socknet");
  if (beg_host == (csize_t)-1) 
  {
    return ComFailure::BAD_ADDRESS;
  }
  beg_host = s.find(":",beg_host);
  if (beg_host >0) 
  {
    std::string str_port;
    beg_port = s.find(":",beg_host+1);
    if (beg_port >0) {
      host = s.substr(beg_host+1, beg_port-beg_host-1 );
      str_port = s.substr(beg_port+1, s.size()-beg_port-1 );
      std::istringstream input(str_port);
      input >> port;
    } else {
      host = s.substr(beg_host+1, s.size()-beg_host );
      port = 0;
    }
  }
  else 
    return ComFailure::BAD_ADDRESS;

  ComFailure::Code err;
  if ((err = set_byname(host)) != ComFailure::OK) return err;
  set_port(port);
  return ComFailure::OK;
}


// --------------------------------------------------------------------------
void Address::set_port(ka::uint16_t p)
{
  _sockaddr.sin_port = htons(p);
}


// --------------------------------------------------------------------------
ka::uint16_t Address::get_port() const
{
  return ntohs(_sockaddr.sin_port);
}


// --------------------------------------------------------------------------
ka::int32_t Address::get_ipv4() const
{
  return ntohl( _sockaddr.sin_addr.s_addr );
}


// --------------------------------------------------------------------------
void Address::set_ipv4(ka::int32_t a) 
{
  _sockaddr.sin_addr.s_addr = htonl( a );
}


// --------------------------------------------------------------------------
std::ostream& Address::print(std::ostream& o ) const
{
  o << "[sin_family= " << _sockaddr.sin_family
  	<< ", sin_port= " << ntohs(_sockaddr.sin_port)
  	<< ", sin_addr= " << inet_ntoa(_sockaddr.sin_addr)
  	<< "]"<<std::endl; 
  return o;
}



// --------------------------------------------------------------------------
Socket::Socket()
 : _addr(), _fd(-1), _state(S_CLOSE), _error_no(0)
{
}


// --------------------------------------------------------------------------
void Socket::initialize()
{
  _error_no = 0;
  _fd = ::socket( AF_INET, SOCK_STREAM, IPPROTO_TCP );
  if (_fd == -1) 
  {
    _error_no = errno;
    _state |= S_FAIL;
    _state |= S_CLOSE;
    KAAPI_CPPLOG( true, "[SOCKNET::Socket::initialize] Socket creation failed");
    return;
  }
  else {
    _state = S_GOOD;
  }
  KAAPI_CPPLOG( false, "[SOCKNET::Socket::initialize] new fd=" << _fd);
}


// --------------------------------------------------------------------------
void Socket::close()
{
  KAAPI_CPPLOG( false, "[SOCKNET::Socket::close] fd=" << _fd);
  ::close(_fd);
  _error_no = errno;  
  _fd = -1;
  _state = S_CLOSE;
}


// --------------------------------------------------------------------------
void Socket::shutdown(SHUT_MODE mode)
{
  _state &= ~S_FAIL;
  int retval = ::shutdown( _fd, (int)mode);
  _error_no = errno;
  if (retval ==-1) 
  {
    _state |= S_FAIL;
    switch (errno) 
    {
      case EBADF:
      case ENOTSOCK:
      case ENOTCONN:
        _fd = -1;
        _state |= S_CLOSE;
        break;
    }
  }
}


// --------------------------------------------------------------------------
void Socket::set_fd( int fd )
{
  _state &= ~S_FAIL;
  if (fd ==-1) 
  {
    _state |= S_CLOSE;
  }
  else 
  {
    _state &= ~S_CLOSE;
    _fd = fd;
  }
}


// --------------------------------------------------------------------------
int Socket::error_no() const
{
  return _error_no;
}

// --------------------------------------------------------------------------
#define OTHER 0 
struct sock_options {
  int level;
  int name;
  const char* cmt;
};
static sock_options soptions[] = {
  { IPPROTO_TCP, TCP_NODELAY, "TCP_NODELAY" },  /* no_delay */
  { OTHER, OTHER, "undefined" },                /* clo_exec */
  { SOL_SOCKET, SO_LINGER, "linger" },          /* linger */
  { SOL_SOCKET, SO_REUSEADDR, "reuseaddr" },    /* reuseaddr */
  { SOL_SOCKET, SO_RCVBUF, "rcvbuf" },          /* rcvbuf */
  { SOL_SOCKET, SO_SNDBUF, "sndbuf" },          /* sndbuf */
  { SOL_SOCKET, SO_RCVLOWAT, "rcvlowat" },      /* rcvlowat */
  { SOL_SOCKET, SO_SNDLOWAT, "sndlowat" },      /* sndlowat */
#if defined(__linux__) 
  { SOL_SOCKET, OTHER, "nosigpipe" },           /* nosigpipe */
#elif defined(__APPLE__)
  { SOL_SOCKET, SO_NOSIGPIPE, "nosigpipe" },    /* nosigpipe */
#endif
  { SOL_SOCKET, SO_DONTROUTE, "dontroute" },    /* dontroute */
  { OTHER, OTHER, "undefined" },                /* nonblock */
};


// --------------------------------------------------------------------------
void Socket::set_sockopt(int opt, int value )
{
  _state &= ~S_FAIL;
  int retval = 0;
  switch (opt)
  {
    case clo_exec :
    {
      int flag =fcntl(_fd, F_GETFD, 0);
      _error_no = errno;
      if (flag == -1) {
        _state |= S_FAIL;
        return;
      }
      if (value !=0)
        flag |= FD_CLOEXEC;
      else
        flag &= ~FD_CLOEXEC;
      retval = fcntl(_fd, F_SETFD, FD_CLOEXEC);
      _error_no = errno;
      if (retval == -1) _state |= S_FAIL;
      return;
    }

    case linger : {
      if (value <=0) {
        /* timeout in second is 0: no linger */
        struct linger l;
        l.l_onoff = 0;
        retval = setsockopt(_fd, soptions[opt].level, soptions[opt].name, &l, sizeof(linger));
        _error_no = errno;
        if (retval == -1) _state |= S_FAIL;
      }
      else if (value >0) {
        struct linger l;
        l.l_onoff = 1;
        l.l_linger = value;
        retval = setsockopt(_fd, soptions[opt].level, soptions[opt].name, &l, sizeof(linger));
        _error_no = errno;
        if (retval == -1) _state |= S_FAIL;
      }
      return;
    }
    
    case nosigpipe:
#if defined(KAAPI_USE_LINUX)
    {
      retval =0;
      if (value !=0) signal( SIGPIPE, SIG_IGN);
      else signal( SIGPIPE, SIG_DFL);
      _error_no = errno;
      if (retval == -1) _state |= S_FAIL;
    } break;
#endif
    case no_delay:
    case reuseaddr :
    case rcvbuf:
    case sndbuf:
    case rcvlowat:
    case sndlowat:
    case dontroute:
      retval = setsockopt(_fd,soptions[opt].level,soptions[opt].name, &value, sizeof(value));
      _error_no = errno;
      if (retval == -1) _state |= S_FAIL;
      return;

    case nonblock :
    {
      int flag =fcntl(_fd, F_GETFD,0);
      _error_no = errno;
      if (flag== -1) {
        _state |= S_FAIL;
        return;
      }
      if (value !=0)
        flag |= O_NONBLOCK;
      else
        flag &= ~O_NONBLOCK;
      retval = fcntl(_fd, F_SETFD, flag);
      _error_no = errno;
      if (retval == -1) _state |= S_FAIL;
      return;
    }

    default:
      _error_no = errno;
      _state |= S_FAIL;
      return;
  }
}


// --------------------------------------------------------------------------
void Socket::get_sockopt(int opt, int& value )
{
  int retval = 0;
  _state &= ~S_FAIL;
  switch (opt) 
  {
    case clo_exec :
    {
      int flag = fcntl(_fd, F_SETFD, FD_CLOEXEC);
      _error_no = errno;
      if (flag == -1)  {
        _state |= S_FAIL;
        return;
      }
      value =flag;
      return;
    }

    case linger : {
      /* timeout in second is 0: no linger */
      socklen_t sz = sizeof(linger);
      struct linger l;
      retval = getsockopt(_fd,soptions[opt].level,soptions[opt].name, &l, &sz);
      _error_no = errno;
      if (_error_no != 0) _state |= S_FAIL;

      if (sz != sizeof(value)) {
        _state |= S_FAIL;
        return;
      }
      if (l.l_onoff ==0) value = 0;
      else value = l.l_linger;
      return;
    }

    case nosigpipe:
#if defined(KAAPI_USE_LINUX)
    {
      retval =0;
      sighandler_t sigh = signal( SIGPIPE, SIG_IGN);
      _error_no = errno;
      if (_error_no != 0) _state |= S_FAIL;
      signal( SIGPIPE, sigh);
      _error_no = errno;
      if (_error_no != 0) _state |= S_FAIL;
    } break;
#endif
    case no_delay:
    case reuseaddr :
    case rcvbuf:
    case sndbuf:
    case rcvlowat:
    case sndlowat:
    case dontroute:
    {
#if defined(__APPLE__) || defined(__linux__)
      socklen_t sz = sizeof(value);
#else
      int sz = sizeof(value);
#endif
      retval = getsockopt(_fd,soptions[opt].level,soptions[opt].name, &value, (socklen_t*)&sz);
      _error_no = errno;
      if (_error_no != 0) _state |= S_FAIL;
      if (sz != sizeof(value)) {
        _state |= S_FAIL;
        return;
      }
      return;
    }

    case nonblock :
    {
      int flag =fcntl(_fd, F_GETFD,0);
      _error_no = errno;
      if (_error_no != 0) _state |= S_FAIL;
      if (flag == -1)
        return;
      value =flag;
      return;
    }

    default:
      _state |= S_FAIL;
      return;
  }
}

// --------------------------------------------------------------------------
std::ostream& Socket::print( std::ostream& o ) const
{
  o << "[@=" << _addr
    << ", file descr=" << _fd
    << ", good=" << (good() ? "true" : "false")
    << ", fail=" << (fail() ? "true" : "false")
    << ", bad=" << (bad() ? "true" : "false")
    << "]"<<std::endl; 
  return o;
}

}//namespace SOCKNET
