 // ==========================================================================
// (c) INRIA, projet MOAIS, 2006
// see the copyright file for more information
// Author: T. Gautier
// - based on socket class from Inuktitut2 & 3, author : C. Martin and T. Gautier
// ==========================================================================
#include <errno.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include "kasocknet_sockserver.h"

namespace SOCKNET {

// --------------------------------------------------------------------------
SocketServer::SocketServer()
 : Socket()
{
}


// --------------------------------------------------------------------------
void SocketServer::bind()
{
  _state &= ~S_FAIL;
  if (_fd == -1)
  {
    _fd = ::socket( AF_INET, SOCK_STREAM, IPPROTO_TCP );
    _error_no = errno;
    if (_fd == -1) {
      _state |= S_FAIL;
      _state |= S_CLOSE;
      return;
    }
    else {
      _state = S_GOOD;
    }
  }

  int err = ::bind(_fd, _addr.get_sockaddr(), _addr.get_sockaddr_size());
  _error_no = errno;

  if (err == -1)
  {
    _state |= S_FAIL;
    return;
  }
  _state &= ~S_CLOSE;

  /* bind success : report internal addr,port to Socknet::_addr */
#if defined(__APPLE__) || defined(__linux__)
  socklen_t sizesock= _addr.get_sockaddr_size();
#elif defined(KAAPI_USE_IRIX)
  int sizesock= _addr.get_sockaddr_size();
#endif
  err = getsockname(_fd, _addr.get_sockaddr(), &sizesock);
  _error_no = errno;
  if (err < 0)
  {
    close();
    _state |= S_BAD;
    return;
  }
}



// --------------------------------------------------------------------------
void SocketServer::listen( int backlog)
{
  if (!good()) return;
  _state &= ~S_FAIL;
  int listen_result = ::listen(_fd, backlog);
  _error_no = errno;
  if (listen_result<0)
  {
    _state |= S_BAD;
    _state |= S_FAIL;
  }
}



// --------------------------------------------------------------------------
void SocketServer::accept( SocketClient& sc )
{
  if (bad()) return;
  Address addr;
#if defined(__APPLE__) || defined(__linux__)
  socklen_t sizesock= addr.get_sockaddr_size();
#elif defined(KAAPI_USE_IRIX)
  int sizesock= addr.get_sockaddr_size();
#endif
  int fd;
  fd = ::accept(_fd, addr.get_sockaddr(), &sizesock);
  _error_no = errno;
  if (fd == -1)
  {
    _state |= S_FAIL;
    return;
  }
  if (!good()) return;
  sc.set_address(addr);
  sc.set_fd(fd);
}


} // namespace
