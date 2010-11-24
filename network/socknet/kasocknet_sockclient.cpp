// ==========================================================================
// (c) INRIA, projet MOAIS, 2006
// see the copyright file for more information
// Author: T. Gautier
// - based on socket class from Inuktitut2 & 3, author : C. Martin and T. Gautier
// ==========================================================================
#include <iostream>
#include <errno.h>

// for writev readv
#include <sys/types.h>
#include <sys/uio.h>
// for perror
#include <stdio.h>
//for close
#include <unistd.h>

// AF_LOCAL
#include <sys/socket.h>
#include <sys/un.h>


#include <netinet/in.h>
//for hostent
#include <netdb.h>

// strerror
#include <string.h>

#include "kasocknet_sockclient.h"

#ifndef UIO_MAXIOV
#define UIO_MAXIOV 128
#endif

namespace SOCKNET {

// --------------------------------------------------------------------------
SocketClient::SocketClient()
 : Socket()
{
}



// --------------------------------------------------------------------------
SocketClient::SocketClient(int fd)
 : Socket()
{
  Socket::set_fd( fd );
}


 
// --------------------------------------------------------------------------
void SocketClient::connect( const Address& serveraddr)
{
  if (_fd == -1)
  {
    _error_no = EBADF;
    _state |= S_FAIL;
    return;
  }

  /* connect */
  _state &= ~S_FAIL;
  int connect_result = ::connect(_fd, serveraddr.get_sockaddr(), serveraddr.get_sockaddr_size());
  _error_no = errno;
  if ((connect_result == -1) && (errno != EISCONN))
  {
    _state |= S_FAIL;
    return;
  }
  /* set the state as good */
  _state &= ~S_CLOSE;
  _state &= ~S_FAIL;

  /* get my address */
#if defined(__APPLE__) || defined(__linux__)
  socklen_t sizesock;
#elif defined(KAAPI_USE_IRIX)
  int sizesock;
#endif
  sizesock= _addr.get_sockaddr_size();
  connect_result = getsockname(_fd, _addr.get_sockaddr(), &sizesock);
  _error_no = errno;
  if (connect_result<0)
  {
    _state |= S_FAIL;
    return;
  }
}



// --------------------------------------------------------------------------
void SocketClient::get_peer_address( Address& peeraddr ) 
{
  _state &= ~S_FAIL;
  int retval =0;
  /* get peer address */
#if defined(__APPLE__) || defined(__linux__)
  socklen_t sizesock;
#elif defined(KAAPI_USE_IRIX)
  int sizesock;
#endif
  sizesock= peeraddr.get_sockaddr_size();
  retval = getpeername(_fd, peeraddr.get_sockaddr(), &sizesock);
  _error_no = errno;
  if (retval == -1)
    _state |= S_FAIL;
}



// --------------------------------------------------------------------------
ssize_t SocketClient::write(const void* data, int size)
{
  if (!good()) return 0;
doit_again:
  _error_no = 0;
  ssize_t nbsent = ::write(_fd, data, size);
  if (nbsent<0) 
  {
    _error_no = errno;
    if (_error_no == EINTR) goto doit_again;
    _state |= S_FAIL;
    switch (_error_no) {
      case EPIPE: close();
      default:
        break;
    }
  }
  return nbsent;
}



// --------------------------------------------------------------------------
ssize_t SocketClient::writen(const void* data, int size)
{
  if (!good()) return 0;
  ssize_t nbsent=0;
  ssize_t nbtosent = size;
  ssize_t retval =0;
  while (nbtosent >0)
  {
    nbsent = (nbtosent < SSIZE_MAX ? nbtosent : SSIZE_MAX);
doit_again:
    _error_no = 0;
    nbsent = ::write(_fd, data, nbsent);
    if (nbsent<0) 
    {
      _error_no = errno;
      if (_error_no == EINTR) goto doit_again;
      _state |= S_FAIL;
      switch (_error_no) {
        case EPIPE: close();
        default: 
          break;
      }
      return retval;
    }
    data = ((char*)data) + nbsent;
    nbtosent  -= nbsent;
    retval += nbsent;
  }
  return retval;
}



// --------------------------------------------------------------------------
ssize_t SocketClient::writev( ka::IOVectEntry* iov, int countiov )
{
  if (!good()) return 0;
  if (countiov ==0) return 0;

  int i, iovcnt;
  ssize_t totalsent = 0; /* total number of bytes sent */
  ssize_t szonesent;     /* number of bytes to send for one ::writev call */
  ssize_t szsent;        /* return value of call to ::writev */
  while (countiov >0) 
  {
    /* split iovector in packet at most of length UIO_MAXIOV */
    iovcnt = (countiov < UIO_MAXIOV ? countiov : UIO_MAXIOV);
    szonesent = 0;
    for (i=0; i<iovcnt; ++i)
      szonesent += iov[i].iov_len;

doit_again:
    szsent = ::writev(_fd, (iovec *) iov, iovcnt);
    if (szsent == szonesent) /* ok ! */
    {
      totalsent += szsent;
      iov       += iovcnt;
      countiov  -= iovcnt;
      continue;
    }
    
    /* error */
    if (szsent == -1) 
    {
      _error_no = errno;
      if (errno == EINTR) {
        goto doit_again;
      }
      _state |= S_FAIL;
      switch (errno) {
        case EPIPE: close(); 
        default:
          break;
      }
      return -1;
    }
    
    /* no error, but sent less than requested, restart sending 
       with update of iovvector.
    */
    szonesent -= szsent;
    totalsent += szsent;
    for (i=0; i<iovcnt; ++i)
    {
      if (szsent < (ssize_t)(iov[i].iov_len)) 
      {
        /* break during sending iov[i], resend the remainder and continue at iov[i+1] */
        size_t iov_len = iov[i].iov_len - szsent;
        char* iov_base = (char*)iov[i].iov_base + szsent;
        iov[i].iov_len  = iov_len;
        iov[i].iov_base = iov_base;
        break;
      }
      szsent -= iov[i].iov_len;
    }
    if (i < iovcnt) 
    {
      iovcnt -= i;
      iov += i;
      countiov -= i;
    }
    goto doit_again;
  }
  return totalsent;
}



// --------------------------------------------------------------------------
ssize_t SocketClient::writes( const std::string& s )
{
  if (!good()) return 0;
  ssize_t sz_write =0;
  ka::uint32_t sz;
  sz = (ka::uint32_t) s.size();
  sz = htonl( sz );
  sz_write = writen(&sz, sizeof(ka::uint32_t));
  if (!good()) return 0;
  sz = ntohl( sz );
  sz_write += writen(s.c_str(), sz*(ka::uint32_t)sizeof(char));
  return sz_write;
}



// --------------------------------------------------------------------------
ssize_t SocketClient::read(void* data, int size)
{
  if (!good()) return 0;
doit_again:
  _error_no = 0;
  ssize_t nbrecv = ::read(_fd, data, size);
  if (nbrecv<0) 
  {
    _error_no = errno;
    if (_error_no == EINTR) goto doit_again;
    _state |= S_FAIL;
    switch (_error_no) {
      case EPIPE: close();
      default:
        break;
    }
  }
  return nbrecv;
}



// --------------------------------------------------------------------------
ssize_t SocketClient::readn(void* data, int size)
{
  if (!good()) return 0;
  ssize_t nbrecv=0;
  ssize_t nbtorecv = size;
  ssize_t retval =0;
  while (nbtorecv >0)
  {
    nbrecv = (nbtorecv < SSIZE_MAX ? nbtorecv : SSIZE_MAX);
doit_again:
    _error_no = 0;
    nbrecv = ::read(_fd, data, nbrecv);
    if (nbrecv ==0) return retval;
    if (nbrecv<0) 
    {
      _error_no = errno;
      if (_error_no == EINTR) goto doit_again;
      _state |= S_FAIL;
      switch (_error_no) {
        case EPIPE: close();
        default:
          break;
      }
      return retval;
    }
    data = ((char*)data) + nbrecv;
    nbtorecv  -= nbrecv;
    retval += nbrecv;
  }
  return retval;
}



// --------------------------------------------------------------------------
ssize_t SocketClient::readv(ka::IOVectEntry* iov, int countiov)
{
  if (!good()) return 0;
  if (countiov ==0) return 0;

  int i, iovcnt;
  ssize_t totalrecv = 0; /* total number of bytes received */
  ssize_t szonerecv;     /* number of bytes to receive for one ::readv call */
  ssize_t szrecv;        /* return value of call to ::readv */
  while (countiov >0) 
  {
    /* split iovector in packet at most of length UIO_MAXIOV */
    iovcnt = countiov < UIO_MAXIOV ? countiov : UIO_MAXIOV;
    szonerecv = 0;
    for (i=0; i<iovcnt; ++i)
      szonerecv += iov[i].iov_len;
doit_again:
    szrecv = ::readv(_fd, (iovec *) iov, iovcnt);
    if (szrecv == szonerecv) /* ok ! */
    {
      totalrecv += szrecv;
      iov       += iovcnt;
      countiov  -= iovcnt;
      continue;
    }
    /* error */
    if (szrecv == -1) 
    {
      if (errno == EINTR) {
        goto doit_again;
      }
      _state |= S_FAIL;
      switch (errno) {
        case EPIPE: close();
        default:
          break;
      }
      return -1;
    }

    /* no error, but recv less than requested, restart receiving 
       with update of iovvector.
    */
    szonerecv -= szrecv;
    totalrecv += szrecv;
    for (i=0; i<iovcnt; ++i)
    {
      if (szrecv < (ssize_t)(iov[i].iov_len)) 
      {
        /* break during receiving iov[i], receive the remainder and continue at iov[i+1] */
        size_t iov_len = iov[i].iov_len - szrecv;
        char* iov_base = (char*)iov[i].iov_base + szrecv;
        iov[i].iov_len  = iov_len;
        iov[i].iov_base = iov_base;
        break;
      }
      szrecv -= iov[i].iov_len;
    }
    if (i < iovcnt) 
    {
      iovcnt   -= i;
      iov      += i;
      countiov -= i;
    }
    goto doit_again;
  }
  return totalrecv;
}



// --------------------------------------------------------------------------
ssize_t SocketClient::reads( std::string& s )
{
  if (!good()) return 0;
  ssize_t sz_read =0;
  ka::uint32_t sz;
  sz_read = readn(&sz, sizeof(ka::uint32_t));
  if (!good()) return 0;
  if (sz_read ==0) return 0;
  sz = ntohl( sz );
  char* c_path = (char*)alloca(sz+1);
  sz_read = readn(c_path, sz*(ka::uint32_t)sizeof(char));
  if (!good()) return sz_read;
  c_path[sz_read] = 0;
  s = std::string( c_path );
  return sz_read;
}



// --------------------------------------------------------------------------
void SocketClient::dup2( SocketClient& new_sock)
{
  _state &= ~S_FAIL;
  _error_no = 0;
  int  retval = ::dup2(_fd, new_sock._fd);
  if (retval == -1) 
  {
    _error_no = errno;
    _state |= S_FAIL;
    new_sock._state |= S_BAD;
    new_sock._state |= S_CLOSE;
    return;
  }
  new_sock._state= S_GOOD;
}


// --------------------------------------------------------------------------
void SocketClient::dup( SocketClient& new_sock)
{
  _state &= ~S_FAIL;
  _error_no = 0;
  int retval = ::dup(_fd);
  if (retval == -1) 
  {
    _error_no = errno;
    _state |= S_FAIL;
    new_sock._state |= S_BAD;
    new_sock._state |= S_CLOSE;
    return;
  }
  if (new_sock.get_fd() != -1) 
    new_sock.close();
  new_sock._state = S_CLOSE;
  new_sock.set_fd( retval );
}



// --------------------------------------------------------------------------
void SocketClient::pipe(SocketClient& s_r, SocketClient& s_w)
{
  int sv[2];
  s_r.close();
  s_w.close();
  s_w._error_no = s_r._error_no = 0;
  int sp = ::pipe( sv);
  if (sp !=0) {
    s_w._error_no = s_r._error_no = errno;
    s_r._state |= S_FAIL;
    s_w._state |= S_FAIL;
    return;
  }
  s_r._state = S_GOOD;
  s_w._state = S_GOOD;
  s_r.set_fd(sv[0]);
  s_w.set_fd(sv[1]);
}



// --------------------------------------------------------------------------
void SocketClient::sock_pair(int d, int type, int proto, SocketClient& one, SocketClient& two)
{
  int sv[2];
  one.close();
  two.close();
  one._error_no = two._error_no = 0;
  int sp = ::socketpair( d, type, proto, sv);
  if (sp !=0) {
    one._error_no = two._error_no = errno;
    one._state |= S_FAIL;
    two._state |= S_FAIL;
    return;
  }

  one._state = S_GOOD;
  two._state = S_GOOD;
  one.set_fd(sv[0]);
  two.set_fd(sv[1]);

  return;
}



// --------------------------------------------------------------------------
void SocketClient::sock_pair
  (SocketClient& one, SocketClient& two)
{
#if defined(__APPLE__) || defined(__linux__)
  sock_pair( AF_LOCAL,SOCK_STREAM, 0, one, two);
#elif defined(KAAPI_USE_SUNOS)
  sock_pair( AF_INET, SOCK_STREAM, 0, one, two);
#endif
}


}//namespace SOCK
