/* KAAPI internal interface */
// ==========================================================================
// (c) INRIA, projet MOAIS, 2006
// see the copyright file for more information
// Author: T. Gautier
// - based on socket class from Inuktitut2 & 3, author : C. Martin and T. Gautier
// ==========================================================================
#ifndef _KAAPI_SN_BASIC_H_
#define _KAAPI_SN_BASIC_H_

#include "kasocknet_error.h"
#include "kanet_types.h"
#include <sys/socket.h>

namespace SOCKNET {

// -------------------------------------------------------------------------
/** Address class : store an IPv4 address for socket endpoints
*/
class Address {
public:
  /*! Constructor, no initialization
  */
  Address();

  /** copy-constructor,initialize with specified address
   */
  Address(const Address& a);

  /** Destructor
   */
  ~Address();

  /** Assignement
   */
  Address& operator=(const Address& a);

  /** initialize with host
   *  \return the error code
   */
  ComFailure::Code set_byname( const std::string& host );

  /** initialize with any address
   *  \return reference to initialized object
   */
  void set_toany(  );

  /** Return a textual form from the adress: "socknet:<port>:<ip>"
   *  \return as string
   */
  std::string to_string() const;

  /** return the adress from textual (ip+port)
   *  \return error code
   */
  ComFailure::Code from_string(const std::string& s);

  /** print address status
   */
  std::ostream& print(std::ostream& ) const;

  /** set port of address
   *  \warning changing the port after connecting will not change the port your
   *           connection is on, but modify data wich will lead to inconsistency
   */
  void set_port(ka::uint16_t p);

  /** return port number of address
   *  \return port in host byte ordered
   */
  ka::uint16_t get_port() const;

  /** return address IPv4 
   *  \return IPV4 address in host byte ordered
   */
  ka::int32_t get_ipv4() const;

  /** Set the address IPv4 
   *  \return IPV4 address in host byte ordered
   */
  void set_ipv4(ka::int32_t  a);

  /** return hostname
   *  \return host name
   */
  std::string get_hostname() const;

protected: 
  /** return host address information
   *  \return sockaddr_in struct (system dependent structure to store address)
   *  \warning this is data inside the class, be carefull not to overwrite !
   */
  const sockaddr* get_sockaddr() const;
  sockaddr* get_sockaddr();

  /**  
   *   \return the size of the addr
   */
  int get_sockaddr_size() const;

private:
  sockaddr_in _sockaddr;

  friend class Device;
  friend class Socket;
  friend class SocketClient;
  friend class SocketServer;
};



// -------------------------------------------------------------------------
/** Base class for Client and Server Socket endpoints
*/
class Socket {
private:
  /** Recopy Constructor
  */
  Socket(const Socket& ) 
  {}

  /** Assignment operator
  */
  Socket& operator=(const Socket& )
  { return *this; }

public:
  /** Default constructor
  */
  Socket();

  /** Destructor
  */
  ~Socket(){}

  /** Initialize the socket: create the fd
  */
  void initialize();

  /** close socket
  */
  void close();

  enum SHUT_MODE {
    READ_P  = SHUT_RD,
    WRITE_P = SHUT_WR,
    READWRITE_P = SHUT_RDWR
  };
  
  /** shutdown socket
      \param mode READ_P : shutdown the read-half part of the socket
      \param mode WRITE_P : shutdown the write-half part of the socket
      \param mode READWRITE_P : shutdown both part of the socket
  */
  void shutdown(SHUT_MODE mode);

  /** Return true iff the socknet status is good
  */
  bool good() const;

  /** Return true iff the last operation does not complete properly
      The last error is returned by error_no()
  */
  bool fail() const;

  /** Return true iff the socket is in an irrecoverable due to an error
  */
  bool bad() const;

  /** Return the content of error_no which stores the last error
  */
  int error_no() const;

  /** initialize from already existing socket
   *  \arg fd valid file-descriptor
   *  \warning initialization may fail, check isConnected() afterwards
   *  \warning using this constructor will result in loss of the socket origin!
   */
  void set_fd( int fd );

  /**
   */
  int get_fd() const;
	
  /**
  */
  void set_address(const Address& addr);

  /**
   */
  std::string to_string() const;

  /**
   */
  const Address& get_address() const;

  /**
   */
  Address& get_address();

  /**
   * values possible for the argument opt of setSockOpt
   */
  enum SocketOptions { 
    no_delay = 0, 
    clo_exec, 
    linger, 
    reuseaddr,
    rcvbuf,
    sndbuf,
    rcvlowat,
    sndlowat,
    nosigpipe,
    dontroute,
    nonblock
  };

  /**
   * permit to manipulate the socket options TCP_NODELAY, SO_LINGER, FD_CLOEXEC, SO_REUSEADDR,
   * SO_RCVBUF, SO_SNDBUF, SO_RCVLOWAT, SO_SNDLOWAT, SO_NOSIGPIPE, SO_DONTROUTE and O_NONBLOCK
   * \arg opt option (no_delay, clo_exec, reuseaddr)
   * \arg value
   */
  void set_sockopt(int opt, int value);

  /**
   * return the socket options TCP_NODELAY, SO_LINGER, FD_CLOEXEC, SO_REUSEADDR, SO_RCVBUF, SO_SNDBUF,
   * SO_RCVLOWAT, SO_SNDLOWAT, SO_NOSIGPIPE, SO_DONTROUTE and O_NONBLOCK
   * \arg opt option (no_delay, clo_exec, reuseaddr, ...)
   * \arg value
   */
  void get_sockopt(int opt, int& value);

  /**
   * print the values of the socket
   */
  std::ostream& print(std::ostream&) const;

protected:
  enum State
    { 
      S_GOOD                = 0,
      S_BAD                 = 1L << 0,
      S_FAIL                = 1L << 1,
      S_CLOSE               = 1L << 2,  ///< 0 iff socket is connected && _fd !=-1, 1 if it is closed
    };

  Address _addr;
  int     _fd;
  int     _state;       ///< State of the socket
  int     _error_no;    ///< last error number errno if !good()
};


/*
 * inline definitions
 */
/*
 * inline definition
 */
inline std::ostream& operator << (std::ostream& o, const Address& a )
{ return a.print(o); }

inline const sockaddr* Address::get_sockaddr() const
{ return (const sockaddr*)&_sockaddr; }

inline sockaddr* Address::get_sockaddr() 
{ return (sockaddr*)&_sockaddr; }

inline int Address::get_sockaddr_size() const
{ return sizeof(sockaddr); }

inline bool Socket::good() const
{ return _state == S_GOOD; }

inline bool Socket::fail() const
{ return (_state & (S_FAIL | S_BAD)) !=0; }

inline bool Socket::bad() const
{ return (_state & S_BAD) != 0; }

inline int Socket::get_fd() const 
{ return _fd; }
  
inline void Socket::set_address(const Address& addr) 
{ _addr = addr; }

inline std::string Socket::to_string() const 
{ return _addr.to_string(); }

inline const Address& Socket::get_address() const 
{ return _addr; }

inline Address& Socket::get_address()  
{ return _addr; }

inline std::ostream& operator << (std::ostream& o, const Socket& s )
{ return s.print(o); }


} //namespace SOCKNET

#endif

