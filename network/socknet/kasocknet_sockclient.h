/* KAAPI internal interface */
 // ==========================================================================
// (c) INRIA, projet MOAIS, 2006
// see the copyright file for more information
// Author: T. Gautier
// - based on socket class from Inuktitut2 & 3, author : C. Martin and T. Gautier
// ==========================================================================
#ifndef _KAAPI_SN_CLIENT_H_
#define _KAAPI_SN_CLIENT_H_

#include "kasocknet_sockbasic.h"

namespace SOCKNET {

class FDSet;


// -------------------------------------------------------------------------
/** SocketClient interface
*/
class SocketClient : public Socket {
public:
  /** Constructor
   */
  SocketClient();

  /** Constructor on a given fd
   */
  SocketClient( int fd );

  /** connect to server on specified address.
      \return ComFailure::OK if no error
      \return ComFailure::TIMEOUT
      \return ComFailure::HOST_UNREACH
      \return other code
  */
  void connect( const Address& serveraddr );

  /** get peer address
      \param address [out] the returned address
  */
  void get_peer_address( Address& address);

  /** write methods
      If the sentry status is good, the method tries to generate
      whatever data is appropriate  for the type of method's arguments.
      On return to these methods: either that  fail bit is set, which indicates
      that the requested number of data to write was not reached, either
      the bad bit is set if the error is irrecoverable.
  */
  //@{
  /** write/send data to socket
      \param buf [in] data to be send
      \param size [in] length of bytes to be written
      \return the size of written data
  */
  ssize_t write(const void* buf, int size);

  /** write/send data to socket
      \param buf [in] data to be send
      \param size [in] length of bytes to be written
      \return the size of written data
  */
  ssize_t writen(const void* buf, int size );
  
  /** write/send data to socket.
      The field iov_len of iovector entry may be modified by the invocation.
      \param iov [in] iovector to be send
      \param count [in] number of entries in iov
      \return the size of written data
  */
  ssize_t writev(ka::IOVectEntry* iov, int count );

  /** Write a string
      \param s [in] the string to write
      \return the size of written data
  */
  ssize_t writes( const std::string& s );
  //@}

  /** Read/recv data to socket
      \param buf [in] data to read
      \param size [in] length of bytes to be read
      \return the size of read data
  */
  ssize_t read(void* buf, int size );

  /** Read/recv data to socket
      \param buf [in] data to read
      \param size [in] length of bytes to be read
      \return the size of read data
  */
  ssize_t readn(void* buf, int size );
  
  /** Read/recv data to socket
      The field iov_len of iovector entry may be modified by the invocation.
      \param iov [in] iovector to read
      \param count [in] number of entries in iov
      \return the size of read data
  */
  ssize_t readv(ka::IOVectEntry* iov, int count);

  /** Read a string
      \param s [in] the string to read
      \return the size of read data
  */
  ssize_t reads( std::string& s );

  /** create a copy of the socket
   * \arg new_sock  new_sock is the copy of the socket
   */
  void dup(SocketClient& new_sock);

  /** create a copy of the socket
   * \arg new_sock  new_sock is the copy of the socket
   */
  void dup2(SocketClient& new_sock);

  /** Same a those in the namespace
   */
  static void pipe(SocketClient& s_r, SocketClient& s_w);
  static void sock_pair(SocketClient& one, SocketClient& two);
  static void sock_pair(int d, int type, int proto, SocketClient& one, SocketClient& two);

protected:
  friend class FDSet;
};

inline void sock_pair(SocketClient& one, SocketClient& two)
{ return SocketClient::sock_pair( one, two); }

inline void sock_pair (int d, int type, int proto, SocketClient& one, SocketClient& two)
{ return SocketClient::sock_pair( d, type, proto, one, two); }

}

#endif

