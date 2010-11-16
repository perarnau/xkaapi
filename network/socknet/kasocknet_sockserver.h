/* KAAPI internal interface */
// ==========================================================================
// (c) INRIA, projet MOAIS, 2006
// see the copyright file for more information
// Author: T. Gautier
// - based on socket class from Inuktitut2 & 3, author : C. Martin and T. Gautier
// ==========================================================================
#ifndef _KAAPI_SN_SERVER_H_
#define _KAAPI_SN_SERVER_H_

#include "kasocknet_sockclient.h"

namespace SOCKNET {

// -------------------------------------------------------------------------
/**
*/
class SocketServer: public Socket {
private:
  /** Constructor using a already opened and good file descriptor
  */
  SocketServer(int fd)
   : Socket()
  { set_fd(fd); }

public:

  /** Constructor
  */
  SocketServer();

  /** bind socket to the associated address.
   */
  void bind();

  /** listen for connections on socket
   * \return zero on success, else -1
   */
  void listen(int backlog);

  /** wait for incoming connection, return Socket object for client.
   *  Blocks until a client has connected
   *  \return socket object, connected to the computer that requested a
   *          connection
   */
  void accept( SocketClient& sc );

private:
};

} // namespace SOCK

#endif

