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
#ifndef _KANETWORK_IOCHANNEL_H
#define _KANETWORK_IOCHANNEL_H

#include "kanet_instr.h"

namespace Net {

typedef kaapi_uint64_t GlobalId;

// --------------------------------------------------------------------------
/** \name Channel
    \ingroup Net
    
    A Channel is a communication channel between multiple 2 peers.
    The local peer and the remote peer.
*/
class Channel {
public:
  /** Return the globalid of the remote node to this channel
  */
  GlobalId get_peer_globalid() const;
  
  /** Return the url of the remote node to this channel
      This url correspond to the network address in order to be able
      to make connexion to this remote node.
      This is different from the peer_address with socket.
  */
  const char* get_peer_url() const;
  
protected:
  /** Cstor
  */
  Channel();

  /** Dstor
  */
  virtual ~Channel();
  
  /** Intialisation method
  */
  virtual int initialize( ) throw();
  
  /** Destructor
  */
  virtual int terminate() throw();
    
  /** Set the global id of the peer node
  */
  void set_peer_globalid( GlobalId gid);

  /** Set the url of the peer node
  */
  void set_peer_url( const char* url );
  
protected:
  char*        _peer_url;      /* url of the peer node */
  GlobalId     _peer_gid;      /* global identifier of the peer node */
};



// -----------------------------------------------------------------------
/** \name OutChannel
    \ingroup Net
    
    An OutChannel allows to decode a sequence of IO instructions in order
    to send message to a remote peer node.
*/
class OutChannel : virtual public Channel, public InstructionStream  {
public:
  /** Default constructor
  */
  int initialize() throw();
  
  /** Destructor
  */
  int terminate() throw();

protected:
  /** Client stub to write message from the InstructionStream
      An error with an error code is thrown in case of error.
  */
  virtual int stub( ) =0;
};


// -----------------------------------------------------------------------
/** \name InChannel
    \brief 
    \ingroup Net
*/
class InChannel : virtual public Channel {
public:
  /** Server stub to read message from a device.
      The call should call do_call on the callstack object in order to invoke the services
  */
  virtual int skel( ) =0;
};


/*
 * inline definition
 */
inline GlobalId Channel::get_peer_globalid() const
{ return _peer_gid; }

inline void Channel::set_peer_globalid( GlobalId gid)
{ _peer_gid = gid; }

inline const char* Channel::get_peer_url() const
{ return _peer_url; }


} // -namespace

#endif // 
