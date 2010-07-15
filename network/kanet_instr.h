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
#ifndef _KANETWORK_INSTR_H
#define _KANETWORK_INSTR_H

#include "kaapi.h"

namespace Net {

class Channel;
typedef void (*Callback_fnc)(int errocode, Channel* ch, void* userarg);

// -----------------------------------------------------------------------
/** \name Callback
    \ingroup Net
    A call back is a function + argument to be call at the end of an
    asynchronous operation.
    The function as a specific signature in order to report error code
    associated to the asynchronous operation.
      - typedef void (*Callback_fnc)(int errocode, Channel* ch, void* userarg);
    The errocode and ch parameters are set by the runtime to report the error
    (or 0 if no error) and the channel object used to send message, or 0 if no
    channel was used. The last parameter is the user argument given at the
    creation of the callback.
    
*/
struct Callback {
  Callback_fnc cbk;  /// Type of the call back function
  void*        arg;  /// The argument of the callback function 
};


// -----------------------------------------------------------------------
/** \name RWInstruction
    \ingroup Net
    Remote write operation. 
    The operation is asynchronous and may be attached with a callback.
    The local pointer points in the address space of the process.
    The remote pointer is defined by the C XKaapi header and is always
    a displacement inside a (remote) memory region.
    
    WARNING: we do not consider heterogeneous architecture. size is valid
    on the local node as well as in the remote node.
*/
struct RWInstruction {
  kaapi_uint64_t     dptr;  /// remote pointer / displacement where to write
  void*           lptr;  /// pointer on the local bloc to send
  kaapi_uint32_t     size;  /// size in bytes of the pointed memory
};


// -----------------------------------------------------------------------
/** \name AMInstruction
    \ingroup Net
    This is an active message.
*/
struct AMInstruction {
  kaapi_uint64_t handler;  /// handler of the service (function to call)
  void*       lptr;     /// points on the bloc of data for the AM
  kaapi_uint32_t size;     /// size in byte of the pointed memory
};


// -----------------------------------------------------------------------
/** \name WBarrierInstruction
    \ingroup Net
    This is a write memory barrier instruction. All previous messages or
    RDMA operations previoulsy posted in the instruction stream will be
    viewed on the remote node before all write operation posted after the
    barrier.
    The memory barrier is not a synchronisation operation but only an
    operation in order to ordered the write in the memory of the remote node.
    
    This is an asynchronous operation that can be associated with a callback.
*/
struct WBarrierInstruction {
};



// -----------------------------------------------------------------------
/** \name Instruction
    \ingroup Net
    An instruction is a struct / union basic instructions. 
    An instruction could always be attached to a callback in order to signal
    the completion of the local effect of the operation. For instance, if
    a callback is attached with a AM instruction, the callback will be called
    after the message as been insert into the channel.
*/
struct Instruction {
  /** */
  enum Type {
    INST_VOID =0,
    INST_NOP,
    INST_RWDMA,
    INST_AM,
    INST_WB
  };
  
  /** */
  Instruction() : type(INST_NOP) {}

  /** */
  Type  type;

  /** */
  union {
    RWInstruction       i_rw;
    AMInstruction       i_am;
    WBarrierInstruction i_wb;
  };
  Callback              i_cbk;
};


// -----------------------------------------------------------------------
/** \name InstructionStream
    \ingroup Net
    
    An InstructionStream serves to build sequence of messages.
    An instruction stream is attached to a stream of data that can be used
    to store data passed by copy.
*/
class InstructionStream {
public:
  /** Insert a Remote Write DMA operation.
  */
  void insert_rwdma(   
      kaapi_uint64_t     dptr,  /// remote pointer / displacement where to write
      void*           lptr,  /// pointer on the local bloc to send
      kaapi_uint32_t     size   /// size in bytes of the pointed memory
  );

  /** Insert a Remote Write DMA operation with callback
  */
  void insert_rwdma(   
      kaapi_uint64_t     dptr,  /// remote pointer / displacement where to write
      void*           lptr,  /// pointer on the local bloc to send
      kaapi_uint32_t     size,  /// size in bytes of the pointed memory
      Callback_fnc    cbk,   /// Type of the call back function
      void*           arg    /// The argument of the callback function 
  );
  
  /** Insert a AM operation.
  */
  void insert_am(   
      kaapi_uint64_t     handler,/// handler of the service (function to call)
      void*           lptr,   /// points on the bloc of data for the AM
      kaapi_uint32_t     size    /// size in byte of the pointed memory
  );

  /** Insert a AM operation with call back
  */
  void insert_am(   
      kaapi_uint64_t     handler,/// handler of the service (function to call)
      void*           lptr,   /// points on the bloc of data for the AM
      kaapi_uint32_t     size,   /// size in byte of the pointed memory
      Callback_fnc    cbk,    /// Type of the call back function
      void*           arg     /// The argument of the callback function 
  );

  /** Insert a Write Memory Barrier
  */
  void insert_wb( );

  /** Insert a Write Memory Barrier with callback
  */
  void insert_wb(   
      Callback_fnc    cbk,    /// Type of the call back function
      void*           arg     /// The argument of the callback function 
  );

  /** Insert a nop
  */
  void insert_nop( );

  /** Insert a nop with callback
  */
  void insert_nop(   
      Callback_fnc    cbk,    /// Type of the call back function
      void*           arg     /// The argument of the callback function 
  );

protected:
  InstructionStream();

  virtual ~InstructionStream();
  
  /** initialize the InstructionStream object.
  */
  int initialize( ) throw();

  /** Should be call to clear and delete internal state of the object.
  */
  int terminate() throw();

protected:
  Instruction*    _start;      /// first instruction in the buffer
  Instruction*    _last;       /// past the last instruction in the buffer

  kaapi_atomic_t  _pos_w __attribute__((aligned(8))); /// next position for writing an entry in _start

  kaapi_uint32_t  _pos_r;      /// next position for reading an entry in _start
};


// -----------------------------------------------------------------------
/*
 * inline definition
 */
inline void InstructionStream::insert_rwdma(   
      kaapi_uint64_t     dptr,
      void*              lptr,
      kaapi_uint32_t     size 
  )
{
  kaapi_uint32_t posw = KAAPI_ATOMIC_INCR( &_pos_w ) -1; /* increment (atomic) and get the previous value */
  kaapi_assert( posw < _last-_start);
  _start[posw].i_rw.dptr = dptr;
  _start[posw].i_rw.lptr = lptr;
  _start[posw].i_rw.size = size;
  _start[posw].i_cbk.cbk = 0;
  _start[posw].i_cbk.arg = 0;
  kaapi_writemem_barrier();

  _start[posw].type = Instruction::INST_RWDMA;
}

inline void InstructionStream::insert_rwdma(   
      kaapi_uint64_t     dptr,
      void*              lptr,
      kaapi_uint32_t     size, 
      Callback_fnc       cbk,
      void*              arg 
  )
{
  kaapi_uint32_t posw = KAAPI_ATOMIC_INCR( &_pos_w ) -1; /* increment (atomic) and get the previous value */
  kaapi_assert( posw < _last-_start);
  _start[posw].i_rw.dptr = dptr;
  _start[posw].i_rw.lptr = lptr;
  _start[posw].i_rw.size = size;
  _start[posw].i_cbk.cbk = cbk;
  _start[posw].i_cbk.arg = arg;
  kaapi_writemem_barrier();

  _start[posw].type = Instruction::INST_RWDMA;
}

inline void InstructionStream::insert_am(   
      kaapi_uint64_t     handler,
      void*              lptr,
      kaapi_uint32_t     size
  )
{
  kaapi_uint32_t posw = KAAPI_ATOMIC_INCR( &_pos_w ) -1; /* increment (atomic) and get the previous value */
  kaapi_assert( posw < _last-_start);
  _start[posw].i_am.handler = handler;
  _start[posw].i_am.lptr    = lptr;
  _start[posw].i_am.size    = size;
  _start[posw].i_cbk.cbk    = 0;
  _start[posw].i_cbk.arg    = 0;
  kaapi_writemem_barrier();

  _start[posw].type = Instruction::INST_AM;
}

inline void InstructionStream::insert_am(   
      kaapi_uint64_t     handler,
      void*              lptr,
      kaapi_uint32_t     size, 
      Callback_fnc       cbk,
      void*              arg 
  )
{
  kaapi_uint32_t posw = KAAPI_ATOMIC_INCR( &_pos_w ) -1; /* increment (atomic) and get the previous value */
  kaapi_assert( posw < _last-_start);
  _start[posw].i_am.handler = handler;
  _start[posw].i_am.lptr    = lptr;
  _start[posw].i_am.size    = size;
  _start[posw].i_cbk.cbk    = cbk;
  _start[posw].i_cbk.arg    = arg;
  kaapi_writemem_barrier();

  _start[posw].type = Instruction::INST_AM;
}


inline void InstructionStream::insert_wb()
{
  kaapi_uint32_t posw = KAAPI_ATOMIC_INCR( &_pos_w ) -1; /* increment (atomic) and get the previous value */
  kaapi_assert( posw < _last-_start);
  _start[posw].i_cbk.cbk    = 0;
  _start[posw].i_cbk.arg    = 0;
  kaapi_writemem_barrier();

  _start[posw].type = Instruction::INST_WB;
}

inline void InstructionStream::insert_wb(   
      Callback_fnc    cbk,
      void*           arg 
  )
{
  kaapi_uint32_t posw = KAAPI_ATOMIC_INCR( &_pos_w ) -1; /* increment (atomic) and get the previous value */
  kaapi_assert( posw < _last-_start);
  _start[posw].i_cbk.cbk    = cbk;
  _start[posw].i_cbk.arg    = arg;
  kaapi_writemem_barrier();

  _start[posw].type = Instruction::INST_WB;
}

inline void InstructionStream::insert_nop()
{
  kaapi_uint32_t posw = KAAPI_ATOMIC_INCR( &_pos_w ) -1; /* increment (atomic) and get the previous value */
  kaapi_assert( posw < _last-_start);
  _start[posw].i_cbk.cbk    = 0;
  _start[posw].i_cbk.arg    = 0;
  kaapi_writemem_barrier();

  _start[posw].type = Instruction::INST_NOP;
}

inline void InstructionStream::insert_nop(   
      Callback_fnc    cbk,
      void*           arg 
  )
{
  kaapi_uint32_t posw = KAAPI_ATOMIC_INCR( &_pos_w ) -1; /* increment (atomic) and get the previous value */
  kaapi_assert( posw < _last-_start);
  _start[posw].i_cbk.cbk    = cbk;
  _start[posw].i_cbk.arg    = arg;
  kaapi_writemem_barrier();

  _start[posw].type = Instruction::INST_NOP;
}

} // -namespace

#endif // 
