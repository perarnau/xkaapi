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
#include "kaapi_impl.h"  // sched_lock routines
#include "kanet_instr.h"

namespace ka {

// -----------------------------------------------------------------------
InstructionStream::InstructionStream() 
 : _start(0), _last(0), _capacity(0)
{
  _pos_w.write( 0 );
}

// -----------------------------------------------------------------------
InstructionStream::~InstructionStream() 
{
  kaapi_assert( _start == 0 );
}

// -----------------------------------------------------------------------
int InstructionStream::initialize(size_t capacity) throw()
{ 
  kaapi_sched_initlock(&_lock);
  _start    = new Instruction[capacity];
  _last     = _start + capacity;
  _capacity = (int32_t)capacity;
  _pos_w.write( 0 );
  _pos_r = 0;
  _tosend._state = SB_FREE;
  _tosend._start = new Instruction[capacity];;
  _tosend._last  = _tosend._start + capacity;
  _tosend._pos_w = 0;
  return 0;
}


// -----------------------------------------------------------------------
int InstructionStream::terminate() throw()
{ 
  if (_start !=0) delete [] _start;
  _start = _last = 0;
  _pos_w.write( 0);

  return 0;
}


// -----------------------------------------------------------------------
void InstructionStream::switch_buffer()
{
redo:
  kaapi_sched_lock(&_lock);
  if (!isfull()) 
  {
    kaapi_sched_unlock(&_lock);
    return;
  }
  if (_tosend._state != SB_FREE)
  {
    /* wait state becomes SB_FREE */
    kaapi_sched_unlock(&_lock);
    while (_tosend._state != SB_FREE)
      kaapi_slowdown_cpu();
    goto redo;
  }
  
  /* reset current buffer: the last op code with reset the _pos_w field ! */
  _tosend._state = SB_POSTED;
  _tosend.swap( *this );

  /* release lock and flush the buffer */
  kaapi_sched_unlock(&_lock);
  flush( _tosend._start + _tosend._pos_r, _tosend._start + _tosend._pos_w );
  kaapi_sched_lock(&_lock);
  _tosend._pos_r = 0;
  _tosend._pos_w = 0;
  _tosend._state = SB_FREE;
  kaapi_sched_unlock(&_lock);
}


// -----------------------------------------------------------------------
void InstructionStream::sync()
{
redo:
  while (_tosend._state != SB_FREE)
  {
    kaapi_network_poll();
    kaapi_slowdown_cpu();
  }
  kaapi_sched_lock(&_lock);
  if (_tosend._state != SB_FREE) 
  {
    kaapi_sched_unlock(&_lock);
    goto redo;
  }
  /* here _tosend is free, flush all message from start to _pos_w */

  /* reset current buffer: the last op code with reset the _pos_w field ! */
  int32_t pos_w = _pos_w.read();
  _tosend._state = SB_POSTED;
  _tosend._pos_w = pos_w;
  _tosend.swap( *this );
  _pos_r = pos_w;
  kaapi_sched_unlock(&_lock);
  flush( _tosend._start + _tosend._pos_r, _tosend._start + _tosend._pos_w );
  kaapi_sched_lock(&_lock);
  _tosend._pos_r = 0;
  _tosend._pos_w = 0;
  _tosend._state = SB_FREE;
  
  kaapi_sched_unlock(&_lock);
}

} // -namespace
