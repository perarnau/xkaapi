/*
** xkaapi
** 
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** xavier.besseron@imag.fr
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
/* Correct 2 bugs:
- In pack_ibegin :
     1. suspend_producer
     2. allocate_data
  -> suspend_producer ne garanti pas un emplacement libre apres le allocate_data
  Semble etre la cause des erreurs de transimission : oubli d'un paquet apres aggregation

- ajout d'un champ _pos_last qui est la derniere instruction a lire pour
  mise a jour dans Channel::stub et utilise pour deplacer le pointeur _pos_r 
  dans notify. Avant: lecture de _pos_close comme marqueur de fin qui aurait pu
  changer de valeur entre chaque lecture !
*/
#include <iostream>
#include "kanet_instr.h"

namespace Net {

// -----------------------------------------------------------------------
InstructionStream::InstructionStream() 
 : _start(0), _last(0), _pos_r(0)
{
  KAAPI_ATOMIC_WRITE(&_pos_w, 0 );
}

// -----------------------------------------------------------------------
InstructionStream::~InstructionStream() 
{
  kaapi_assert( _start == 0 );
}

// -----------------------------------------------------------------------
int InstructionStream::initialize() throw()
{ 
  _start      = new Instruction[1024];
  _last = _start + 1024;
  KAAPI_ATOMIC_WRITE(&_pos_w, 0 );
  _pos_r      = 0;

  return 0;
}


// -----------------------------------------------------------------------
int InstructionStream::terminate() throw()
{ 
  if (_start !=0) delete [] _start;
  _start = _last = 0;
  KAAPI_ATOMIC_WRITE(&_pos_w, 0);
  _pos_r = 0;

  return 0;
}

#if 0

// -----------------------------------------------------------------------
bool InstructionStream::empty() const
{ return (_pos_r == _pos_close) && !_full; }


// -----------------------------------------------------------------------
bool InstructionStream::full() const
{ return (_pos_w == _pos_r) && _full; }


// -----------------------------------------------------------------------
void InstructionStream::suspend_producer()
{
  KAAPI_LOG( false,
               "[Suspend producer]" 
            << ", pos_r @=" << (void*)_pos_r 
            << ", last_r @=" << (void*)_pos_last
            << ", pos_w @=" << (void*)_pos_w 
            << ", pos_close @=" << (void*)_pos_close 
            << ", full? =" << (_full ? "yes" : "no") 
            << ", full? =" << (_full ? "yes" : "no") 
  );
  
  if (!full()) return;
  Util::CriticalGuard critical( *_sync );
  _wait_prod = true;
  _manager->get_network()->wakeup_iodaemon();
  while (_wait_prod) _cond_prod.wait( *_sync );
}
#endif

} // -namespace
