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
#include "kagasnet_device.h"

namespace GASNET {

// --------------------------------------------------------------------
void* Device::bind( uintptr_t addr, size_t size )
{
  return (void*) (((char*)_segaddr)+addr);
}

// --------------------------------------------------------------------
void* Device::allocate( size_t size )
{
#if 0
  printf("%i:[GASNET::Device::allocate] IN memory allocate, addr@:%p, sp:%lu, size:%lu\n", 
      _wcom_rank,
      (void*)_segaddr,
      _segsp,
      _segsize
  );
  fflush(stdout);
#endif

  uintptr_t sp = _segsp;
  if (sp + size >= _segsize) return 0;
  _segsp = (_segsp + size + sizeof(double)) & ~(sizeof(double)-1);

#if 0
  printf("%i:[GASNET::Device::allocate] allocate @:%p, size:%lu\n", 
      _wcom_rank,
      (void*) (((char*)_segaddr)+sp),
      size
  );
  fflush(stdout);
#endif
  return (void*) (((char*)_segaddr)+sp);
}


// --------------------------------------------------------------------
void Device::deallocate( void* addr )
{
}


// --------------------------------------------------------------------
ka::SegmentInfo Device::get_seginfo( ka::GlobalId gid ) const
{
  if (gid >= (ka::GlobalId)_wcom_size) 
    return ka::SegmentInfo();
  
  return ka::SegmentInfo( (uintptr_t)_seginfo[gid].addr, _seginfo[gid].size );
}


} // - namespace Net...
