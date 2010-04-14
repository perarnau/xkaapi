/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
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
#include "kaapi_impl.h"
#include "athapascan-1.h"
#include <map>


namespace a1 {

static std::map<std::string,KaapiMonotonicBoundRep*> allmb;


class KaapiMonotonicBoundRep {
public:
  kaapi_atomic_t        lock;
  void*                 data;
  const kaapi_format_t* fmtdata;
  const kaapi_format_t* fupdate;
  kaapi_uint64_t        version;
};


/**
*/
void KaapiMonotonicBound::initialize( 
  const std::string& name, 
  void* value, 
  const Format* format, 
  const Format* fupdate
)
{
  KaapiMonotonicBoundRep* mbrep = new KaapiMonotonicBoundRep;
  KAAPI_ATOMIC_WRITE(&mbrep->lock, 0);
  mbrep->data    = value;
  mbrep->fmtdata = format->get_c_format();
  mbrep->fupdate = fupdate->get_c_format();
  mbrep->version = 0;

  _data      = value;
  _fmtdata   = format->get_c_format();
  _fmtupdate = fupdate->get_c_format();
  _reserved  = mbrep;

  allmb.insert( std::make_pair(name, mbrep) );
}



/**
*/
void KaapiMonotonicBound::terminate()
{
}


/**
*/
void KaapiMonotonicBound::acquire()
{
  KaapiMonotonicBoundRep* rep = _reserved;
  /* \TODO: active lock, using pthread lock if pthread is part of xkaapi */
  while (!KAAPI_ATOMIC_CAS(&rep->lock, 0, 1 )) ;
  kaapi_readmem_barrier();
  if (_data != rep->data) 
  {
    if (_data !=0)
    {
      _fmtdata->dstor( _data );
      free(_data);
    }
    _data = rep->data;
    _fmtdata = rep->fmtdata;
  }
}



/**
*/
void KaapiMonotonicBound::release()
{
  KaapiMonotonicBoundRep* rep = _reserved;
  if (_data != rep->data) 
  {
    if (rep->data !=0)
    {
      rep->fmtdata->dstor( rep->data );
      free(rep->data);
    }
    rep->data = _data;
    rep->fmtdata = _fmtdata;
  }
  kaapi_writemem_barrier();
  KAAPI_ATOMIC_WRITE(&rep->lock, 0);
}



/**
*/
const void* KaapiMonotonicBound::read() const
{
  return _data;
}



/**
*/
void KaapiMonotonicBound::update(const void* value, const Format* fmtvalue)
{
  throw;
#if 0 //TODO
  (*fmtvalue->update_mb)( _data, _fmtdata->get_c_format(), value, fmtvalue->get_c_format() );
#endif
}


} // namespace