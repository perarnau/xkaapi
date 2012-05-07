/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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

/* exported function to user land... */
int kaapi_api_listrequest_iterator_count(
  struct kaapi_listrequest_iterator_t* lrrange
)
{ return (int)kaapi_listrequest_iterator_count(lrrange); }

kaapi_request_t* kaapi_api_listrequest_iterator_get( 
  struct kaapi_listrequest_t* lrequests, struct kaapi_listrequest_iterator_t* lrrange 
)
{ return kaapi_listrequest_iterator_get(lrequests, lrrange); }

kaapi_request_t* kaapi_api_listrequest_iterator_next( 
  struct kaapi_listrequest_t* lrequests, struct kaapi_listrequest_iterator_t* lrrange 
)
{ return kaapi_listrequest_iterator_next( lrequests, lrrange ); }

int kaapi_listrequest_api_iterator_empty( 
  struct kaapi_listrequest_iterator_t* lrrange
)
{ return kaapi_listrequest_iterator_empty( lrrange ); }
