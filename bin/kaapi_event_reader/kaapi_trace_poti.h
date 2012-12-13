/*
 ** xkaapi
 **
 **
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 **
 ** Thierry Gautier, thierry.gautier@inrialpes.fr
 ** Joao.Lima@imagf.r / joao.lima@inf.ufrgs.br
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

#ifndef _KAAPI_TRACE_POTI_H_
#define _KAAPI_TRACE_POTI_H_

#include "poti.h"

//pajeOpen
static inline int kaapi_trace_poti_Open(const char* filename)
{
  return poti_open(filename);
}

//pajeClose
static inline void kaapi_trace_poti_Close(void)
{
  poti_close();
}

//pajeHeader
static inline void kaapi_trace_poti_Header( int basic, int old_header)
{
  poti_header(basic, old_header);
}

//pajeCreateContainer
static inline void kaapi_trace_poti_CreateContainer(double timestamp,
                         const char *alias,
                         const char *type,
                         const char *container,
                         const char *name)
{
  poti_CreateContainer(timestamp, alias, type, container, name);
}

//pajeDestroyContainer
static inline void kaapi_trace_poti_DestroyContainer(double timestamp,
                          const char *type,
                          const char *container)
{
  poti_DestroyContainer(timestamp, type, container);
}

//pajeDefineContainerType
static inline void kaapi_trace_poti_DefineContainerType(const char *alias,
                             const char *containerType,
                             const char *name)
{
  poti_DefineContainerType(alias, containerType, name);
}

//pajeDefineEntityValue
static inline void kaapi_trace_poti_DefineEntityValue(const char *alias,
                           const char *entityType,
                           const char *name,
                           const char *color)
{
  poti_DefineEntityValue(alias, entityType, name, color);
}

//pajeDefineStateType
static inline void kaapi_trace_poti_DefineStateType(const char *alias,
                         const char *containerType,
                         const char *name)
{
  poti_DefineStateType(alias, containerType, name);
}

//pajeDefineLinkType
static inline void kaapi_trace_poti_DefineLinkType(const char *alias,
                        const char *containerType,
                        const char *sourceContainerType,
                        const char *destContainerType,
                        const char *name)
{
  poti_DefineLinkType(alias, containerType, sourceContainerType, destContainerType, name);
}

//pajeDefineEventType
static inline void kaapi_trace_poti_DefineEventType(const char *alias,
                         const char *containerType,
                         const char *name,
                         const char *color)
{
  poti_DefineEventType(alias, containerType, name);
}

//pajeStartLink
static inline void kaapi_trace_poti_StartLink(double timestamp,
                   const char *container,
                   const char *type,
                   const char *sourceContainer,
                   const char *value,
                   const char *key)
{
  poti_StartLink(timestamp, container, type, sourceContainer, value, key);
}

//pajeEndLink
static inline void kaapi_trace_poti_EndLink(double timestamp,
                 const char *container,
                 const char *type,
                 const char *endContainer,
                 const char *value,
                 const char *key)
{
  poti_EndLink(timestamp, container, type, endContainer, value, key);
}


static inline void kaapi_trace_poti_SetState(double timestamp,
                  const char *container,
                  const char *type,
                  const char *value)
{
  poti_SetState(timestamp, container, type, value);
}

//pajePushState
static inline void kaapi_trace_poti_PushState(double timestamp,
                   const char *container,
                   const char *type,
                   const char *value)
{
  poti_PushState(timestamp, container, type, value);
}

//pajePopState
static inline void kaapi_trace_poti_PopState(double timestamp,
                  const char *container,
                  const char *type)
{
  poti_PopState(timestamp, container, type);
}

//pajeNewEvent
static inline void kaapi_trace_poti_NewEvent(double timestamp,
                  const char *container,
                  const char *type,
                  const char *value )
{
  poti_NewEvent(timestamp, container, type, value);
}

#endif /* _KAAPI_TRACE_POTI_H_ */