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


#if !defined(KAAPI_USE_STATICSCHED)
int kaapi_threadgroup_create(kaapi_threadgroup_t* thgrp, int size, 
  void (*mapping)(void*, int /*nodecount*/, int /*tid*/, kaapi_globalid_t* /*gid*/, unsigned int /*proctype*/),
  void* ctxt_mapping
)
{
  return 0;
}

/**
*/
int kaapi_threadgroup_begin_partition(kaapi_threadgroup_t thgrp, int flag )
{
  return 0;
}


/**
*/
void kaapi_threadgroup_force_archtype(kaapi_threadgroup_t group, unsigned int part, unsigned int type)
{
}

/**
*/
void kaapi_threadgroup_force_kasid(kaapi_threadgroup_t group, unsigned int part, unsigned int arch, unsigned int user)
{
}

/**
*/
int kaapi_threadgroup_set_iteration_step(kaapi_threadgroup_t thgrp, int maxstep )
{
  return 0;
}

/** Check and compute dependencies for task 'task' to be pushed into the i-th partition.
    On return the task is pushed into the partition if it is local for the execution.
    
    \return EINVAL if the task is not pushed 
*/
int kaapi_threadgroup_computedependencies(kaapi_threadgroup_t thgrp, int partitionid, kaapi_task_t* task)
{
  return 0;
}


/**
*/
int kaapi_threadgroup_end_partition(kaapi_threadgroup_t thgrp )
{
  return 0;
}


/**
*/
int kaapi_threadgroup_begin_execute(kaapi_threadgroup_t thgrp )
{
  return 0;
}


/**
*/
int kaapi_threadgroup_end_execute(kaapi_threadgroup_t thgrp )
{
  return 0;
}



/** Memory synchronization with copies to the original memory
*/
int kaapi_threadgroup_synchronize(kaapi_threadgroup_t thgrp )
{
  return 0;
}


/**
    \retval 0 in case of success
    \retval EBUSY if threads are already attached to the group
*/
int kaapi_threadgroup_destroy(kaapi_threadgroup_t thgrp )
{
  return 0;
}


/**
*/
int kaapi_threadgroup_print(FILE* file, kaapi_threadgroup_t thgrp )
{
  return 0;
}


/**
*/
int kaapi_threadgroup_save(kaapi_threadgroup_t thgrp )
{
  return 0;
}


/**
*/
int kaapi_threadgroup_restore(kaapi_threadgroup_t thgrp )
{
  return 0;
}


#endif