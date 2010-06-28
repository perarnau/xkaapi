/*
** kaapi_procinfo.h
** 
** Created on Jun 23 2010
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


#ifndef KAAPI_PROCINFO_H_INCLUDED
# define KAAPI_PROCINFO_H_INCLUDED


#if 0 /* unit */
typedef unsigned int kaapi_processor_id_t;
#define KAAPI_MAX_PROCESSOR 32
#define KAAPI_PROC_TYPE_CPU 0
#define KAAPI_PROC_TYPE_GPU 1
#else
#include "kaapi_impl.h"
#endif /* unit */


typedef struct kaapi_procinfo
{
  struct kaapi_procinfo* next;

  unsigned int bound_cpu;
  unsigned int proc_type;
  unsigned int proc_index;

  /* here to avoid thread arg allocation */
  kaapi_processor_id_t kid;

} kaapi_procinfo_t;


typedef struct kaapi_procinfo_list
{
  unsigned int count;
  kaapi_procinfo_t* head;
  kaapi_procinfo_t* tail;
} kaapi_procinfo_list_t;


kaapi_procinfo_t* kaapi_procinfo_alloc(void);
void kaapi_procinfo_free(kaapi_procinfo_t*);
void kaapi_procinfo_list_init(kaapi_procinfo_list_t*);
void kaapi_procinfo_list_free(kaapi_procinfo_list_t*);
void kaapi_procinfo_list_add(kaapi_procinfo_list_t*, kaapi_procinfo_t*);
int kaapi_procinfo_list_parse_string
(kaapi_procinfo_list_t*, const char*, unsigned int, unsigned int);



#endif /* ! KAAPI_PROCINFO_H_INCLUDED */
