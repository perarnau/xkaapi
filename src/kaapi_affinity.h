/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** joao.limag@imag.fr
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
#ifndef _KAAPI_AFFINITY_H_
#define _KAAPI_AFFINITY_H_

#include "kaapi_impl.h"

struct kaapi_taskdescr_t;

/*
 * Default affinity function, returns local kproc.
 */
extern kaapi_processor_t *kaapi_push_by_affinity_default(
    kaapi_processor_t * kproc,
    struct kaapi_taskdescr_t * td
);

/*
 * Return random kprocessor.
 */
extern kaapi_processor_t *kaapi_push_by_affinity_rand(
    kaapi_processor_t * kproc,
    struct kaapi_taskdescr_t * td
);

/*
 * Consider valid data to pick a processor.
 */
extern kaapi_processor_t *kaapi_push_by_affinity_locality(
    kaapi_processor_t * kproc,
    struct kaapi_taskdescr_t * td
);

extern kaapi_processor_t *kaapi_push_by_affinity_writer(kaapi_processor_t * kproc,
					      struct kaapi_taskdescr_t * td);

extern int kaapi_affinity_exec_readylist( kaapi_processor_t* kproc );

/* return the sum of data valid in the kproc asid 
*/
extern uint64_t kaapi_data_get_affinity_hit_size(
     const kaapi_processor_t * kproc,
     struct kaapi_taskdescr_t * td
);

/* return true if td has an WR parameter valid in kproc
*/
extern int kaapi_data_get_affinity_is_valid_writer(
     const kaapi_processor_t * kproc,
     struct kaapi_taskdescr_t * td
);

/*
*/
extern struct kaapi_taskdescr_t* kaapi_steal_by_affinity_first( const kaapi_processor_t* thief, struct kaapi_taskdescr_t* tail );


/*
*/
extern struct kaapi_taskdescr_t* kaapi_steal_by_affinity_maxctpath( const kaapi_processor_t* thief, struct kaapi_taskdescr_t* tail );


/*
*/
extern struct kaapi_taskdescr_t* kaapi_steal_by_affinity_maxhit( const kaapi_processor_t* thief, struct kaapi_taskdescr_t* tail );

extern struct kaapi_taskdescr_t* kaapi_steal_by_affinity_writer( const kaapi_processor_t* thief, struct kaapi_taskdescr_t* tail );

#endif /* _KAAPI_AFFINITY_H_ */
