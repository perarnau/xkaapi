
#ifndef _KAAPI_AFFINITY_H_
#define _KAAPI_AFFINITY_H_

#include "kaapi_impl.h"

struct kaapi_taskdescr_t;

/*
 * Default affinity function, returns local kproc.
 */
kaapi_processor_t *kaapi_affinity_default(kaapi_processor_t * kproc,
					      struct kaapi_taskdescr_t * td);

/*
 * Return random kprocessor.
 */
kaapi_processor_t *kaapi_affinity_rand(kaapi_processor_t * kproc,
					      struct kaapi_taskdescr_t * td);

/*
 * Consider valid data to pick a processor.
 */
kaapi_processor_t *kaapi_affinity_datawizard(kaapi_processor_t * kproc,
					      struct kaapi_taskdescr_t * td);

kaapi_processor_t *kaapi_affinity_wrmode(kaapi_processor_t * kproc,
					      struct kaapi_taskdescr_t * td);

int kaapi_affinity_exec_readylist( 
	kaapi_processor_t* kproc
    );

#endif /* _KAAPI_AFFINITY_H_ */
