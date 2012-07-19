
#ifndef _KAAPI_AFFINITY_H_
#define _KAAPI_AFFINITY_H_

#include "kaapi_impl.h"

struct kaapi_taskdescr_t;

kaapi_processor_t* kaapi_affinity_get_by_data(
       kaapi_processor_t*   kproc,
       struct kaapi_taskdescr_t*   td
       );

int kaapi_affinity_exec_readylist( 
	kaapi_processor_t* kproc
    );

#endif /* _KAAPI_AFFINITY_H_ */
