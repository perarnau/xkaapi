
#include "kaapi_impl.h"

#include "kaapi_affinity.h"


kaapi_processor_t* kaapi_affinity_get_by_data(
       kaapi_processor_t*   kproc,
       kaapi_taskdescr_t*   td
       )
{
    int kid_current = kproc->kid;
    int kid_remote = (kid_current+1)%kaapi_count_kprocessors;
    if( td->fmt == NULL )
	return kproc;
    else
        return kaapi_all_kprocessors[kid_remote];

    return NULL;
}

