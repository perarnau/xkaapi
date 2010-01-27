#include "kaapi_impl.h"

void kaapi_set_workload( kaapi_uint32_t workload ) 
{
  kaapi_processor_t* const self_kproc =
    _kaapi_get_current_processor();

  KAAPI_ATOMIC_WRITE(&self_kproc->workload, workload);
}
