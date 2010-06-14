#include "kaapi_impl.h"


void kaapi_set_workload(kaapi_processor_t* kproc, kaapi_uint32_t workload) 
{
  KAAPI_ATOMIC_WRITE(&kproc->workload, workload);
}

void kaapi_set_self_workload(kaapi_uint32_t workload) 
{
  KAAPI_ATOMIC_WRITE(&_kaapi_get_current_processor()->workload, workload);
}

kaapi_processor_t* kaapi_stealcontext_kproc(kaapi_stealcontext_t* sc)
{
  return sc->ctxtthread->proc;
}

kaapi_processor_t* kaapi_request_kproc(kaapi_request_t* kr)
{
  return kr->proc;
}

unsigned int kaapi_request_kid(kaapi_request_t* kr)
{
  return kr->proc->kid;
}
