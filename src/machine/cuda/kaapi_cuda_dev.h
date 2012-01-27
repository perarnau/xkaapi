
#ifndef KAAPI_CUDA_DEV_H_INCLUDED
#define KAAPI_CUDA_DEV_H_INCLUDED

#include "kaapi_cuda_proc.h"

int kaapi_cuda_dev_open( kaapi_cuda_proc_t* proc, unsigned int index );

void kaapi_cuda_dev_close( kaapi_cuda_proc_t* proc );

kaapi_processor_t*
kaapi_cuda_mem_get_proc( void );

kaapi_processor_t*
kaapi_cuda_get_proc_by_asid( kaapi_address_space_id_t asid );

#endif /* ! KAAPI_CUDA_DEV_H_INCLUDED */
