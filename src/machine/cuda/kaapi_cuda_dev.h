
#ifndef KAAPI_CUDA_DEV_H_INCLUDED
#define KAAPI_CUDA_DEV_H_INCLUDED

#include "kaapi_cuda_proc.h"

int kaapi_cuda_dev_open(kaapi_cuda_proc_t *, unsigned int);

void kaapi_cuda_dev_close(kaapi_cuda_proc_t *);

kaapi_processor_t *kaapi_cuda_mem_get_proc(void);

int kaapi_cuda_dev_enable_peer_access(kaapi_cuda_proc_t * const);

static inline int kaapi_cuda_dev_has_peer_access(unsigned int peer)
{
  kaapi_assert_debug((peer >= 0) && (peer < KAAPI_CUDA_MAX_DEV));
  return kaapi_get_current_processor()->cuda_proc.peers[peer];
}

#endif				/* ! KAAPI_CUDA_DEV_H_INCLUDED */
