#include "kaapi_impl.h"

kaapi_uint32_t kaapi_perf_reduce(void)
{
  kaapi_uint64_t cnt_stealreqok = 0;
  kaapi_uint64_t cnt_stealreq = 0;
  int i;

  for (i=0; i<kaapi_count_kprocessors; ++i)
  {
    cnt_stealreqok  += kaapi_all_kprocessors[i]->cnt_stealreqok;
    cnt_stealreq    += kaapi_all_kprocessors[i]->cnt_stealreq;
  }

  return (kaapi_uint32_t)(cnt_stealreq - cnt_stealreqok);
}
