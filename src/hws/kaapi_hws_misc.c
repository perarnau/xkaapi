#include "kaapi_impl.h"
#include "kaapi_hws.h"


unsigned int kaapi_hws_get_request_nodeid(const kaapi_request_t* req)
{
  return (unsigned int)req->kid;
} 


unsigned int kaapi_hws_get_node_count(kaapi_hws_levelid_t levelid)
{
  kaapi_assert_debug(kaapi_hws_is_levelid_set(levelid));
  return hws_levels[levelid].block_count;
}
