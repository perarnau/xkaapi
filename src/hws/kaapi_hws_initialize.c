#include <stdio.h>

#include "kaapi_impl.h"
#include "kaapi_procinfo.h"


#if 0
typedef struct kaapi_ws_block
{
} kaapi_ws_block_t;

typedef sruct kaapi_hws_level
{
  /* describes the kids of a given memory level */

  kaapi_bitmap_t members;  

} kaapi_hws_level_t;
#endif


#if 0

/* what is it used for:
 */

static void get_self_siblings
(kaapi_processor_t* self, kaapi_bitmap_t* siblings)
{
  /* get all the sibling nodes of a given */
}

#endif

static void print_levels(kaapi_processor_t* kproc)
{
  int depth;

  printf("-- KID %u\n", kproc->kid);

  for (depth = 0; depth < kproc->hlevel.depth; ++depth)
  {
    const unsigned int nkids = kproc->hlevel.levels[depth].nkids;
    const kaapi_processor_id_t* kids = kproc->hlevel.levels[depth].kids;
    unsigned int i;
    printf("level[%u]: ", depth);
    for (i = 0; i < nkids; ++i) printf(" %u", kids[i]);
    printf("\n");
  }

  printf("\n");
}


int kaapi_hws_init_perproc(kaapi_processor_t* kproc)
{
  /* initialize the per memory level stealing structures */
  /* assume kaapi_processor_computetopo called */

  print_levels(kproc);

  return 0;
}


int kaapi_hws_fini_perproc(kaapi_processor_t* kproc)
{
  /* assume kaapi_hws_initialize called */
  return 0;
}


int kaapi_hws_init_global(void)
{
  return 0;
}


int kaapi_hws_fini_global(void)
{
  return 0;
}

