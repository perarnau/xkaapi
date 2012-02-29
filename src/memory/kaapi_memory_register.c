
#include <stdio.h>

#include "kaapi_impl.h"
#include "kaapi_mem.h"
#include "kaapi_mem_host_map.h"
#include "kaapi_mem_data.h"

#if defined(KAAPI_USE_CUDA)
#include "machine/cuda/kaapi_cuda_mem.h"
#include "machine/cuda/kaapi_cuda_ctx.h"
#endif

struct kaapi_mem_reg_data_item_t;

typedef struct kaapi_mem_reg_data_item_t {
	struct kaapi_mem_reg_data_item_t* next;
	struct kaapi_mem_reg_data_item_t* prev;
	
	kaapi_mem_data_t	kmd;
	kaapi_memory_view_t	view;
} kaapi_mem_reg_data_item_t;

static kaapi_mem_reg_data_item_t*
kaapi_mem_reg_data_add( kaapi_mem_reg_data_t *mem_data, 
	const kaapi_mem_asid_t host_asid,
	void *ptr, kaapi_memory_view_t* view )
{
    kaapi_mem_reg_data_item_t *item= (kaapi_mem_reg_data_item_t*)malloc(
		    sizeof(kaapi_mem_reg_data_item_t) );
    if( item == NULL )
	    return NULL;


    kaapi_mem_data_init( &item->kmd );
    kaapi_mem_data_set_addr( &item->kmd, host_asid, (kaapi_mem_addr_t)ptr );
    item->view = *view;
    item->prev = item->next = NULL;
    if( mem_data->beg == NULL ) {
	    mem_data->beg = item;
    } else {
	    item->prev= mem_data->end;
	    mem_data->end->next = item;
    }
    mem_data->end = item;

    return item;
}

static void
kaapi_mem_reg_data_remove( kaapi_mem_reg_data_t *mem_data, 
	kaapi_mem_reg_data_item_t* item
	)
{
    kaapi_mem_reg_data_item_t *next, *prev;

    next = item->next;
    prev = item->prev;

    if( NULL != next )
	next->prev = prev;
    else
	mem_data->end = prev;
    if( NULL != prev )
	prev->next = next;
    else
	mem_data->beg = next;
    /* TODO */
}

#if defined(KAAPI_USE_CUDA)
static int
kaapi_mem_register_data_cuda( kaapi_processor_t* cuda_proc,
	kaapi_mem_reg_data_item_t*  item,
	kaapi_mem_host_map_t* host_map,
	kaapi_mem_asid_t host_asid  )
{
#if KAAPI_DEBUG
    uint64_t t0 = kaapi_get_elapsedns();
#endif
    const kaapi_mem_host_map_t* cuda_map = 
	kaapi_processor_get_mem_host_map(cuda_proc);
    const kaapi_mem_asid_t cuda_asid = kaapi_mem_host_map_get_asid(cuda_map);
    kaapi_mem_addr_t addr;
    kaapi_cuda_ctx_set( cuda_proc->cuda_proc.index );
    kaapi_cuda_mem_alloc_( &addr, kaapi_memory_view_size(&item->view) );
    kaapi_cuda_mem_register_( (void*)kaapi_mem_data_get_addr( &item->kmd,
		    host_asid), kaapi_memory_view_size(&item->view) );
    /* TODO: check this again */
    kaapi_cuda_mem_copy_htod(
	    kaapi_make_pointer(0, (void*)addr), &item->view,
	    kaapi_make_pointer(0, (void*)kaapi_mem_data_get_addr( &item->kmd,
		    host_asid)), &item->view );
    kaapi_mem_data_set_addr( &item->kmd, cuda_asid, addr );
    kaapi_mem_data_set_all_dirty_except( &item->kmd, host_asid );
    fflush(stdout);
#if KAAPI_DEBUG
    uint64_t t1 = kaapi_get_elapsedns();
    fprintf( stdout, "%lu:%x:%s:%s:%d\n", kaapi_get_current_kid(),
	    kaapi_get_current_processor()->proc_type,
	    __FUNCTION__,
	    "",
	    t1-t0);
#endif
    return 0;
}

static int
kaapi_mem_alloc_data( kaapi_mem_reg_data_item_t* item,
   kaapi_mem_host_map_t* host_map, kaapi_mem_asid_t host_asid )
{
    kaapi_processor_t** pos = kaapi_all_kprocessors;
    size_t i;

    for (i = 0; i < kaapi_count_kprocessors; ++i, ++pos)
	if ((*pos)->proc_type == KAAPI_PROC_TYPE_CUDA)
	    kaapi_mem_register_data_cuda( *pos, item, host_map, host_asid );
    return 0;
}

static int
kaapi_mem_free_data_cuda( kaapi_processor_t* cuda_proc,
	kaapi_mem_reg_data_item_t*  item )
{
    const kaapi_mem_host_map_t* cuda_map = 
	kaapi_processor_get_mem_host_map(cuda_proc);
    const kaapi_mem_asid_t cuda_asid = kaapi_mem_host_map_get_asid(cuda_map);
    if( kaapi_mem_data_has_addr( &item->kmd, cuda_asid ) ) {
	kaapi_pointer_t ptr = kaapi_make_pointer(0, (void*)kaapi_mem_data_get_addr(
			&item->kmd, cuda_asid));
	kaapi_cuda_ctx_set( cuda_proc->cuda_proc.index );
	kaapi_cuda_mem_free( &ptr );
	kaapi_mem_data_clear_addr( &item->kmd, cuda_asid );
    }
    return 0;
}

static int
kaapi_mem_free_data( kaapi_mem_reg_data_item_t*  item )
{
    kaapi_processor_t** pos = kaapi_all_kprocessors;
    size_t i;

    for (i = 0; i < kaapi_count_kprocessors; ++i, ++pos)
	if ((*pos)->proc_type == KAAPI_PROC_TYPE_CUDA)
	    kaapi_mem_free_data_cuda( *pos, item );
    return 0;
}
#endif

int kaapi_memory_register( void* ptr, kaapi_memory_view_t view )
{
    kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    kaapi_mem_reg_data_item_t* item;

    item = kaapi_mem_reg_data_add( &host_map->data, host_asid, ptr, &view );

#if defined(KAAPI_USE_CUDA)
    kaapi_mem_alloc_data( item, host_map, host_asid );
#endif
    return 0;
}

static inline kaapi_mem_reg_data_item_t*
kaapi_mem_reg_data_begin( const kaapi_mem_host_map_t* map )
{
    return map->data.beg;
}

static inline kaapi_mem_reg_data_item_t*
kaapi_mem_reg_data_next( const kaapi_mem_reg_data_item_t* item )
{
    return item->next;
}

void kaapi_memory_unregister( void* ptr )
{
    kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    kaapi_mem_reg_data_item_t *item;

    for( item = kaapi_mem_reg_data_begin(host_map);
	    item != NULL; item = kaapi_mem_reg_data_next(item) ){
	if( kaapi_mem_data_get_addr( &item->kmd, host_asid ) ==
		(kaapi_mem_addr_t)ptr ) {
	    kaapi_mem_reg_data_remove( &host_map->data, item );
#if defined(KAAPI_USE_CUDA)
	    kaapi_cuda_mem_unregister_(
		(void*)kaapi_mem_data_get_addr( &item->kmd, host_asid ) );
	    kaapi_mem_free_data( item );
#endif
	    free( item );
	    break;
	}
    }
}

kaapi_mem_data_t*
kaapi_memory_register_find( kaapi_mem_addr_t addr )
{
    kaapi_mem_host_map_t* host_map = 
	kaapi_processor_get_mem_host_map(kaapi_all_kprocessors[0]);
    const kaapi_mem_asid_t host_asid = kaapi_mem_host_map_get_asid(host_map);
    kaapi_mem_reg_data_item_t *item;
    kaapi_mem_addr_t item_addr;

    for( item = kaapi_mem_reg_data_begin(host_map);
	    item != NULL; item = kaapi_mem_reg_data_next(item) ){
	item_addr = kaapi_mem_data_get_addr( &item->kmd, host_asid );
#if 0
	fprintf(stdout, "%s: search addr=%p min=%p max=%p size=%lu\n",
		__FUNCTION__,
		(void*)addr,
		(void*)item_addr,
		(void*)(item_addr+kaapi_memory_view_size(&item->view)),
	      kaapi_memory_view_size(&item->view) );
	fflush(stdout);
#endif
	if(	( addr >= item_addr ) &&
		( addr < ( item_addr+kaapi_memory_view_size(&item->view) )  )
	     ) {
#if 0
	    fprintf(stdout, "%s: found addr=%p\n",
		__FUNCTION__,
		    (void*)addr);
	    fflush(stdout);
#endif
	    return &item->kmd;
	}
    }

    return NULL;
}

kaapi_mem_addr_t 
kaapi_memory_register_convert( kaapi_mem_asid_t dev_asid,
       kaapi_mem_asid_t host_asid, kaapi_mem_data_t* kmd )
{
    kaapi_mem_addr_t h_reg_addr = kaapi_mem_data_get_addr( 
	    kaapi_mem_data_get_parent(kmd), host_asid );
    kaapi_mem_addr_t d_reg_addr = kaapi_mem_data_get_addr(
	    kaapi_mem_data_get_parent(kmd), dev_asid );
    kaapi_data_t* h_data = (kaapi_data_t*)kaapi_mem_data_get_addr(
	    kmd, host_asid );
    kaapi_mem_addr_t h_addr =
	(kaapi_mem_addr_t)kaapi_pointer2void(h_data->ptr);
//    ptrdiff_t ptrdiff = h_addr - h_reg_addr;
    kaapi_mem_addr_t d_addr = d_reg_addr + (h_addr - h_reg_addr);
    return d_addr;
}
