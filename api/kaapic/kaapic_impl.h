/*
 ** xkaapi
 ** 
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
 ** 
 ** This software is a computer program whose purpose is to execute
 ** multithreaded computation with data flow synchronization between
 ** threads.
 ** 
 ** This software is governed by the CeCILL-C license under French law
 ** and abiding by the rules of distribution of free software.  You can
 ** use, modify and/ or redistribute the software under the terms of
 ** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
 ** following URL "http://www.cecill.info".
 ** 
 ** As a counterpart to the access to the source code and rights to
 ** copy, modify and redistribute granted by the license, users are
 ** provided only with a limited warranty and the software's author,
 ** the holder of the economic rights, and the successive licensors
 ** have only limited liability.
 ** 
 ** In this respect, the user's attention is drawn to the risks
 ** associated with loading, using, modifying and/or developing or
 ** reproducing the software by the user in light of its specific
 ** status of free software, that may mean that it is complicated to
 ** manipulate, and that also therefore means that it is reserved for
 ** developers and experienced professionals having in-depth computer
 ** knowledge. Users are therefore encouraged to load and test the
 ** software's suitability as regards their requirements in conditions
 ** enabling the security of their systems and/or data to be ensured
 ** and, more generally, to use and operate it in the same conditions
 ** as regards security.
 ** 
 ** The fact that you are presently reading this means that you have
 ** had knowledge of the CeCILL-C license and that you accept its
 ** terms.
 ** 
 */
#ifndef KAAPIC_HIMPL_INCLUDED
# define KAAPIC_HIMPL_INCLUDED

/* kaapic_save,restore_frame */
#include "kaapi_impl.h"
#include "kaapic.h"

/* implementation for kaapic API */
#if defined(__cplusplus)
extern "C" {
#endif

/* some important macros */
#define KAAPIC_USE_KPROC_LOCK 1

/* enable worker to steal slice reserved to other threads
   - may degrade locality ?
*/
#define KAAPIC_ALLOWS_WORKER_STEAL_SLICE 1

/* add extension to pass memory mapping of array to the scheduler
   - at the begining of a foreach + attributes the runtime computes a transformation
   to allows local iteration per thread.
   - this option may be mixed with KAAPIC_ALLOWS_WORKER_STEAL_SLICE. A better study
   of the impact of to guarantee locality versus to balance the workload must be do!
*/
//#define KAAPI_USE_FOREACH_WITH_DATADISTRIBUTION 1 


extern void _kaapic_register_task_format(void);

typedef void (*kaapic_body_arg_f_c_t)(int32_t, int32_t, int32_t, ...);
typedef void (*kaapic_body_arg_f_c_ull_t)(unsigned long long, unsigned long long, int32_t, ...);
typedef void (*kaapic_body_arg_f_f_t)(int32_t*, int32_t*, int32_t*, ...);


/* closure for the body of the for each */
typedef struct kaapic_body_arg_t {
  union {
    kaapic_body_arg_f_c_t f_c;
    kaapic_body_arg_f_c_ull_t f_c_ull;
    kaapic_body_arg_f_f_t f_f;
  } u;
  unsigned int        nargs;
  void*               args[];
} kaapic_body_arg_t;


/* Signature of foreach body 
   Called with (first, last, tid, arg) in order to do
   computation over the range [first,last[.
*/
typedef void (*kaapic_foreach_body_t)(int32_t, int32_t, int32_t, kaapic_body_arg_t* );

/* Signature of foreach body 
   Called with (first, last, tid, arg) in order to do
   computation over the range [first,last[.
*/
typedef void (*kaapic_foreach_body_ull_t)(
  unsigned long long, 
  unsigned long long, 
  int32_t, 
  kaapic_body_arg_t* 
);


/* Default attribut if not specified
*/
extern kaapic_foreach_attr_t kaapic_default_attr;


/* Return true iff attribut define a distribution 
*/
static inline int kaapic_foreach_attr_hasdatadistribution( const kaapic_foreach_attr_t*  attr )
{ return attr->datadist.type != KAAPIC_DATADIST_VOID; }


/* exported foreach interface 
   evaluate body_f(first, last, body_args) in parallel, assuming
   that the evaluation of body_f(i, j, body_args) does not impose
   dependency with evaluation of body_f(k,l, body_args) if [i,j[ and [k,l[
   does not intersect.
*/
extern int kaapic_foreach_common
(
  kaapi_workqueue_index_t first,
  kaapi_workqueue_index_t last,
  const kaapic_foreach_attr_t*  attr,
  kaapic_foreach_body_t   body_f,
  kaapic_body_arg_t*      body_args
);


/* exported foreach interface 
   evaluate body_f(first, last, body_args) in parallel, assuming
   that the evaluation of body_f(i, j, body_args) does not impose
   dependency with evaluation of body_f(k,l, body_args) if [i,j[ and [k,l[
   does not intersect.
*/
extern int kaapic_foreach_common_ull
(
  kaapi_workqueue_index_ull_t  first,
  kaapi_workqueue_index_ull_t  last,
  const kaapic_foreach_attr_t* attr,
  kaapic_foreach_body_ull_t    body_f,
  kaapic_body_arg_t*           body_args
);


/* wrapper for kaapic_foreach(...) and kaapic_foreach_withformat(...)
*/
extern void kaapic_foreach_body2user(
  int32_t first, 
  int32_t last, 
  int32_t tid, 
  kaapic_body_arg_t* arg 
);


extern void kaapic_foreach_body2user_ull(
  unsigned long long first, 
  unsigned long long last, 
  int32_t tid, 
  kaapic_body_arg_t* arg 
);

/* work array distribution. allow for random access. 
   The thread tid has a reserved slice [first, last)
   iff map[tid] != 0.
   Then it should process work on its slice, where:
     first = startindex[tid2pos[tid]]   (inclusive)
     last  = startindex[tid2pos[tid]+1] (exclusive)
  
   tid2pos is a permutation to specify slice of each tid
*/
typedef struct work_array
{
  kaapi_bitmap_t       map;
  uint8_t              tid2pos[KAAPI_MAX_PROCESSOR];
  union {
    kaapi_workqueue_index_t     startindex[1+KAAPI_MAX_PROCESSOR];
    kaapi_workqueue_index_ull_t startindex_ull[1+KAAPI_MAX_PROCESSOR];
  };
} kaapic_work_distribution_t;


/* work container information */
typedef struct work_info
{
  /* grains */
  union {
    struct {
      long par_grain;
      long seq_grain;
    } li;
    struct {
      unsigned long long par_grain;
      unsigned long long seq_grain;
    } ull;
  } rep;
  unsigned int         nthreads;  /* number of threads for initial splitting */
  kaapi_cpuset_t       threadset; /* thread set used for initial distribution i = kid */
  size_t               itercount; /* number of iterations */
#if defined(KAAPI_USE_FOREACH_WITH_DATADISTRIBUTION)
  _kaapic_foreach_attr_datadist_t dist;
#endif
} kaapic_work_info_t;


#if defined(KAAPI_USE_FOREACH_WITH_DATADISTRIBUTION)
/* store current range from previous poped ranged from the workqueue
   Used to inorder to transform range indexes between internal iteration and
   user level iteration in order to ensure local access of data.
*/
typedef union kaapic_local_workqueue_t {
  struct {
    kaapi_workqueue_index_t end;
    kaapi_workqueue_index_t beg;
  } li;
  struct {
    kaapi_workqueue_index_ull_t end;
    kaapi_workqueue_index_ull_t beg;
  } ull;
} kaapic_local_workqueue_t;
#endif

/* local work: used by each worker to process its work */
typedef struct local_work
{
  kaapi_workqueue_t       cr;
#if defined(KAAPIC_USE_KPROC_LOCK)
#else
  kaapi_lock_t            lock;
#endif
#if defined(KAAPI_USE_FOREACH_WITH_DATADISTRIBUTION)
  kaapic_local_workqueue_t local_cr;
#endif
  int volatile            init;      /* !=0 iff init */

  void*                   context __attribute__((aligned(KAAPI_CACHE_LINE)));
  struct global_work*     global;    /* go up to all local information */
  kaapi_workqueue_index_t workdone;  /* to compute completion */
  int                     tid;       /* identifier : ==kid */
} kaapic_local_work_t __attribute__ ((aligned (KAAPI_CACHE_LINE)));


/* global work common 
   Initialized by one thread (the master thread).
   Contains all local works and global information to compute completion.
   - wa: the work distribution structure 
   - lwork[tid]; the work for the thread tid
*/
typedef struct global_work
{
  kaapi_atomic64_t            workremain __attribute__((aligned(KAAPI_CACHE_LINE)));
  kaapi_atomic64_t            workerdone;
  
  /* global distribution */
  kaapic_work_distribution_t  wa  __attribute__((aligned(KAAPI_CACHE_LINE)));
  kaapic_local_work_t         lwork[KAAPI_MAX_PROCESSOR];

  /* work routine */
  union {
    kaapic_foreach_body_t     body_f;
    kaapic_foreach_body_ull_t body_f_ull;
  };
  kaapic_body_arg_t*          body_args;

  /* infos container */
  kaapic_work_info_t          wi;
} kaapic_global_work_t;



/* Lower level function used by libgomp implementation */

/* init work 
   \retval returns non zero if there is work to do, else returns 0
*/
extern kaapic_local_work_t*  kaapic_foreach_workinit(
  kaapi_thread_context_t*      self_thread,
  kaapi_workqueue_index_t      first, 
  kaapi_workqueue_index_t      last,
  const kaapic_foreach_attr_t* attr,
  kaapic_foreach_body_t        body_f,
  kaapic_body_arg_t*           body_args
);


/* init work 
   \retval returns non zero if there is work to do, else returns 0
*/
extern kaapic_local_work_t*  kaapic_foreach_workinit_ull(
  kaapi_thread_context_t*      self_thread,
  kaapi_workqueue_index_ull_t  first, 
  kaapi_workqueue_index_ull_t  last,
  const kaapic_foreach_attr_t* attr,
  kaapic_foreach_body_ull_t    body_f,
  kaapic_body_arg_t*           body_args
);


/*
*/
extern kaapic_global_work_t* kaapic_foreach_global_workinit
(
  kaapi_thread_context_t*      self_thread,
  kaapi_workqueue_index_t      first, 
  kaapi_workqueue_index_t      last,
  const kaapic_foreach_attr_t* attr,
  kaapic_foreach_body_t        body_f,
  kaapic_body_arg_t*           body_args
);

/*
*/
extern kaapic_global_work_t* kaapic_foreach_global_workinit_ull
(
  kaapi_thread_context_t*      self_thread,
  kaapi_workqueue_index_ull_t  first, 
  kaapi_workqueue_index_ull_t  last,
  const kaapic_foreach_attr_t* attr,
  kaapic_foreach_body_ull_t    body_f,
  kaapic_body_arg_t*           body_args
);


/* init local work if know global work.
   May be called by each runing threads that decide to cooperate together
   to execute in common a global work.
   \retval returns non zero if there is work to do, else returns 0
*/
extern kaapic_local_work_t* kaapic_foreach_local_workinit(
  kaapic_local_work_t*    lwork,
  kaapi_workqueue_index_t first,
  kaapi_workqueue_index_t last
);

/* init local work if know global work.
   May be called by each runing threads that decide to cooperate together
   to execute in common a global work.
   \retval returns non zero if there is work to do, else returns 0
*/
extern kaapic_local_work_t* kaapic_foreach_local_workinit_ull(
  kaapic_local_work_t*        lwork,
  kaapi_workqueue_index_ull_t first,
  kaapi_workqueue_index_ull_t last
);


extern int kaapic_global_work_pop
(
  kaapic_global_work_t* gw,
  kaapi_processor_id_t tid, 
  kaapi_workqueue_index_t* i, 
  kaapi_workqueue_index_t* j
);

extern int kaapic_global_work_pop_ull
(
  kaapic_global_work_t*        gw,
  kaapi_processor_id_t         tid, 
  kaapi_workqueue_index_ull_t* i, 
  kaapi_workqueue_index_ull_t* j
);

/* 
  Return !=0 iff first and last have been filled for the next piece
  of work to execute
*/
extern int kaapic_foreach_worknext(
  kaapic_local_work_t*    work,
  kaapi_workqueue_index_t* first,
  kaapi_workqueue_index_t* last
);


/* 
  Return !=0 iff first and last have been filled for the next piece
  of work to execute
*/
extern int kaapic_foreach_worknext_ull(
  kaapic_local_work_t*         work,
  kaapi_workqueue_index_ull_t* first,
  kaapi_workqueue_index_ull_t* last
);


/* To be called by the caller of kaapic_foreach_local_workinit
   that returns success
*/
extern int kaapic_foreach_local_workend(
  kaapi_thread_context_t* self_thread,
  kaapic_local_work_t*    lwork
);


/*
*/
int kaapic_foreach_workend
(
  kaapi_thread_context_t* self_thread,
  kaapic_local_work_t*    work
);



#if defined(KAAPI_USE_FOREACH_WITH_DATADISTRIBUTION)
/*
*/
static inline int kaapic_local_workqueue_isempty( const kaapic_local_workqueue_t* lcr )
{
  return (lcr->li.end <= lcr->li.beg);
}
static inline int kaapic_local_workqueue_isempty_ull( const kaapic_local_workqueue_t* lcr )
{
  return (lcr->ull.end <= lcr->ull.beg);
}


static inline int kaapic_local_workqueue_set( 
  kaapic_local_workqueue_t* lcr,
  kaapi_workqueue_index_t begin, 
  kaapi_workqueue_index_t end
)
{
  lcr->li.end = end;
  lcr->li.beg = begin;
  return 0;
}
static inline int kaapic_local_workqueue_set_ull( 
  kaapic_local_workqueue_t* lcr,
  kaapi_workqueue_index_ull_t begin, 
  kaapi_workqueue_index_ull_t end
)
{
  lcr->ull.end = end;
  lcr->ull.beg = begin;
  return 0;
}


static inline kaapi_workqueue_index_t* kaapic_local_workqueue_begin_ptr( kaapic_local_workqueue_t* lcr )
{
  return &lcr->li.beg;
}
static inline kaapi_workqueue_index_t* kaapic_local_workqueue_end_ptr( kaapic_local_workqueue_t* lcr )
{
  return &lcr->li.end;
}

static inline kaapi_workqueue_index_ull_t* kaapic_local_workqueue_begin_ptr_ull( kaapic_local_workqueue_t* lcr )
{
  return &lcr->ull.beg;
}
static inline kaapi_workqueue_index_ull_t* kaapic_local_workqueue_end_ptr_ull( kaapic_local_workqueue_t* lcr )
{
  return &lcr->ull.end;
}


/*
 * @param       int             indice in kernel space
 * @param       int             B bloc size
 * @param       int             P parallel thread number
 * @param       int             N iteration end number
 *
 */
static inline long _kaapi_kernel2user (long indice, unsigned long B, unsigned long L, unsigned long N)
{
                int id_thread = 0;
                int offset = 0;
                int indice_local = 0;

                id_thread = (indice / (N / L)) * B;
//              printf("id_cycle: %d = (%d / (%d / %d)) * %d)\t", id_thread, indice, N, L, B);

                indice_local = (indice - ((indice / (N / L)) * (N / L))) / B * L * B;
//              printf("indice_local: %d\t", indice_local);

                offset = indice % B;
//              printf("offset: %d\n", offset);

                return (id_thread + indice_local + offset);
}


/*  Benjamin, a implementer:
    - doit retourner dans beg,end le range utilisateur qui est le plus grand range possible de valeurs contigues
    contenu dans la local_workqueue (ie. kwq->li.beg..kwq->li.end)
    - la taille retournée doit être <= sgrain
    - la localworkqueue doit etre mise a jour pour le prochain pop
    Return 0 in case of success 
    Return EBUSY is the queue is empty
    Return EINVAL if invalid arguments

    Usefull data:
    wi->dist : is the _kaapic_foreach_attr_datadist_t attribut about the data distribution
    wi->itercount : the initial iteration count given by the foreach
    wi->nthreads : the number of threads that is used for iteration
    [not really tested: wi->threadset: the indexes of threads used for iteration]
*/
static inline int kaapic_local_workqueue_pop_withdatadistribution(
          kaapic_local_workqueue_t*              kwq, 
          const kaapic_work_info_t*              wi,
          kaapi_workqueue_index_t*               beg,
          kaapi_workqueue_index_t*               end,
          kaapi_workqueue_index_t                sgrain
)
{
  const _kaapic_foreach_attr_datadist_t* attr = &wi->dist;
  switch (attr->type) 
  {
  case KAAPIC_DATADIST_VOID:
    *beg = kwq->li.beg; kwq->li.beg = 0;
    *end = kwq->li.end; kwq->li.end = 0;
    break;
  case KAAPIC_DATADIST_BLOCCYCLIC:
//  printf("li.beg: %dn li.end: %d\t", kwq->li.beg, kwq->li.end);

        if (kwq->li.beg < kwq->li.end) {

      int limit = ((kwq->li.beg / attr->dist.bloccyclic.size) * attr->dist.bloccyclic.size) + attr->dist.bloccyclic.size;
      if (limit > kwq->li.end)
        limit = kwq->li.end;
//      printf("limit: %d\n", limit);

      *beg = kwq->li.beg;
      if ((kwq->li.beg + sgrain) < limit)
        *end = *beg + sgrain;
      else 
        *end = limit;
      kwq->li.beg = *end;

#if 0
      int a = *beg, 
                            b = *end;
#endif

      *beg = _kaapi_kernel2user(*beg, attr->dist.bloccyclic.size, attr->dist.bloccyclic.length, wi->itercount);
      *end = _kaapi_kernel2user(*end - 1, attr->dist.bloccyclic.size, attr->dist.bloccyclic.length, wi->itercount) + 1;

#if 0
      printf("[%8d, %8d) --- [%8d, %8d)\n", a, b, *beg, *end);
      printf("size: %d %d\n", b -a, *end - *beg);
#endif
        }
        else
      return EBUSY;

    break;
  default:
    break;
  }
  if (*end <= *beg) return EBUSY;
  return 0;
}


/*  Benjamin, a implementer: idem kaapic_local_workqueue_pop_withdatadistribution mais avec des indices de type
    unsigned long long [attention au calcul...]
    - doit retourner dans beg,end le range utilisateur qui est le plus grand range possible de valeurs contigues
    contenu dans la local_workqueue (ie. kwq->li.beg..kwq->li.end)
    - la taille retournée doit être <= sgrain
    - la localworkqueue doit etre mise a jour pour le prochain pop
    Return 0 in case of success 
    Return EBUSY is the queue is empty
    Return EINVAL if invalid arguments
    
    Usefull data:
    wi->dist : is the _kaapic_foreach_attr_datadist_t attribut about the data distribution
    wi->itercount : the initial iteration count given by the foreach
    wi->nthreads : the number of threads that is used for iteration
    [not really tested: wi->threadset: the indexes of threads used for iteration]
*/
static inline int kaapic_local_workqueue_pop_withdatadistribution_ull(
  kaapic_local_workqueue_t*              kwq, 
  const kaapic_work_info_t*              wi,
  kaapi_workqueue_index_ull_t*           beg,
  kaapi_workqueue_index_ull_t*           end,
  kaapi_workqueue_index_ull_t            sgrain
)
{
  const _kaapic_foreach_attr_datadist_t* attr = &wi->dist;
  switch (attr->type) 
  {
  case KAAPIC_DATADIST_VOID:
    *beg = kwq->ull.beg; kwq->ull.beg = 0;
    *end = kwq->ull.end; kwq->ull.end = 0;
    break;
  case KAAPIC_DATADIST_BLOCCYCLIC:
    kaapi_assert_m(0,"Benjamin: todo");
    break;
  default:
    break;
  }
  if (*end <= *beg) return EBUSY;
  return 0;
}
#endif


/* misc: for API F */
extern void kaapic_save_frame(void);
extern void kaapic_restore_frame(void);

#if defined(__cplusplus)
}
#endif

#endif /* KAAPIC_HIMPL_INCLUDED */
