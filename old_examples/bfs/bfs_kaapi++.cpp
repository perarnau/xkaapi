/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
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

#include <list>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stddef.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#ifndef __USE_BSD
# define __USE_BSD
#endif
#include <time.h>
#include <sys/time.h>
#include "kaapi.h"


using std::list;
using std::vector;


#define CONFIG_ITER 100
#define CONFIG_PAR_GRAIN 16
#define CONFIG_SEQ_GRAIN 16


template<typename T>
struct list_with_size : public list<T>
{
  typedef list_with_size<T> self_type;
  typedef list<T> base_type;
  typedef typename base_type::iterator iterator;

  size_t _size;

  list_with_size() :
    base_type(), _size(0) {}

  void push_back(T& elem)
  {
    ++_size;
    ((base_type*)this)->push_back(elem);
  }

  void push_front(T& elem)
  {
    ++_size;
    ((base_type*)this)->push_front(elem);
  }

  void pop_front()
  {
    --_size;
    ((base_type*)this)->pop_front();
  }

  size_t size() const
  { return _size; }

#if 0 // unused
  void check_size() const
  {
    if (((base_type*)this)->size() == _size) return ;
    printf("INVALID_SIZE\n"); exit(-1);
  }
#endif

  void swap(self_type& l)
  {
    _size = l._size; l._size = 0;
    ((base_type*)this)->swap(l);    
  }
};

// we want list::size() in constant time
struct node;
typedef list_with_size<struct node*> nodeptr_list;

typedef struct node
{
#if CONFIG_PARALLEL
  volatile bool state __attribute__((aligned));
#else
  bool state;
#endif

  nodeptr_list adjlist;

#if CONFIG_NODEID
  unsigned int id;
#endif

  bool has_adj(const struct node* adj) const
  {
    nodeptr_list::const_iterator pos = adjlist.begin();
    nodeptr_list::const_iterator end = adjlist.end();

    for (; pos != end; ++pos)
      if (*pos == adj)
	return true;

    return false;
  }

  void add_adj(struct node* adj)
  { adjlist.push_back(adj); }

  void unmark()
  { state = false; }

#if CONFIG_PARALLEL
  bool mark_ifnot()
  {
    if (state == true) return false;
    return !__sync_fetch_and_or(&state, 1);
  }
#else
  bool mark_ifnot()
  { 
    if (state == true) return false;
    state = true; return true;
  }
#endif


} node_t;


typedef struct graph
{
  vector<node_t> nodes;

  void unmark_nodes()
  {
    vector<node_t>::iterator pos = nodes.begin();
    vector<node_t>::iterator end = nodes.end();
    for (; pos != end; ++pos)
      pos->unmark();
  }

  void initialize(unsigned int node_count)
  {
    nodes.resize(node_count);

    vector<node_t>::iterator pos = nodes.begin();
    vector<node_t>::iterator end = nodes.end();

#if CONFIG_NODEID
    unsigned int id = 0;
#endif

    for (; pos != end; ++pos)
    {
      pos->adjlist.clear();
      pos->state = false;
#if CONFIG_NODEID
      pos->id = id++;
#endif
    }
  }

  unsigned int node_count() const
  {
    return nodes.size();
  }

  const node_t* at_const(unsigned int i) const
  {
    return &nodes[i];
  }

  node_t* at(unsigned int i)
  {
    return &nodes[i];
  }

#if 0
  void print() const
  {
    vector<node_t>::const_iterator pos = nodes.begin();
    vector<node_t>::const_iterator end = nodes.end();
    for (; pos != end; ++pos)
    {
      printf("%u:", (*pos).id);

      nodeptr_list::const_iterator adjpos = (*pos).adjlist.begin();
      nodeptr_list::const_iterator adjend = (*pos).adjlist.end();

      for (; adjpos != adjend; ++adjpos)
	printf(" %u", (*adjpos)->id);
      printf("\n");
    }
    printf("\n");
  }
#endif

} graph_t;


static void __attribute__((unused)) make_a_path
(graph_t& graph, node_t* a, node_t* b, unsigned int depth)
{
  // depth the number of edges in between

  node_t* prev = a;

  for (; depth > 1; --depth)
  {
    node_t* const node = graph.at(rand() % graph.node_count());

    prev->add_adj(node);
    node->add_adj(prev);

    prev = node;
  }

  if (prev->has_adj(b) == false)
  {
    prev->add_adj(b);
    b->add_adj(prev);
  }
}


static void __attribute__((unused)) generate_random_graph
(graph_t& graph, unsigned int node_count, unsigned int node_degree)
{
  graph.initialize(node_count);

  unsigned int edge_count = node_count * node_degree / 2;
  for (; edge_count; --edge_count)
  {
    node_t* const a = graph.at(rand() % node_count);
    node_t* const b = graph.at(rand() % node_count);
   
    if ((a == b) || (a->has_adj(b))) continue ;

    a->add_adj(b);  b->add_adj(a);
  }
}


// graph loading

typedef struct mapped_file
{
  unsigned char* base;
  size_t off;
  size_t len;
} mapped_file_t;

static int map_file(mapped_file_t* mf, const char* path)
{
  int error = -1;
  struct stat st;

  const int fd = open(path, O_RDONLY);
  if (fd == -1)
    return -1;

  if (fstat(fd, &st) == -1)
    goto on_error;

  mf->base = (unsigned char*)mmap
    (NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
  if (mf->base == MAP_FAILED)
    goto on_error;

  mf->off = 0;
  mf->len = st.st_size;

  /* success */
  error = 0;

 on_error:
  close(fd);

  return error;
}

static void unmap_file(mapped_file_t* mf)
{
  munmap((void*)mf->base, mf->len);
  mf->base = (unsigned char*)MAP_FAILED;
  mf->len = 0;
}

static int next_line(mapped_file_t* mf, char* line)
{
  const unsigned char* end = mf->base + mf->len;
  const unsigned char* const base = mf->base + mf->off;
  const unsigned char* p;
  size_t skipnl = 0;
  size_t len = 0;
  char* s;

  for (p = base, s = line; p != end; ++p, ++s, ++len)
  {
    if (*p == '\n')
    {
      skipnl = 1;
      break;
    }

    *s = (char)*p;
  }

  *s = 0;

  if (p == base) return -1;

  mf->off += (p - base) + skipnl;

  return 0;
}

static bool next_edge
(mapped_file_t& mf, unsigned int& from, unsigned int& to)
{
  char line[64];
  if (next_line(&mf, line) == -1) return false;
  sscanf(line, "%u %u", &from, &to);
  return true;
}

static bool next_uint
(mapped_file_t& mf, unsigned int& ui)
{
  char line[64];
  if (next_line(&mf, line) == -1) return false;
  sscanf(line, "%u", &ui);
  return true;
}

static void load_graph(graph_t& graph, const char* path)
{
  unsigned int node_count, from, to;

  mapped_file_t mapped = {NULL, 0, 0};

  if (map_file(&mapped, path) == -1)
  {
    graph.initialize(0);
    return ;
  }

  // node count
  next_uint(mapped, node_count);
  graph.initialize(node_count);

  // edges
  while (next_edge(mapped, from, to))
    graph.at(from)->add_adj(graph.at(to));

  unmap_file(&mapped);
}


static void __attribute__((unused)) store_graph
(const graph_t& graph, const char* path)
{
  char line[256];
  int len;

  const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC);
  if (fd == -1) return ;

  // node count
  len = sprintf(line, "%u\n", graph.node_count());
  write(fd, line, strlen(line));

  // edges
  vector<node_t>::const_iterator pos = graph.nodes.begin();
  vector<node_t>::const_iterator end = graph.nodes.end();

  for (; pos != end; ++pos)
  {
    nodeptr_list::const_iterator adjpos = (*pos).adjlist.begin();
    nodeptr_list::const_iterator adjend = (*pos).adjlist.end();

    for (; adjpos != adjend; ++adjpos)
    {
#if CONFIG_NODEID
      len = sprintf(line, "%u %u\n", pos->id, (*adjpos)->id);
#endif
      write(fd, line, len);
    }
  }

  close(fd);
}


#if CONFIG_SEQUENTIAL // sequential version

static void append_node_adjlist
(node_t* node, nodeptr_list& to_visit)
{
  nodeptr_list::iterator pos = node->adjlist.begin();
  nodeptr_list::iterator end = node->adjlist.end();

  for (; pos != end; ++pos)
  {
    if ((*pos)->mark_ifnot() == true)
      to_visit.push_back(*pos);
  }
}

static unsigned int find_shortest_path_seq
(graph_t& g, node_t* from, node_t* to)
{
  // current and levels
  nodeptr_list to_visit[2];
  unsigned int depth = 0;

  // bootstrap algorithm
  to_visit[1].push_front(from);

  // while next level not empty
  while (to_visit[1].empty() == false)
  {
    // process nodes at level
    to_visit[0].swap(to_visit[1]);

    while (to_visit[0].empty() == false)
    {
      node_t* const node = to_visit[0].front();
      to_visit[0].pop_front();

      if (node == to) return depth;

      append_node_adjlist(node, to_visit[1]);
    }

    ++depth;
  }

  return 0;
}

#endif // CONFIG_SEQUENTIAL


#if CONFIG_PARALLEL // parallel version

#include "kaapi++"

// parallel work type

typedef struct par_work
{
  kaapi_workqueue_t* range;

  node_t** volatile nodes;
  size_t* volatile sums;

  node_t** adj_nodes;
  size_t* adj_sums;

  node_t* to_find;

  par_work(kaapi_workqueue_t* _range, node_t* _to_find)
    : range(_range), nodes(NULL), sums(NULL), to_find(_to_find)
  {}

  par_work
  (
   kaapi_workqueue_t* _range,
   node_t* _to_find,
   node_t** _nodes, size_t* _sums,
   node_t** _adj_nodes, size_t* _adj_sums
  )
  {
    range = _range;

    to_find = _to_find;

    nodes = _nodes;
    sums = _sums;

    adj_nodes = _adj_nodes;
    adj_sums = _adj_sums;
  }

} par_work_t;


// result types

typedef struct thief_result
{
  kaapi_workqueue_t range;

  node_t** adj_nodes;
  size_t* adj_sums;
  size_t adj_sum;
  size_t i, j;
  bool is_found;

  thief_result
  (
   kaapi_workqueue_index_t beg,
   kaapi_workqueue_index_t end,
   node_t** _adj_nodes,
   size_t* _adj_sums,
   size_t _i
  )
  {
    kaapi_workqueue_init(&range, beg, end);

    adj_nodes = _adj_nodes;
    adj_sums = _adj_sums;
    adj_sum = 0;

    i = _i;
    j = _i;

    is_found = false;
  }

  thief_result()
  {
    kaapi_workqueue_init(&range, 0, 0);
    is_found = false;
  }

} thief_result_t;

typedef thief_result_t victim_result_t;


// splitter

static void thief_entry(void*, kaapi_thread_t*, kaapi_stealcontext_t*);

static int splitter
(kaapi_stealcontext_t* ksc, int nreq, kaapi_request_t* req, void* args)
{
  par_work_t* const vw = (par_work_t*)args;

  // stolen range
  kaapi_workqueue_index_t i, j;
  kaapi_workqueue_index_t range_size;

  // reply count
  int nrep = 0;

  // size per request
  kaapi_workqueue_index_t unit_size;

 redo_steal:
  // do not steal if range size <= PAR_GRAIN
  range_size = kaapi_workqueue_size(vw->range);
  if (range_size <= CONFIG_PAR_GRAIN)
    return 0;

  // how much per req
  unit_size = range_size / (nreq + 1);
  if (unit_size == 0)
  {
    nreq = (range_size / CONFIG_PAR_GRAIN) - 1;
    unit_size = CONFIG_PAR_GRAIN;
  }

  // perform the actual steal. if the range
  // changed size in between, redo the steal
  if (kaapi_workqueue_steal(vw->range, &i, &j, nreq * unit_size))
    goto redo_steal;

  for (; nreq; --nreq, ++req, ++nrep, j -= unit_size)
  {
    // thief result
    kaapi_taskadaptive_result_t* const ktr = kaapi_request_pushdata(req, sizeof(thief_result_t));

// TODO    thief_result_t* const tres = (thief_result_t*)ktr->data;
    new (tres) thief_result_t
      (j - unit_size, j, vw->adj_nodes, vw->adj_sums, vw->sums[j - unit_size]);

    // thief work
    par_work_t* const tw = (par_work_t*)kaapi_reply_init_adaptive_task
      (ksc, req, (kaapi_task_body_t)thief_entry, sizeof(par_work_t), ktr);
    new (tw) par_work_t
      (&tres->range, vw->to_find, vw->nodes, vw->sums, vw->adj_nodes, vw->adj_sums);

    kaapi_reply_pushhead_adaptive_task(ksc, req);
  }

  return nrep;
}


// reduction

static int abort_thief
(kaapi_stealcontext_t* sc, void* targ, void* tdata, size_t tsize, void* varg)
{ return 0; }

static void abort_thieves(kaapi_stealcontext_t* ksc)
{
  kaapi_taskadaptive_result_t* ktr;
  while ((ktr = kaapi_get_thief_head(ksc)) != NULL)
    kaapi_preempt_thief(ksc, ktr, NULL, abort_thief, NULL);
}

static int common_reducer
(victim_result_t* vres, thief_result_t* tres)
{
  if (tres->is_found == true)
  {
    vres->is_found = true;
    return 0;
  }

  // reduce thief adj nodes and sums
  // todo: should be done in parallel
  const size_t thief_size = tres->j - tres->i;
  if (thief_size && (vres->j != tres->i))
  {
    for (size_t i = 0; i < thief_size; ++i)
    {
      vres->adj_nodes[vres->j + i] = tres->adj_nodes[tres->i + i];
      vres->adj_sums[vres->j + i] = tres->adj_sums[tres->i + i] + vres->adj_sum;
    }
  }
  else if (vres->j == tres->i)
  {
    for (size_t i = 0; i < thief_size; ++i)
      vres->adj_sums[vres->j + i] += vres->adj_sum;
  }

  // accumulate adjacent sum
  vres->adj_sum += tres->adj_sum;

  // adjust index
  vres->j += thief_size;

  // work continuation
  if (tres->range.beg != tres->range.end)
#warning "Now may be not safe for concurrent exec"
    kaapi_workqueue_set(&vres->range, tres->range.beg, tres->range.end);

  return 0;
}

#if 0 // unused

static int thief_reducer
(kaapi_taskadaptive_result_t* ktr, void* varg, void* targ)
{
  common_reducer((victim_result_t*)varg, (thief_result_t*)ktr->data);
  ((thief_result_t*)ktr->data)->is_reduced = true;
  return 0;
}
static int victim_reducer
(kaapi_stealcontext_t* sc, void* targ, void* tdata, size_t tsize, void* varg)
{
  if (((thief_result_t*)tdata)->is_reduced == 0)
    common_reducer((victim_result_t*)varg, (thief_result_t*)tdata);
  return 0;
}

#else

static int victim_reducer
(kaapi_stealcontext_t* sc, void* targ, void* tdata, size_t tsize, void* varg)
{
  common_reducer((victim_result_t*)varg, (thief_result_t*)tdata);
  return 0;
}

#endif


// entrypoints

static bool extract_seq
(par_work_t* pw, node_t**& beg, node_t**& end)
{
  kaapi_workqueue_index_t i, j;
  
  if (kaapi_workqueue_pop(pw->range, &i, &j, CONFIG_SEQ_GRAIN)) return false;

  beg = pw->nodes + i;
  end = pw->nodes + j;

  return true;
}

static void process_node
(node_t* node, node_t* to_find, thief_result_t& res)
{
  // add adjacent nodes to adj_nodes if not marked

  nodeptr_list::iterator pos = node->adjlist.begin();
  nodeptr_list::iterator end = node->adjlist.end();

  for (; pos != end; ++pos)
  {
    if ((*pos)->mark_ifnot() == false) continue ;

    // push adjacent node and integrate adjlist size
    res.adj_nodes[res.j] = *pos;
    res.adj_sums[res.j] = res.adj_sum;
    res.adj_sum += (*pos)->adjlist.size();
    ++res.j;
  }
}

static void thief_entry
(void* args, kaapi_thread_t* thread, kaapi_stealcontext_t* ksc)
{
  par_work_t* const pw = (par_work_t*)args;
  thief_result_t* const res = (thief_result_t*)kaapi_adaptive_result_data(ksc);
  kaapi_taskadaptive_result_t* ktr;

  // enable stealing
  kaapi_steal_setsplitter(ksc, splitter, pw);

  node_t** pos, **end;
 continue_par_work:
  while (extract_seq(pw, pos, end) == true)
  {
    const unsigned int is_preempted = kaapi_preemptpoint
      (ksc, NULL, NULL, NULL, 0, NULL);
    if (is_preempted) return ;

    for (; pos != end; ++pos)
    {
      if (*pos == pw->to_find)
      { res->is_found = true; goto on_abort; }

      process_node(*pos, pw->to_find, *res);
    }
  }

  // par_work_done. preempt and continue thieves
  if ((ktr = kaapi_get_thief_head(ksc)) != NULL)
  {
    kaapi_preempt_thief(ksc, ktr, NULL, victim_reducer, (void*)res);
    if (res->is_found == true) goto on_abort;
    goto continue_par_work;
  }
  return ;

 on_abort:
  abort_thieves(ksc);
}

static unsigned int find_shortest_path_par
(graph_t& graph, node_t* from, node_t* to)
{
  // kaapi related
  kaapi_thread_t* const thread = kaapi_self_thread();
  kaapi_stealcontext_t* ksc;
  kaapi_taskadaptive_result_t* ktr;

  // victim result
  victim_result_t res;

  // parallel work
  par_work_t pw(&res.range, to);

  // sequential range
  node_t** pos, **end;

  // current depth
  unsigned int depth = 0;

  // bootstrap algorithm: nodes, sums, index
  res.adj_nodes = (node_t**)malloc(sizeof(node_t*));
  res.adj_nodes[0] = from;

  res.adj_sum = from->adjlist.size();
  res.adj_sums = (size_t*)malloc(sizeof(size_t));
  res.adj_sums[0] = 0;

  res.j = 1;

  // enable adaptive stealing
  ksc = kaapi_task_begin_adaptive
    (thread, KAAPI_SC_CONCURRENT | KAAPI_SC_PREEMPTION, splitter, &pw);

  // while next level not empty
  while (res.j != 0)
  {
    // optim: reuse saved_nodes instead of freeing

    // adjacent layer becomes the current one
    node_t** const saved_nodes = pw.nodes;
    size_t* const saved_sums = pw.sums;
    pw.nodes = res.adj_nodes;
    pw.sums = res.adj_sums;

    // allocate next layer adjacent nodes, sums
    pw.adj_nodes = NULL;
    pw.adj_sums = NULL;
    if (res.adj_sum)
    {
      pw.adj_nodes = (node_t**)malloc(res.adj_sum * sizeof(node_t*));
      pw.adj_sums = (size_t*)malloc(res.adj_sum * sizeof(size_t));
    }

    // commit the parallel work
#warning "Now may be not safe for concurrent exec"
    kaapi_workqueue_set(&res.range, 0, (kaapi_workqueue_index_t)res.j);

    if (saved_nodes != NULL) free(saved_nodes);
    if (saved_sums != NULL) free(saved_sums);

    // prepare result with next layer
    res.adj_nodes = pw.adj_nodes;
    res.adj_sums = pw.adj_sums;
    res.j = 0;

    // initialize adjacent sum to 0
    res.adj_sum = 0;

  continue_par_work:
    // extract_seq
    while (extract_seq(&pw, pos, end) == true)
    {
      // seq_loop
      for (; pos != end; ++pos)
      {
	// if found, abort thieves
	if (*pos == to)
	{
	  // ensure no more parallel work, otherwise
	  // a thief could be missed during abortion
	  kaapi_steal_setsplitter(ksc, 0, 0);
#warning "Now may be not safe for concurrent exec"
	  kaapi_workqueue_set(pw.range, 0, 0);
	  goto on_abort;
	}

	process_node(*pos, to, res);

      } // endof_seq_loop

    } // endof_extract_seq

    // from here there wont be any parallel
    // work added until to_visit is swapped.
    // this is important since xkaapi has the
    // following constraint: we cannot preempt
    // if there is some parallel work because
    // we may miss a thief
    if ((ktr = kaapi_get_thief_head(ksc)) != NULL)
    {
      kaapi_preempt_thief(ksc, ktr, NULL, victim_reducer, (void*)&res);
      if (res.is_found == true) goto on_abort;
      goto continue_par_work;
    }

    // no more thief, no more node. next layer.

    ++depth;
  }

  // not found
  depth = 0;

 on_done:
  kaapi_task_end_adaptive(ksc);
  if (pw.nodes != NULL) free(pw.nodes);
  if (pw.sums != NULL) free(pw.sums);
  return depth;

  // abort the remaining thieves
 on_abort:
  abort_thieves(ksc);
  goto on_done;
}

#endif // CONFIG_PARALLEL


static void __attribute__((unused)) peek_random_pair
(graph_t& g, node_t*& a, node_t*& b)
{
  a = g.at(rand() % g.node_count());
  b = g.at(rand() % g.node_count());
}


static void initialize_stuff(int* ac, char*** av)
{
  srand(getpid() * time(0));

#if CONFIG_PARALLEL
  kaapi_init(1, ac, av);
#endif
}


static void finalize_stuff()
{
#if CONFIG_PARALLEL
  kaapi_finalize();
#endif
}


int main(int ac, char** av)
{
  graph_t g;
  node_t* from;
  node_t* to;
  struct timeval tms[3];

  initialize_stuff(&ac, &av);

#if !(CONFIG_PARALLEL || CONFIG_SEQUENTIAL)
  // generate and store a random graph
  const unsigned int node_count = atoi(av[1]);
  const unsigned int node_degree = atoi(av[2]);
  generate_random_graph(g, node_count, node_degree);
  store_graph(g, av[3]);
  return 0;
#endif

  load_graph(g, av[1]);
  from = g.at(atoi(av[2]));
  to = g.at(atoi(av[3]));

  unsigned int depth;
  double usecs = 0.f;

  for (unsigned int iter = 0; iter < CONFIG_ITER; ++iter)
  {
    g.unmark_nodes();
    gettimeofday(&tms[0], NULL);
#if CONFIG_SEQUENTIAL
    depth = find_shortest_path_seq(g, from, to);
#elif CONFIG_PARALLEL
    depth = find_shortest_path_par(g, from, to);
#endif
    gettimeofday(&tms[1], NULL);
    timersub(&tms[1], &tms[0], &tms[2]);
    usecs += (double)tms[2].tv_sec * 1E6 + (double)tms[2].tv_usec;
  }

  printf("%u %lf\n", depth, usecs / CONFIG_ITER);

  finalize_stuff();

  return 0;
}
