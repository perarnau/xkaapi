/* =========================================================================
* (c) INRIA, projet MOAIS, 2007
* Author: T. Gautier
*
* ========================================================================= */
#include <stdarg.h>
#include <ctype.h>
#include "kaapi_fortran_impl.h"

/* Experimental API for Fortran */

struct Graph2Sched {
  DFG::Thread* thread;
  RFO::Frame*  frame;
};

/* -------------------------------------------------------------------- */
extern "C" 
void kaapi_graph_begin_( KAAPI_Graph* fgraph, KAAPI_Fint* ierr )
{ 
  *ierr  = 0;
  *graph = 0;
  
  Graph2Sched* graph = new Graph2Sched;
  graph->thread = (DFG::Thread*)Core::Thread::get_current();
  graph->frame = new (graph->thread->allocate(sizeof(RFO::Frame))) RFO::Frame;
  
  /* push the frame in order to capture tasks that will be scheduled */
  graph->thread->push( graph->frame );
  *fgraph = (KAAPI_Graph*)graph;
}


/* -------------------------------------------------------------------- */
extern "C" 
void kaapi_graph_end_( KAAPI_Graph* fgraph, KAAPI_Fint* ierr )
{ 
  /* get the graph */
  Graph2Sched* graph = (Graph2Sched*)*fgraph;

  *ierr = 0;
}


/* -------------------------------------------------------------------- */
extern "C" 
void kaapi_graph_schedule_( KAAPI_Graph* fgraph, const char* attribut, KAAPI_Fint* ierr )
{ 
  /* get the graph */
  Graph2Sched* graph = (Graph2Sched*)*fgraph;

  *ierr = 0;
}


/* -------------------------------------------------------------------- */
extern "C" 
void kaapi_graph_execute_( KAAPI_Graph* fgraph, KAAPI_Fint* iteration, KAAPI_Fint* ierr )
{ 
  /* get the graph */
  Graph2Sched* graph = (Graph2Sched*)*fgraph;

  *ierr = 0;
}
