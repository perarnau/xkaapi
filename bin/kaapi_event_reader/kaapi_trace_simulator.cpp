/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** Thierry Gautier, thierry.gautier@inrialpes.fr
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
/* Open a set of event files
*/
#include "kaapi_trace_simulator.h"
#include <assert.h>
#include <vector>

struct ProcessorState {
  int      active;            /* 1 iff active, 0 iff idle */
  uint64_t taskcount;
  uint64_t stealcount;        /* successfull steal operation */
  uint64_t tactive;           /* time where is active  */
  uint64_t tidle;             /* time where is idle */
  uint64_t lastdate;          /* last date where state changed */
  uint64_t lasttactive;       /* time where is active  */
  uint64_t lasttidle;         /* time where is idle */
};

static void processor_simulate_event(
  ProcessorState& p,
  const kaapi_event_t* event
);

/* state of the simulator */
struct Simulator {
  FileSet*                    fds;
  uint64_t                    hres;  /* in ns */
  uint64_t                    tmin;  /* in s */
  uint64_t                    tmax;  /* in s */
  uint64_t                    tsim; /* in ns */
  std::vector<ProcessorState> procs;
  std::vector<double>         efficiencies;
  std::vector<uint64_t>       count_steakok;
  std::vector<uint64_t>       count_task;
};

/* Return the efficiency of the processor at the given date 
   tsim is the next sampling time.
*/
static double ProcessorEfficiencyAt( ProcessorState& proc, uint64_t tsim )
{
  uint64_t tidle;
  uint64_t tactive;
  uint64_t diff_tidle;
  uint64_t diff_tactive;
  
  if (proc.lastdate ==0) return 0;

  assert(proc.lastdate <= tsim);
  
  tidle   = proc.tidle;
  tactive = proc.tactive;

  /* update with state between lastdate and tsim */
  if (proc.active ==1) 
    tactive += (tsim - proc.lastdate);
  else
    tidle   += (tsim - proc.lastdate);

  /* monotone increasing functions bound by tsim ! */
  assert( tactive >= proc.tactive);
  assert( tidle >= proc.tidle);
  assert( tactive <= tsim );
  assert( tidle <= tsim );

  /* diff with previous sampling time at t0 */
  assert( tidle >= proc.lasttidle);
  diff_tidle  = tidle - proc.lasttidle;
  assert( tactive >= proc.lasttactive);
  diff_tactive = tactive - proc.lasttactive;
  
  /* */
  proc.lasttactive = tactive;
  proc.lasttidle   = tidle;

  
  return (double)(diff_tactive) / (double)(diff_tactive+diff_tidle);
}

static uint64_t ProcessorStealOkAt( ProcessorState& proc, uint64_t tsim )
{
  return proc.stealcount;
}

static uint64_t ProcessorTaskAt( ProcessorState& proc, uint64_t tsim )
{
  return proc.taskcount;
}

/*
*/
extern "C"
Simulator* OpenSimulator( FileSet* fds, double hres )
{
  int i, err;
  int count;
  
  Simulator* sim = new Simulator;
  if (sim ==0) return 0;
  sim->fds   = fds;
  sim->hres  = (uint64_t)(hres * 1e9); /* count in ns */
  sim->tsim  = 0;

  err = GetInterval(fds, &sim->tmin, &sim->tmax);
  if (err !=0)
    goto return_on_error;

  count = GetProcessorCount(fds);
  if (count <=0) 
    goto return_on_error;

  sim->procs.resize(count);
  sim->efficiencies.resize(count);
  sim->count_steakok.resize(count);
  sim->count_task.resize(count);
    
  for (i=0;i<count; ++i)
  {
    ProcessorState& p = sim->procs[i];
    p.active     = 0;
    p.taskcount  = 0;
    p.stealcount = 0;
    p.tactive    = 0;
    p.tidle      = 0;
    p.lastdate   = 0;
    p.lasttactive= 0;
    p.lasttidle  = 0;
  }
  
  return sim;

return_on_error:
  delete sim;
  return 0;
}

/* Do one step of simulation and return the synthetized information
   (date, count, efficiencies[i]) for each processor.
   On return the next simulation date is date+hres
*/
extern "C"
int OneStepSimulator( Simulator*       sim,
                      double*          date,
                      int*             count, 
                      const double**   efficiencies,
                      const uint64_t** count_stealok
)
{
  int retval = 0;
  const kaapi_event_t* event;
  /* simulate until sim->tsim + sim->hres */
  uint64_t nextTsim   = sim->tsim+sim->hres;
  
  if (EmptyEvent(sim->fds))
  {
    *date          = 0;
    *count         = 0;
    *efficiencies  = 0;
    *count_stealok = 0;
    return 0;
  }
  
  while (!EmptyEvent(sim->fds))
  {
    event = TopEvent(sim->fds);
    if (event->date >= nextTsim)
    {
      retval = 1;
      break;
    }
    
    processor_simulate_event( sim->procs[event->kid], event );
    
    NextEvent(sim->fds);
  }
 
  /* update global information. Do not pop event */
  sim->tsim = nextTsim;
  *date  = 1e-9*(double)nextTsim;
  *count = (int)sim->procs.size();
  
  /* compute efficiencies at date nextTsim */
  for (int i=0; i<sim->procs.size(); ++i)
  {
    sim->efficiencies[i]  = ProcessorEfficiencyAt( sim->procs[i], nextTsim );
    sim->count_steakok[i] = ProcessorStealOkAt( sim->procs[i], nextTsim );
    sim->count_task[i]    = ProcessorTaskAt( sim->procs[i], nextTsim );
  }
  *efficiencies  = &sim->efficiencies[0];
  *count_stealok = &sim->count_steakok[0];
   
  return retval;
}



/* Read and call callback on each event, ordered by date
*/
extern "C"
int CloseSimulator(Simulator* sim )
{
  delete sim;
  return 0;
}


/*
*/
static void processor_simulate_event(
  ProcessorState& p,
  const kaapi_event_t* event
)
{
  switch (event->evtno) 
  {
    case KAAPI_EVT_KPROC_START:
      p.active   = 1;
      p.lastdate = event->date;
      p.tidle    = 0;
      p.tactive  = 0;
      break;

    case KAAPI_EVT_KPROC_STOP:
      assert( event->date >= p.lastdate);
      p.tactive += (event->date - p.lastdate);
      assert( p.tactive <= event->date);
      p.lastdate = event->date;
      p.active   = 0;
      break;

    /* standard task exec */
    case KAAPI_EVT_TASK_BEG:
      break;

    case KAAPI_EVT_TASK_END:
    break;

    /* */
    case KAAPI_EVT_FRAME_TL_BEG:
    break;

    case KAAPI_EVT_FRAME_TL_END:
    break;

    /* unroll graph for static schedule */
    case KAAPI_EVT_STATIC_BEG:
    break;

    case KAAPI_EVT_STATIC_END:
    break;

    /* exec task in static graph partitioning */
    case KAAPI_EVT_STATIC_TASK_BEG:
      ++p.taskcount;
    break;

    case KAAPI_EVT_STATIC_TASK_END:
    break;

    /* idle = steal state */
    case KAAPI_EVT_SCHED_IDLE_BEG:
      assert( p.active == 1);
      assert( event->date >= p.lastdate);
      p.tactive += (event->date - p.lastdate);
      assert( p.tactive <= event->date);
      p.lastdate = event->date;
      --p.active;
    break;

    case KAAPI_EVT_SCHED_IDLE_END:
      assert( p.active == 0);
      assert( event->date >= p.lastdate);
      p.tidle   += (event->date - p.lastdate);
      assert( p.tidle <= event->date);
      p.lastdate = event->date;
      ++p.active;
    break;

    /* suspend state: may appear in idle or active state */
    case KAAPI_EVT_SCHED_SUSPEND_BEG:
      assert( event->date >= p.lastdate);
      if (--p.active == 0)
      {
        p.tactive += (event->date - p.lastdate);
        assert( p.tactive <= event->date);
      }
      else
      {
        p.tidle += (event->date - p.lastdate);
        assert( p.tidle <= event->date);
      }
      p.lastdate = event->date;
    break;

    case KAAPI_EVT_SCHED_SUSPEND_END:
      assert( event->date >= p.lastdate);
      p.tidle += (event->date - p.lastdate);
      assert( p.tidle <= event->date);
      p.lastdate = event->date;
      ++p.active;
    break;

    case KAAPI_EVT_SCHED_SUSPEND_POST:
    break;

    /* thread is always active when recording this event */
    case KAAPI_EVT_SCHED_SUSPWAIT_BEG:
      assert( event->date >= p.lastdate);
      p.tactive += (event->date - p.lastdate);
      assert( p.tactive <= event->date);
      p.lastdate = event->date;
      --p.active;
    break;

    case KAAPI_EVT_SCHED_SUSPWAIT_END:
      assert( event->date >= p.lastdate);
      p.tidle   += (event->date - p.lastdate);
      assert( p.tidle <= event->date);
      p.lastdate = event->date;
      ++p.active;
    break;

    case KAAPI_EVT_FOREACH_STEAL:
    break;

    case KAAPI_EVT_FOREACH_BEG:
    break;

    case KAAPI_EVT_FOREACH_END:
    break;

    /* processing request */
    case KAAPI_EVT_REQUESTS_BEG:
    break;

    case KAAPI_EVT_REQUESTS_END:
    break;

    /* emit steal */
    case KAAPI_EVT_STEAL_OP:
    break;

    /* emit reply */
    case KAAPI_EVT_SEND_REPLY:
    break;

    /* recv reply */
    case KAAPI_EVT_RECV_REPLY:
      if (event->d2.i !=0)
        ++p.stealcount;
    break;

    default:
      printf("***Unkown event number: %i\n", event->evtno);
      break;
  }
}
