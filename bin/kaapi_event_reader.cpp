/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
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
#include <stdlib.h>
#include "kaapi_impl.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <float.h>
#include "PajeGenerator.hpp"

/** see kaapi_event.h for coding of event name */
static const char* kaapi_event_name[] = {
  "<nop>",
  "TaskBegin",
  "TaskEnd",
  "BeginFrame",
  "EndFrame",
  "StaticBeg",
  "StaticEnd",
  "UnrollBeg",
  "UnrollEnd",
  "IdleBeg",
  "IdleEnd"
};


/* Type of function to process data
*/
typedef void (*kaapi_fnc_event)( const char*, int );



/*
*/
static void print_usage()
{
  fprintf(stderr, "Merge and convert internal Kaapi trace format to human readeable formats\n");
  fprintf(stderr, "*** options: -a | -m | -s. Only one of the options may be selected at a time.\n");
  fprintf(stderr, "  -a: display all data associated to each events\n");
  fprintf(stderr, "  -m: display the mapping of tasks\n");
  fprintf(stderr, "  -s: display stats about runtime of all the tasks\n");
  fprintf(stderr, "  -g: output Gnuplot commands for Gantt diagram, one row per core. \n");
  fprintf(stderr, "      Output filename is gantt.gnuplot\n");
  fprintf(stderr, "  -p: output Paje format for Gantt diagram, one row per core\n");
  fprintf(stderr, "      Output filename is gantt.paje\n");
  exit(1);
}

/*
*/
static void fnc_print_evt( const char* filename, int fd )
{
  kaapi_event_buffer_t evb;
  kaapi_event_t* e = evb.buffer;

  while (1)
  {
    ssize_t sz_read = read(fd, evb.buffer, KAAPI_EVENT_BUFFER_SIZE*sizeof(kaapi_event_t) );
    if (sz_read ==0) break;
    if (sz_read ==-1)
    {
      fprintf(stderr, "*** error while reading file '%s' : %s", filename, strerror(errno));
      break;
    }
    ssize_t sevents = sz_read/sizeof(kaapi_event_t);

    for (ssize_t i=0; i< sevents; ++i)
    {
      /* full print */
      /*         date         kid          name  */
//      printf( "%" PRIu64": %" PRIu16 " -> %6s |", e[i].date, e[i].kid, kaapi_event_name[e[i].evtno] );
      std::cout << e[i].date << ": " << e[i].kid << " -> " << kaapi_event_name[e[i].evtno];
      switch (e[i].evtno) {
        case KAAPI_EVT_TASK_BEG:
        case KAAPI_EVT_TASK_END:
//          printf(" %p", e[i].d0.p );
          std::cout << (void*)e[i].d0.p;
          break;
        default:
          break;
      }
//      printf("\n");
      std::cout << std::endl;
    }
  }
}

/*
*/
static void fnc_print_map_task_kid( const char* filename, int fd )
{
  kaapi_event_buffer_t evb;
  kaapi_event_t* e = evb.buffer;

  while (1)
  {
    ssize_t sz_read = read(fd, evb.buffer, KAAPI_EVENT_BUFFER_SIZE*sizeof(kaapi_event_t) );
    if (sz_read ==0) break;
    if (sz_read ==-1)
    {
      fprintf(stderr, "*** error while reading file '%s' : %s", filename, strerror(errno));
      break;
    }
    ssize_t sevents = sz_read/sizeof(kaapi_event_t);

    for (ssize_t i=0; i< sevents; ++i)
    {
      switch (e[i].evtno) {
        case KAAPI_EVT_TASK_BEG:
          std::cout << e[i].kid << " " << (void*)e[i].d0.p << std::endl;
//          printf( "%" PRIu16" %p\n", e[i].kid, e[i].d0.p );
          break;
        default:
          break;
      }
    }
  }
}


/*
*/
static void fnc_statistic_task_kid( const char* filename, int fd )
{
  double   t_min   = 0.0;
  double   t_max   = 0.0;
  double   t_total = 0.0;
  double   t_var2  = 0.0;
  uint64_t count   = 0.0;
  uint64_t t_idle  = 0;

  kaapi_event_buffer_t evb;
  kaapi_event_t* e = evb.buffer;
    
  uint64_t d0 =0, d1 =0;
  uint64_t d0_sched =0, d1_sched =0;
  void* task =0;

  while (1)
  {
    ssize_t sz_read = read(fd, evb.buffer, KAAPI_EVENT_BUFFER_SIZE*sizeof(kaapi_event_t) );
    if (sz_read ==0) break;
    if (sz_read ==-1)
    {
      fprintf(stderr, "*** error while reading file '%s' : %s", filename, strerror(errno));
      break;
    }
    ssize_t sevents = sz_read/sizeof(kaapi_event_t);

    for (ssize_t i=0; i< sevents; ++i)
    {
      switch (e[i].evtno) {
        case KAAPI_EVT_TASK_BEG:
          d0   = e[i].date;
          task = e[i].d0.p;
          break;
        case KAAPI_EVT_TASK_END:
          if (e[i].d0.p != task)
          {
            printf("*** invalid events format, ignore time for task: %p\n", task );
          }
          else {
            d1 = e[i].date;
            uint64_t d = d1-d0;
            double v = 1.0e-9*(double)d; /* time in s */
            if (count ==0) 
            { 
              t_max    = v;
              t_min    = v;
              t_total  = v;
              t_var2   = v*v;
            } else {
              if (v > t_max) t_max = v;
              if (v < t_min) t_min = v;
              t_total += v;
              t_var2  += v*v;
            }
            ++count;
          }
          task = 0;
        case KAAPI_EVT_SCHED_IDLE_BEG:
          d0_sched = e[i].date;
        break;

        case KAAPI_EVT_SCHED_IDLE_END:
          d1_sched = e[i].date;
          t_idle   += (d1_sched - d0_sched);
        break;

        default:
          break;
      }
    }
  }

  
  /* print statistic */
  double average  = t_total / (double)count;
  double variance = t_var2/(double)count - average * average;
  double stddev = sqrt( variance );  
  size_t c = count;
  if (c >1) --c;
  double delta = 2.576 * sqrt( variance / c);
  
  printf("*** Statistics:\n");
  printf("Ttotal    (s)     : %e\n", t_total);
  printf("Tidle     (s)     : %e\n", 1.0e-9 * (double)t_idle);
  printf("Count     (#tasks): %llu\n", count);
//  printf("Count     (#tasks): %llu\n", count);
  printf("Tmin      (s)     : %e\n", t_min);
  printf("Tmax      (s)     : %e\n", t_max);
  printf("Taverage  (s)     : %e\n", average);
  printf("Tvariance (s^2)   : %e\n", variance);
  printf("Tstd_dev  (s)     : %e\n", stddev);
  printf("IC, 95%%   (s)     : %e\n", delta);  
}


/*
*/
static FILE* output_gnuplot = 0;
static double t_min_gp = DBL_MAX;
static double t_max_gp = DBL_MIN;
static int gp_core = 1; 
static void fnc_gnuplot_gantt( const char* filename, int fd )
{
  static int count = 1;  
  kaapi_event_buffer_t evb;
  kaapi_event_t* e = evb.buffer;
    
  double d0 =0.0, d1 =0.0;
  void* task =0;

  while (1)
  {
    ssize_t sz_read = read(fd, evb.buffer, KAAPI_EVENT_BUFFER_SIZE*sizeof(kaapi_event_t) );
    if (sz_read ==0) break;
    if (sz_read ==-1)
    {
      fprintf(stderr, "*** error while reading file '%s' : %s", filename, strerror(errno));
      break;
    }
    ssize_t sevents = sz_read/sizeof(kaapi_event_t);

    for (ssize_t i=0; i< sevents; ++i)
    {
      switch (e[i].evtno) {
        case KAAPI_EVT_TASK_BEG:
          d0   = 1e-9*(double)e[i].date;
          task = e[i].d0.p;
          if (d0 < t_min_gp) t_min_gp = d0;
          if (d0 > t_max_gp) t_max_gp = d0;
          break;
        case KAAPI_EVT_TASK_END:
          if (e[i].d0.p != task)
          {
            printf("*** invalid events format, ignore time for task: %p\n", task );
          }
          else {
            d1 = 1e-9*(double)e[i].date;
            fprintf(output_gnuplot,"set object %i rect from %f,%f to %f,%f fc lt %i\n", 
                count, d0, ((double)gp_core)-1.0, d1, ((double)gp_core)-0.1, gp_core );
            ++count;
          }
          task = 0;

        case KAAPI_EVT_FRAME_TL_BEG:
          d0   = (double)e[i].date;
        break;

        case KAAPI_EVT_FRAME_TL_END:
          d1   = (double)e[i].date;
        break;

        case KAAPI_EVT_STATIC_BEG:
          d0   = (double)e[i].date;
        break;

        case KAAPI_EVT_STATIC_END:
          d1   = (double)e[i].date;
        break;

        case KAAPI_EVT_STATIC_TASK_BEG:
          task = e[i].d0.p;
          d0   = 1e-9*(double)e[i].date;
          if (d0 < t_min_gp) t_min_gp = d0;
          if (d0 > t_max_gp) t_max_gp = d0;
        break;

        case KAAPI_EVT_STATIC_TASK_END:
          d1   = (double)e[i].date;
          if (e[i].d0.p != task)
          {
            printf("*** invalid events format, ignore time for task: %p\n", task );
          }
          else {
            d1 = 1e-9*(double)e[i].date;
            fprintf(output_gnuplot,"set object %i rect from %f,%f to %f,%f fc lt 1\n", 
                count, d0, ((double)gp_core)-1.0, d1, ((double)gp_core)-0.1 );
            ++count;
          }
          task = 0;
        break;

        case KAAPI_EVT_SCHED_IDLE_BEG:
          d0   = (double)e[i].date;
          task = (void*)-8;
        break;

        case KAAPI_EVT_SCHED_IDLE_END:
          d1   = (double)e[i].date;
          if (task != (void*)-8)
          {
            printf("*** invalid events format, ignore time for task: %p\n", task );
          }
          else {
            d1 = 1e-9*(double)e[i].date;
            fprintf(output_gnuplot,"set object %i rect from %f,%f to %f,%f fc lt 0\n", 
                count, d0, ((double)gp_core)-1.0, d1, ((double)gp_core)-0.1 );
            ++count;
          }

        break;
        default:
          break;
      }
    }
  } // while
  ++gp_core;
}


static PajeGenerator* pg =0;
static double tmin = DBL_MAX;
static void fnc_paje_gantt( const char* filename, int fd )
{
  static int core  = 0; 
  kaapi_event_buffer_t evb;
  kaapi_event_t* e = evb.buffer;
    

  double d0 =0.0, d1 = 0.0;
  void* task = 0;

  while (1)
  {
    ssize_t sz_read = read(fd, evb.buffer, KAAPI_EVENT_BUFFER_SIZE*sizeof(kaapi_event_t) );
    if (sz_read ==0) break;
    if (sz_read ==-1)
    {
      fprintf(stderr, "*** error while reading file '%s' : %s", filename, strerror(errno));
      break;
    }
    ssize_t sevents = sz_read/sizeof(kaapi_event_t);

    for (ssize_t i=0; i< sevents; ++i)
    {
      if (tmin == DBL_MAX) tmin = (double)e[i].date;;
      switch (e[i].evtno) {
        case KAAPI_EVT_TASK_BEG:
          d0   = (double)e[i].date;
          task = e[i].d0.p;
          pg->addState( e[i].kid, KAAPI_EVT_TASK_BEG-1, 1e-9*(d0-tmin) );
          break;

        case KAAPI_EVT_TASK_END:
          if (e[i].d0.p != task)
          {
            printf("*** invalid events format, ignore time for task: %p\n", task );
          }
          else {
            d1 = (double)e[i].date;
            pg->addState( e[i].kid, KAAPI_EVT_TASK_END-1, 1e-9*(d1-tmin) );
          }
          task = 0;
        break;

        case KAAPI_EVT_FRAME_TL_BEG:
          d0   = (double)e[i].date;
          pg->addState( e[i].kid, KAAPI_EVT_FRAME_TL_BEG-1, 1e-9*(d0-tmin) );
        break;

        case KAAPI_EVT_FRAME_TL_END:
          d1   = (double)e[i].date;
          pg->addState( e[i].kid, KAAPI_EVT_FRAME_TL_END-1, 1e-9*(d1-tmin) );
        break;

        case KAAPI_EVT_STATIC_BEG:
          d0   = (double)e[i].date;
          pg->addState( e[i].kid, KAAPI_EVT_STATIC_BEG-1, 1e-9*(d0-tmin) );
        break;

        case KAAPI_EVT_STATIC_END:
          d1   = (double)e[i].date;
          pg->addState( e[i].kid, KAAPI_EVT_STATIC_END-1, 1e-9*(d1-tmin) );
        break;

        case KAAPI_EVT_STATIC_TASK_BEG:
          d0   = (double)e[i].date;
          pg->addState( e[i].kid, KAAPI_EVT_STATIC_TASK_BEG-1, 1e-9*(d0-tmin) );
        break;

        case KAAPI_EVT_STATIC_TASK_END:
          d1   = (double)e[i].date;
          pg->addState( e[i].kid, KAAPI_EVT_STATIC_TASK_END-1, 1e-9*(d1-tmin) );
        break;

        case KAAPI_EVT_SCHED_IDLE_BEG:
          d0   = (double)e[i].date;
          pg->addState( e[i].kid, KAAPI_EVT_SCHED_IDLE_BEG-1, 1e-9*(d0-tmin) );
        break;

        case KAAPI_EVT_SCHED_IDLE_END:
          d1   = (double)e[i].date;
          pg->addState( e[i].kid, KAAPI_EVT_SCHED_IDLE_END-1, 1e-9*(d1-tmin) );
        break;

        default:
          printf("***Unkown event number: %i\n", e[i].evtno);
          break;
      }
    }
  } // while
  ++core;
}


/* Parse options:
   * nooption : nocode: print usage
   * -a       : 0 : display all data, using function fnc_print_evt
   * -m       : 1 : display the mapping of tasks only, use function fnc_print_map_task_kid
   * -s       : 2 : display statistics
   * -g       : 2 : display gnuplot gantt
   * -g       : 2 : display paje gantt
*/
static kaapi_fnc_event parse_option( int* argc, char*** argv)
{
  kaapi_fnc_event function = 0;
  char** destargv = *argv;
  int newargc =0;
  for (int i=0; i<*argc; ++i)
  {
    if (strcmp((*argv)[i], "-a") ==0)
    {
      if (function !=0) 
        print_usage();
      function = fnc_print_evt;
    }
    else if (strcmp((*argv)[i], "-m") ==0)
    {
      if (function !=0) 
        print_usage();
      function = fnc_print_map_task_kid;
    }
    else if (strcmp((*argv)[i], "-s") ==0)
    {
      if (function !=0) 
        print_usage();
      function = fnc_statistic_task_kid;
    }
    else if (strcmp((*argv)[i], "-g") ==0)
    {
      if (function !=0) 
        print_usage();
      if (output_gnuplot ==0)
      {
        output_gnuplot = fopen("gantt.gnuplot", "w");
        if (output_gnuplot ==0)
        {
          fprintf(stderr, "*** cannot open file: 'gantt.gnuplot'\n");
          exit(1);
        }
      }
      function = fnc_gnuplot_gantt;
    }
    else if (strcmp((*argv)[i], "-p") ==0)
    {
      if (pg ==0)
      {
        pg = new PajeGenerator;
        int err = pg->initTrace (
          "gantt.paje",
          2,
          KAAPI_MAX_PROCESSOR, /* as defined in config.h*/
          KAAPI_EVT_NUMBER, /* number of events */
          0,
          0,
          0
        );
        if (err !=0) 
        {
          fprintf(stderr, "*** cannot open file: 'gantt.gnuplot'\n");
          exit(1);
        }
      }
      if (function !=0) 
        print_usage();
      function = fnc_paje_gantt;
    }
    else {
      *destargv++ = (*argv)[i];
      ++newargc;
    }
  }
  if (function ==0) print_usage();
  *destargv = 0;
  *argc = newargc;
  return function;
}


/* main entry point : Kaapi initialization
*/
int main(int argc, char** argv)
{
  kaapi_fnc_event function = parse_option( &argc, &argv );
  
  for (int i=1; i<argc; ++i)
  {
    const char* filename = argv[i];
    int fd = open(filename, O_RDONLY);
    if (fd == -1) 
    {
      fprintf(stderr, "*** cannot open file '%s'\n", filename);
      continue;
    }
    fprintf(stdout, "*** file '%s'\n", filename);
    
    function( filename, fd );

    close(fd);
  }

  if (output_gnuplot !=0) 
  {
    double delta = (t_max_gp-t_min_gp);
    fprintf(output_gnuplot,"set xrange [%f:%f ]\n", t_min_gp -delta*0.01, t_max_gp +delta*0.1);
    fprintf(output_gnuplot,"set yrange [%i:%i ]\n", 0, gp_core-1 );
    fprintf(output_gnuplot,"set key outside bottom\n");
    fprintf(output_gnuplot,"set ytics 1\n" );
    fprintf(output_gnuplot,"set ylabel \"Core identifier\"\n" );
    fprintf(output_gnuplot,"set xlabel \"Time\"\n" );
    fprintf(output_gnuplot,"set title \"Gantt\"\n" );
    fprintf(output_gnuplot, "plot ");
    for (int i=0; i<gp_core-1; ++i)
    {
      fprintf(output_gnuplot,"%i lc %i lw 4 title \"core %i\"", i, i+1, i );
      if (i +1 < gp_core-1)
        fprintf(output_gnuplot,", " );
    }
    
    fclose(output_gnuplot);
    output_gnuplot = 0;
  }
  if (pg !=0) 
  {
    pg->endTrace();
    delete pg;
    pg = 0;
  }

  return 0;
}
