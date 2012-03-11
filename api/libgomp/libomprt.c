/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** francois.broquedis@imag.fr
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
#include "libgomp.h"
#include <stdio.h>

void
omp_set_num_threads (int n)
{
  if (n >0)
  {
    kompctxt_t* ctxt = komp_get_ctxt();
    ctxt->icv.nextnumthreads = n;
  }
}

int
omp_get_num_threads (void)
{
  kompctxt_t* ctxt = komp_get_ctxt();
  if (ctxt->teaminfo ==0) return 1;
  return ctxt->teaminfo->numthreads;
}

int
omp_get_thread_num (void)
{
  return komp_get_ctxt()->icv.threadid;
}

/*
*/
int 
omp_get_max_threads (void)
{
  kompctxt_t* ctxt = komp_get_ctxt();
  if (ctxt->icv.nextnumthreads < kaapi_getconcurrency())
    return ctxt->icv.nextnumthreads;
  return kaapi_getconcurrency();
}

int omp_get_num_procs (void)
{
  return kaapi_getconcurrency();
}

/*
*/
int 
omp_in_parallel(void)
{
  kompctxt_t* ctxt = komp_get_ctxt();
  return ctxt->teaminfo !=0;
}

/*
*/
void 
omp_set_dynamic(int dynamic_threads __attribute__((unused)) )
{
}

/*
*/
int 
omp_get_dynamic(void)
{
  return 1;
}

/*
*/
void 
omp_set_nested(int nested __attribute__((unused)))
{
  kompctxt_t* ctxt = komp_get_ctxt();
  ctxt->icv.nestedparallel = (nested !=0);
}

/*
*/
int 
omp_get_nested(void)
{
  kompctxt_t* ctxt = komp_get_ctxt();
  return (ctxt->icv.nestedparallel !=0);
}

/*
*/
void 
omp_set_schedule(omp_sched_t kind __attribute__((unused)), int modifier __attribute__((unused)))
{
}

/*
*/
void 
omp_get_schedule(omp_sched_t * kind, int * modifier )
{
  *kind = omp_sched_dynamic;
  *modifier = -1;
}

/*
*/
int 
omp_get_thread_limit(void)
{ 
  return kaapi_getconcurrency();
}

/*
*/
void 
omp_set_max_active_levels (int max_levels __attribute__((unused)))
{
}

/*
*/
int 
omp_get_max_active_levels(void)
{
  return 1;
}

/*
*/
int 
omp_get_level(void)
{
  kompctxt_t* ctxt = komp_get_ctxt();
  if (ctxt->teaminfo ==0) 
    return 0;
  return 1;
}

/*
*/
int 
omp_get_ancestor_thread_num(int level)
{
  return kaapi_getconcurrency();
}

/*
*/
int 
omp_get_team_size(int level)
{
  kompctxt_t* ctxt = komp_get_ctxt();
  if (ctxt->teaminfo ==0) return 1;
  return kaapi_getconcurrency();
}

/*
*/
int 
omp_get_active_level(void)
{
  kompctxt_t* ctxt = komp_get_ctxt();
  if (ctxt->teaminfo ==0) return 0;
  return 1;
}

/*
*/
int 
omp_in_final(void)
{
  return 0;
}

double omp_get_wtime(void)
{
  return kaapi_get_elapsedtime();
}

double omp_get_wtick(void)
{
  return 1e6; /* elapsed time is assumed to be in micro second ?? */
}

