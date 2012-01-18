/***************************************************************************
*                                                                         *
*             OpenMP MicroBenchmark Suite - Version 2.0                   *
*                                                                         *
*                            produced by                                  *
*                                                                         *
*                     Mark Bull and Fiona Reid                            *
*                                                                         *
*                                at                                       *
*                                                                         *
*                Edinburgh Parallel Computing Centre                      *
*                                                                         *
*         email: markb@epcc.ed.ac.uk or fiona@epcc.ed.ac.uk               *
*                                                                         *
*                                                                         *
*      This version copyright (c) The University of Edinburgh, 2004.      *
*                         All rights reserved.                            *
*                                                                         *
**************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <omp.h> 

#define MHZ 900 
#define OUTERREPS 20 
#define CONF95 1.96 

   int nthreads, delaylength, innerreps, itersperthr, cksz; 
   double times[OUTERREPS+1], reftime, refsd; 

   void delay(int);
   void getdelay(void); 
   void refer(void); 
   void stats(double*, double*);
   void teststatic(void); 
   void teststaticn(void); 
   void testdynamicn(void);
   void testguidedn(void); 

#ifdef EPCC_LOG
static FILE *static_log,
  **static_logs,
  **dynamic_logs,
  **guided_logs;

static int static_cpt = 0;
static int dynamic_cpt = 0;
static int guided_cpt = 0;

static struct tm *tm;
static char the_time[128];

static FILE *
create_file (const char *compiler, const char *sched, int size)
{
  char filename[256];
  
  sprintf (filename, "%s%s-%i%s", compiler, sched, size, ".log");
  return fopen (filename, "a");
}

static void
open_log_files ()
{
  struct stat file_stat;
  int i;
  int status = stat ("log", &file_stat);
  if (status == -1 && errno == ENOENT)
    mkdir ("log", S_IRWXU);

  int len = 1 + (int)(log (itersperthr) / log (2)) + 1;
  int guided_len = 1 + (int)(log (itersperthr / nthreads) / log (2)) + 1;

  char static_filename[len][128];
  char dynamic_filename[len][128];
  char guided_filename[guided_len][128];

  static_logs = malloc (len * sizeof (FILE *));
  dynamic_logs = malloc (len * sizeof (FILE *));
  guided_logs = malloc (guided_len * sizeof (FILE *));

#if defined GCC
  static_log = fopen ("log/gcc-static.log", "a");
  const char *compiler = "log/gcc-";
#elif defined FORESTGOMP
  static_log = fopen ("log/fgomp-static.log", "a");
  const char *compiler = "log/fgomp-";
#elif defined ICC
  static_log = fopen ("log/icc-static.log", "a");
  const char *compiler = "log/icc-";
#else
# warning "Building for GCC's libgomp, as you did not specify the runtime system you want to build for (ex: -DGCC, -DFORESTGOMP, -DICC)."
  static_log = fopen ("log/gcc-static.log", "a");
  const char *compiler = "log/gcc-";
#endif

  for (i = 0; i < len; i++)
    {
      static_logs[i] = create_file (compiler, "static", pow (2, i));
      dynamic_logs[i] = create_file (compiler, "dynamic", pow (2, i));

      if (i < guided_len)
	guided_logs[i] = create_file (compiler, "guided", pow (2, i));
    }
}

static void
close_log_files ()
{
  int i;
  int len = 1 + (int)(log (itersperthr) / log (2)) + 1;
  int guided_len = 1 + (int)(log (itersperthr / nthreads) / log (2)) + 1;

  for (i = 0; i < len; i++)
    {
      fclose (static_logs[i]);
      fclose (dynamic_logs[i]);

      if (i < guided_len)
	fclose (guided_logs[i]);
    }

  free (static_logs);
  free (dynamic_logs);
  free (guided_logs);
}

static void
time_pretty_print (struct tm *tm, char *the_time)
{
  sprintf (the_time, "%s%i/%s%i/%s%i", 
	   tm->tm_mday < 10 ? "0" : "", 
	   tm->tm_mday, 
	   tm->tm_mon < 10 ? "0" : "", 
	   tm->tm_mon,
	   tm->tm_year - 100 < 10 ? "0" : "",
	   tm->tm_year - 100);
}
#endif /* EPCC_LOG */

int main (int argv, char **argc)
{
#pragma omp parallel
  { 
#pragma omp master 
    {
  nthreads = omp_get_num_threads(); 
    }
  } 
 
  printf("Running OpenMP benchmark on %d thread(s)\n", nthreads); 

  /* TUNE LENGTH OF LOOP BODY */ 
  getdelay(); 
  itersperthr = 128;
  innerreps = 1000;

#ifdef EPCC_LOG
  time_t today;
  time (&today);
  tm = localtime (&today);
  time_pretty_print (tm, the_time);

  printf(" Benchmark launched on %s\n", the_time);

  open_log_files ();
#endif

  /* GENERATE REFERENCE TIME */ 
  refer();   

  /* TEST STATIC */ 
  teststatic(); 

  /* TEST STATIC,n */
  cksz = 1;
  while (cksz <= itersperthr){
    teststaticn();
    cksz *= 2;    
  }

  /* TEST DYNAMIC,n */

  cksz = 1;
  while (cksz <= itersperthr){
    testdynamicn();
    cksz *= 2;    
  }

  /* TEST GUIDED,n */
  cksz = 1;
  while (cksz <= itersperthr/nthreads){
    testguidedn();
    cksz *= 2;    
  }
  
#ifdef EPCC_LOG
  close_log_files ();
#endif
}

void getdelay()
{
  int i,reps; 
  double actualtime, targettime, start; 

  double getclock(void); 

  /*  
      CHOOSE delaylength SO THAT call delay(delaylength) 
      TAKES APPROXIMATELY 100 CPU CLOCK CYCLES 
  */ 

  delaylength = 0;
  reps = 10000;

  actualtime = 0.;
  targettime = 100.0 / (double) MHZ * 1e-06;

  delay(delaylength); 

  while (actualtime < targettime) {
    delaylength = delaylength * 1.1 + 1; 
    start = getclock();
    for (i=0; i< reps; i++) {
      delay(delaylength); 
    }
    actualtime  = (getclock() - start) / (double) reps; 
  }

  start = getclock();
  for (i=0; i< reps; i++) {
    delay(delaylength); 
  }
  actualtime  = (getclock() - start) / (double) reps; 

  printf("Assumed clock rate = %d MHz \n",MHZ); 
  printf("Delay length = %d\n", delaylength); 
  printf("Delay time  = %f cycles\n",  actualtime * MHZ * 1e6); 

}

void refer()
{
  int i,j,k; 
  double start; 
  double meantime, sd; 

  double getclock(void); 

  printf("\n");
  printf("--------------------------------------------------------\n");
  printf("Computing reference time\n"); 

  for (k=0; k<=OUTERREPS; k++){
    start  = getclock(); 
    for (j=0; j<innerreps; j++){
      for (i=0; i<itersperthr; i++){
	delay(delaylength); 
      }
    }
    times[k] = (getclock() - start) * 1.0e6 / (double) innerreps;
  }

  stats (&meantime, &sd);

  printf("Reference_time_1 =                        %f microseconds +/- %f\n", meantime, CONF95*sd);

  reftime = meantime;
  refsd = sd;  
}

void teststatic()
{

  int i,j,k; 
  double start; 
  double meantime, sd; 

  double getclock(void); 

  printf("\n");
  printf("--------------------------------------------------------\n");
  printf("Computing STATIC time\n"); 

  for (k=0; k<=OUTERREPS; k++){
    start  = getclock(); 
#pragma omp parallel private(j) 
      { 
	for (j=0; j<innerreps; j++){
#pragma omp for schedule(static)  
	  for (i=0; i<itersperthr*nthreads; i++){
	    delay(delaylength); 
	  }
	}
      }
    times[k] = (getclock() - start) * 1.0e6 / (double) innerreps;
  }

  stats (&meantime, &sd);

  printf("STATIC time =                           %f microseconds +/- %f\n", meantime, CONF95*sd);
  
  printf("STATIC overhead =                       %f microseconds +/- %f\n", meantime-reftime, CONF95*(sd+refsd));

#ifdef EPCC_LOG
  fprintf (static_log, "%s        %f %f\n", the_time, meantime-reftime, CONF95*(sd+refsd));
#endif /* EPCC_LOG */
}

void teststaticn()
{

  int i,j,k; 
  double start; 
  double meantime, sd; 

  double getclock(void); 

  printf("\n");
  printf("--------------------------------------------------------\n");
  printf("Computing STATIC %d time\n",cksz); 

  for (k=0; k<=OUTERREPS; k++){
    start  = getclock(); 
#pragma omp parallel private(j) 
      { 
	for (j=0; j<innerreps; j++){
#pragma omp for schedule(static,cksz)  
	  for (i=0; i<itersperthr*nthreads; i++){
	    delay(delaylength); 
	  }
	}
      }
    times[k] = (getclock() - start) * 1.0e6 / (double) innerreps;
  }

  stats (&meantime, &sd);

  printf("STATIC %d time =                           %f microseconds +/- %f\n", cksz, meantime, CONF95*sd);
  
  printf("STATIC %d overhead =                       %f microseconds +/- %f\n", cksz, meantime-reftime, CONF95*(sd+refsd));

#ifdef EPCC_LOG
  fprintf (static_logs[static_cpt++], "%s        %f %f\n", the_time, meantime-reftime, CONF95*(sd+refsd));
#endif /* EPCC_LOG */
}

void testdynamicn()
{

  int i,j,k; 
  double start; 
  double meantime, sd; 

  double getclock(void); 

  printf("\n");
  printf("--------------------------------------------------------\n");
  printf("Computing DYNAMIC %d time\n",cksz); 

  for (k=0; k<=OUTERREPS; k++){
    start  = getclock(); 
#pragma omp parallel private(j) 
      { 
	for (j=0; j<innerreps; j++){
#pragma omp for schedule(dynamic,cksz)  
	  for (i=0; i<itersperthr*nthreads; i++){
	    delay(delaylength); 
	  }
	}
      }
    times[k] = (getclock() - start) * 1.0e6 / (double) innerreps;
  }

  stats (&meantime, &sd);

  printf("DYNAMIC %d time =                           %f microseconds +/- %f\n", cksz, meantime, CONF95*sd);
  
  printf("DYNAMIC %d overhead =                       %f microseconds +/- %f\n", cksz, meantime-reftime, CONF95*(sd+refsd));

#ifdef EPCC_LOG
  fprintf (dynamic_logs[dynamic_cpt++], "%s        %f %f\n", the_time, meantime-reftime, CONF95*(sd+refsd));
#endif /* EPCC_LOG */
}


void testguidedn()
{

  int i,j,k; 
  double start; 
  double meantime, sd; 

  double getclock(void); 

  printf("\n");
  printf("--------------------------------------------------------\n");
  printf("Computing GUIDED %d time\n",cksz); 

  for (k=0; k<=OUTERREPS; k++){
    start  = getclock(); 
#pragma omp parallel private(j) 
      { 
	for (j=0; j<innerreps; j++){
#pragma omp for schedule(guided,cksz)  
	  for (i=0; i<itersperthr*nthreads; i++){
	    delay(delaylength); 
	  }
	}
      }
    times[k] = (getclock() - start) * 1.0e6 / (double) innerreps;
  }

  stats (&meantime, &sd);

  printf("GUIDED %d time =                           %f microseconds +/- %f\n", cksz, meantime, CONF95*sd);
  
  printf("GUIDED %d overhead =                       %f microseconds +/- %f\n", cksz, meantime-reftime, CONF95*(sd+refsd));

#ifdef EPCC_LOG
  fprintf (guided_logs[guided_cpt++], "%s        %f %f\n", the_time, meantime-reftime, CONF95*(sd+refsd));
#endif /* EPCC_LOG */
}

void stats (double *mtp, double *sdp) 
{

  double meantime, totaltime, sumsq, mintime, maxtime, sd, cutoff; 

  int i, nr; 

  mintime = 1.0e10;
  maxtime = 0.;
  totaltime = 0.;

  for (i=1; i<=OUTERREPS; i++){
    mintime = (mintime < times[i]) ? mintime : times[i];
    maxtime = (maxtime > times[i]) ? maxtime : times[i];
    totaltime +=times[i]; 
  } 

  meantime  = totaltime / OUTERREPS;
  sumsq = 0; 

  for (i=1; i<=OUTERREPS; i++){
    sumsq += (times[i]-meantime)* (times[i]-meantime); 
  } 
  sd = sqrt(sumsq/(OUTERREPS-1));

  cutoff = 3.0 * sd; 

  nr = 0; 
  
  for (i=1; i<=OUTERREPS; i++){
    if ( fabs(times[i]-meantime) > cutoff ) nr ++; 
  }
  
  printf("\n"); 
  printf("Sample_size       Average     Min         Max          S.D.          Outliers\n");
  printf(" %d                %f   %f   %f    %f      %d\n",OUTERREPS, meantime, mintime, maxtime, sd, nr); 
  printf("\n");

  *mtp = meantime; 
  *sdp = sd; 

} 

