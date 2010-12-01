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
#include "kaapi_impl.h"
#include "ka_init.h"
#include "ka_timer.h"
#include "ka_error.h"
#include "ka_debug.h"
#include <iomanip>

extern "C" {
# include <sys/time.h>
#if not defined (_WIN32)
# include <sys/resource.h>
#endif
# include <stdio.h>
# include <unistd.h>
#if defined( KAAPI_USE_APPLE ) || defined(KAAPI_USE_IPHONEOS)
# include <sys/types.h>
# include <sys/sysctl.h>
#endif
}


namespace ka {

// --------------------------------------------------------------------
const std::string& HighResTimer::unit()
{
  static std::string u = "s";
  return u;
}

// --------------------------------------------------------------------
const std::string& WallTimer::unit()
{
  static std::string u    = "s";
  return u;
}

// --------------------------------------------------------------------
const std::string& CpuTimer::unit()
{
  static std::string u     = "s";
  return u;
}

// --------------------------------------------------------------------
const std::string& SysTimer::unit()
{
  static std::string u     = "s";
  return u;
}

// --------------------------------------------------------------------
const std::string& LogicalTimer::unit()
{
  static std::string u = "#";
  return u;
}

// --------------------------------------------------------------------
HighResTimer::type HighResTimer::gettick()
{
#if defined(KAAPI_USE_ARCH_PPC) || defined(KAAPI_USE_ARCH_PPC64)
  /* Linux or Mac */
  register unsigned long t_u;
  register unsigned long t_l;
  asm volatile ("mftbu %0" : "=r" (t_u) );
  asm volatile ("mftb %0" : "=r" (t_l) );
  HighResTimer::type retval = t_u;
  retval <<= 32UL;
  retval |= t_l;
  return retval;
#elif defined(KAAPI_USE_ARCH_X86) || defined(KAAPI_USE_ARCH_IA64)
  uint32_t lo,hi;
  __asm__ volatile ( "rdtsc" : "=a" ( lo ) , "=d" ( hi ) );
  return (uint64_t)hi << 32UL | lo;
#elif defined(KAAPI_USE_ARCH_ITA)
  register unsigned long ret;
  __asm__ __volatile__ ("mov %0=ar.itc" : "=r"(ret));
  return ret;
#elif defined(KAAPI_USE_IPHONEOS)
  return WallTimer::gettime();
#elif defined(_WIN32)
  return WallTimer::gettime();
#else
#  error "no time counter for this architecture"
#endif
}


// --------------------------------------------------------------------
uint64_t HighResTimer::latency =0;
double HighResTimer::dtick_per_s =0;
double HighResTimer::inv_dtick_per_s =0;




// --------------------------------------------------------------------
//#define USE_SOFT_CALIBRATION

void HighResTimer::calibrate()
{
#if defined(USE_SOFT_CALIBRATION)
  size_t count=0;
  double t00=0, t11=0;
#  if defined(KAAPI_USE_ARCH_PPC) || defined(KAAPI_USE_ARCH_PPC64) || defined(KAAPI_USE_ARCH_X86) || defined(KAAPI_USE_ARCH_IA64)
  HighResTimer::type hrt0, hrt1, hrt2;
  hrt0 = HighResTimer::gettick();
#  else 
#    error "Cannot make soft calibration of clock"
#  endif
  /* loop 1 */
  for (int i=0; i<128; ++i)
    t00 += WallTimer::gettime();
#  if defined(KAAPI_USE_ARCH_PPC) || defined(KAAPI_USE_ARCH_PPC64) || defined(KAAPI_USE_ARCH_X86) || defined(KAAPI_USE_ARCH_IA64)
  hrt1 = HighResTimer::gettick();
#  endif

  /* loop 2 */
  t00 = WallTimer::gettime();
  for (int i=0; i<200; ++i)
    for (int j=0; j<100000; ++j)
      count++;
  t11 = WallTimer::gettime();
#  if defined(KAAPI_USE_ARCH_PPC) || defined(KAAPI_USE_ARCH_PPC64) || defined(KAAPI_USE_ARCH_X86) || defined(KAAPI_USE_ARCH_IA64)
  hrt2 = HighResTimer::gettick();
#  endif

#  if defined(KAAPI_USE_ARCH_PPC) || defined(KAAPI_USE_ARCH_PPC64) || defined(KAAPI_USE_ARCH_X86) || defined(KAAPI_USE_ARCH_IA64)
  /* hrt0: ticks for 128 gettimeofday */
  hrt0 = hrt1 - hrt0;

  /* hrt1: ticks for the loops + 2 gettimeofday -> hrt1 : loop 2 */
  hrt1 = hrt2 - hrt1;

  /* ticks for loop */
  double d = double(hrt1)-double(hrt0)/64.0;
  HighResTimer::dtick_per_s = d / (t11-t00);

#  endif

#else // NO SOFT_CALIBRATION

#  if defined( KAAPI_USE_APPLE ) || defined(KAAPI_USE_IPHONEOS)
  int mib[2];
  size_t len;
  unsigned long mhz;
  mib[0] = CTL_HW;
  mib[1] = HW_CPU_FREQ;
  len =sizeof(mhz);
  sysctl(mib, 2, &mhz, &len, NULL, 0);
  HighResTimer::dtick_per_s = double(mhz); /* it seems that tick freq is about 2/3e9 of the frequency on PowerbOOK*/
#  elif defined(KAAPI_USE_LINUX)
  /* open cpuinfo and scan cpu Mhz: */
  FILE* file = fopen("/proc/cpuinfo","r");
  double mhz = 1.0;
  KAAPI_ASSERT_M(file !=0, "Cannot open /proc/cpuinfo");

  int retval =0;
  char line[256];
  do {
    retval = fscanf(file,"%[^:]:%lf\n",line,&mhz);
    if ( strncmp(line,"cpu MHz", 7) ==0) break; 
  } while (retval != EOF);
  HighResTimer::dtick_per_s = mhz*1e6;
  fclose(file);
#  else
#  error "unknown system / architecture"
#  endif

#endif // END CALIBRATION

  /** Compute latency for call to start/stop */
  HighResTimer::type tick_loop;
  HighResTimer::type t0, t1;
  t0 = HighResTimer::gettick();
  for (int i=0; i<10000; ++i) 
    ;
  t1 = HighResTimer::gettick();
  tick_loop = t1-t0;

  HighResTimer hr;
  t0 = HighResTimer::gettick();
  for (int i=0; i<10000; ++i) 
  { hr.start(); hr.stop(); } 
  t1 = HighResTimer::gettick();

  HighResTimer::latency = (t1-t0 - tick_loop)/10000;
  HighResTimer::inv_dtick_per_s = 1.0 / HighResTimer::dtick_per_s;
  KAAPI_CPPLOG( Init::verboseon, "[atha::HighResTimer::calibrate] Ticks per second  :" << std::setprecision(10) << HighResTimer::dtick_per_s );
  KAAPI_CPPLOG( Init::verboseon, "[atha::HighResTimer::calibrate] Latency start/stop:" << std::setprecision(10) << HighResTimer::latency );
}



// --------------------------------------------------------------------
long Timer::seed() 
{
  struct timeval tp;
  gettimeofday(&tp, 0) ;
  return(tp.tv_usec+tp.tv_sec);
}



// --------------------------------------------------------------------
std::ostream& Timer::print( std::ostream& o ) const
{
  return 
   o << "Time (elapsed highres) s= " << hrrealtime() << std::endl
     << "Time (elapsed) s= " << realtime() << std::endl
     << "Time (cpu) s= " << usertime() << std::endl
     << "Time (sys) s= " << systime()  << std::endl
     << "#Ticks = " << tickcount() << std::endl
     << "#Call  = " << callcount() << std::endl;
}

} // -namespace
