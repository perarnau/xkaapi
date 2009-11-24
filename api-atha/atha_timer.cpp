// ==========================================================================
// Copyright(c)'94-97 by Givaro Team
// see the copyright file.
// Authors: T. Gautier
// Based on Givaro's givtimer.C file.  
// review: [3/3/98]
//
//
//
// ==========================================================================
#include "kaapi_impl.h"
#include "atha_init.h"
#include "atha_timer.h"
#include "atha_error.h"
#include "atha_debug.h"
#include <iomanip>

extern "C" {
# include <sys/time.h>
# include <sys/resource.h>
# include <stdio.h>
# include <unistd.h>
#if defined( KAAPI_USE_DARWIN ) || defined(KAAPI_USE_IPHONEOS)
# include <sys/types.h>
# include <sys/sysctl.h>
#endif
}


namespace atha {

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


#if defined(KAAPI_USE_ARCH_X86) && defined(KAAPI_USE_DARWIN)
#  ifndef myrdtsc
#    define myrdtsc(x) \
      __asm__ volatile ("rdtsc" : "=A" (x));
#  endif
#endif

//       __asm__ __volatile__("rdtsc" : "=a" (low), "=d" (high))

// --------------------------------------------------------------------
HighResTimer::type HighResTimer::gettick()
{
#if (defined( KAAPI_USE_APPLE ) && defined(KAAPI_USE_ARCH_PPC) ) ||  defined(KAAPI_USE_ARCH_PPC64)
  register unsigned long t_u;
  register unsigned long t_l;
  asm volatile ("mftbu %0" : "=r" (t_u) );
  asm volatile ("mftb %0" : "=r" (t_l) );
  HighResTimer::type retval = t_u;
  retval <<= 32UL;
  retval |= t_l;
  return retval;
#elif defined( KAAPI_USE_APPLE)&& (defined(KAAPI_USE_ARCH_X86) || defined(KAAPI_USE_ARCH_IA64)) 
  return (HighResTimer::type)WallTimer::gettime();
#elif defined( KAAPI_USE_LINUX)&& (defined(KAAPI_USE_ARCH_X86) || defined(KAAPI_USE_ARCH_IA64)) 
  uint32_t lo,hi;
  __asm__ volatile ( "rdtsc" : "=a" ( lo ) , "=d" ( hi ) );
  return (uint64_t)hi << 32 | lo;
#elif defined(KAAPI_USE_ARCH_ITA)
  register unsigned long ret;
  __asm__ __volatile__ ("mov %0=ar.itc" : "=r"(ret));
  return ret;
#elif defined(KAAPI_USE_IRIX)
#  if defined(HAVE_CLOCK_GETTIME)
  struct timespec tp;
  clock_gettime( CLOCK_SGI_CYCLE, &tp );
  return type(double(tp.tv_sec)  + double(tp.tv_nsec)*1e-9);
#  else
#    error "no time counter for this architecture"
#  endif
#elif defined(KAAPI_USE_IPHONEOS)
  return WallTimer::gettime();
#else
#  error "no time counter for this architecture"
#endif
}


// --------------------------------------------------------------------
kaapi_uint64_t HighResTimer::latency =0;
double HighResTimer::dtick_per_s =0;
double HighResTimer::inv_dtick_per_s =0;




// --------------------------------------------------------------------
//#define USE_SOFT_CALIBRATION

void HighResTimer::calibrate()
{
#if defined(USE_SOFT_CALIBRATION)
  size_t count;
  double t00, t11;
#  if defined( KAAPI_USE_DARWIN ) && defined(KAAPI_USE_ARCH_PPC)
  union utype {
    struct {
      ka_uint32_t t_h;
      ka_uint32_t t_l;
    } u;
    kaapi_uint64_t ll;
  };
  utype hrt0, hrt1, hrt2;
  asm volatile ("mftbu %0" : "=r" (hrt0.u.t_h) );
  asm volatile ("mftb %0" : "=r" (hrt0.u.t_l) );
  asm volatile ("mftbu %0" : "=r" (hrt2.u.t_h) );
  asm volatile ("mftb %0" : "=r" (hrt2.u.t_l) );
#  elif defined(KAAPI_USE_ARCH_X86) || defined(KAAPI_USE_ARCH_IA64)
  HighResTimer::type hrt0, hrt1, hrt2;
  hrt0 = HighResTimer::gettick();
#  else 
#    warning "Cannot make soft calibration of clock"
#  endif
  /* loop 1 */
  for (int i=0; i<128; ++i)
    t00 = WallTimer::gettime();
#  if defined( KAAPI_USE_DARWIN ) && defined(KAAPI_USE_ARCH_PPC)
  asm volatile ("mftbu %0" : "=r" (hrt1.u.t_h) );
  asm volatile ("mftb %0" : "=r" (hrt1.u.t_l) );
#  elif defined(KAAPI_USE_ARCH_X86) || defined(KAAPI_USE_ARCH_IA64)
  hrt1 = HighResTimer::gettick();
#  endif

  /* loop 2 */
  t00 = WallTimer::gettime();
#  if defined( KAAPI_USE_DARWIN ) && defined(KAAPI_USE_ARCH_PPC)
  for (int i=0; i<200; ++i)
#  else
  for (int i=0; i<200; ++i)
#  endif
    for (int j=0; j<100000; ++j)
      count++;
  t11 = WallTimer::gettime();
#  if defined( KAAPI_USE_DARWIN ) && defined(KAAPI_USE_ARCH_PPC)
  asm volatile ("mftbu %0" : "=r" (hrt2.u.t_h) );
  asm volatile ("mftb %0" : "=r" (hrt2.u.t_l) );
#  elif defined(KAAPI_USE_ARCH_X86) || defined(KAAPI_USE_ARCH_IA64)
  hrt2 = HighResTimer::gettick();
#  endif

#  if defined( KAAPI_USE_DARWIN ) && defined(KAAPI_USE_ARCH_PPC)
  /* hrt0: ticks for 128 gettimeofday */
  hrt0.ll = hrt1.ll - hrt0.ll;

  /* hrt1: ticks for the loops + 2 gettimeofday -> hrt1 : loop 2 */
  hrt1.ll = hrt2.ll - hrt1.ll;

  /* ticks for loop */
  double d = double(hrt1.ll)-double(hrt0.ll)/64.0;
  HighResTimer::dtick_per_s = d / (t11-t00);

#  elif defined(KAAPI_USE_ARCH_X86) || defined(KAAPI_USE_ARCH_IA64)
  /* hrt0: ticks for 128 gettimeofday */
  hrt0 = hrt1 - hrt0;

  /* hrt1: ticks for the loops + 2 gettimeofday -> hrt1 : loop 2 */
  hrt1 = hrt2 - hrt1;

  /* ticks for loop */
  double d = double(hrt1)-double(hrt0)/64.0;
  HighResTimer::dtick_per_s = d / (t11-t00);

#  endif

#else // NO SOFT_CALIBRATION

#  if defined( KAAPI_USE_DARWIN ) || defined(KAAPI_USE_IPHONEOS)
  int mib[2];
  size_t len;
  unsigned long mhz;
  mib[0] = CTL_HW;
  mib[1] = HW_CPU_FREQ;
  len =sizeof(mhz);
  sysctl(mib, 2, &mhz, &len, NULL, 0);
  HighResTimer::dtick_per_s = double(mhz); /* it seems that tick freq is about 2/3e9 of the frequency on PowerbOOK*/
#  elif defined( KAAPI_USE_IRIX )
  struct timespec tp; 
  clock_getres( CLOCK_SGI_CYCLE, &tp );
//  std::cout << "SGI res: " << tp.tv_sec << ", " << tp.tv_nsec << std::endl;
  HighResTimer::dtick_per_s = double(tp.tv_sec) + double(tp.tv_nsec)*1e-9;
  HighResTimer::dtick_per_s = 1/dtick_per_s;
#  elif defined(KAAPI_USE_ARCH_X86) || defined(KAAPI_USE_ARCH_IA64)
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
#  warning "unknown system / architecture"
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
