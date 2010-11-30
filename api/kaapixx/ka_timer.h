/* KAAPI public interface */
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
#ifndef _UTILS_TIMER_H_
#define _UTILS_TIMER_H_

#include <iostream>
#include <string>
#include <limits.h>
#if defined(KAAPI_USE_ARCH_IA32) || defined(KAAPI_USE_ARCH_IA64)
#  ifndef rdtsc
#    define rdtsc(low,high) \
       __asm__ __volatile__("rdtsc" : "=a" (low), "=d" (high))
#  endif
#endif
#include <sys/time.h>
#if not defined (_WIN32)
#include <sys/resource.h>
#endif
#include <unistd.h>
#ifdef KAAPI_USE_IRIX
#include <time.h> // clock
#endif

namespace ka {

/** \class WallTimer
    \brief Wall clock timer class
    \ingroup Misc
*/
class WallTimer {
public:
  /** */
  static const std::string& unit();

  typedef double type;

  /** */
  WallTimer( )
   : _delta(0) {}

  /** */
  void clear();

  /** */
  void start();

  /** */
  void stop();

  /** */
  double time() const;

  /** */
  static double gettime();
protected:
  type _delta;
};



/** \class CpuTimer
    \brief CPU timer class
    \ingroup Misc
*/
class CpuTimer {
public:
  /** */
  static const std::string& unit();

  /** */
  typedef double type;

  /** */
  CpuTimer( )
   : _delta(0) {}

  /** */
  void clear();

  /** */
  void start();

  /** */
  void stop();

  /** */
  double time() const;

  /** */
  static double gettime();
protected:
  type _delta;
};



/** \class SysTimer
    \brief SYS timer class
    \ingroup Misc
*/
class SysTimer {
public:
  /** */
  static const std::string& unit();

  /** */
  typedef double type;

  /** */
  SysTimer( )
   : _delta(0) {}

  /** */
  void clear();

  /** */
  void start();

  /** */
  void stop();

  /** */
  double time() const;

  /** */
  static double gettime();
protected:
  type _delta;
};


// -- 
// HighResolution timer class
// --
class HighResTimer {
public:
  /** */
  static const std::string& unit();

  /** */
  typedef uint64_t type;

  /** */
  HighResTimer( )
   : _delta() {}

  /** */
  ~HighResTimer() {}

  /** */
  void clear();
  /** */
  void start();
  /** */
  void stop();

  /** */
  type tick() const
  { return _delta; }

  /** */
  double time() const
  { return double(_delta) * inv_dtick_per_s; }

  /** */
  static type gettick();

  /** */
  static double gettime();

  /** number of tick per second */
  static double dtick_per_s;

  /** 1/ dtick_per_s*/
  static double inv_dtick_per_s;

protected:
  friend class Init;
  static void calibrate();
  static uint64_t latency;
#ifdef KAAPI_USE_IRIX
  static struct timespec tspec;
#endif
  type  _delta;
};



// -- 
// LogicalTimer class:
// --
class LogicalTimer {
public:
  /** */
  static const std::string& unit();

  /** */
  typedef unsigned short type;

  /** */
  LogicalTimer( )
   : _delta(0) {}

  /** */
  void clear();

  /** */
  void start();

  /** */
  void stop();

  /** */
  type tick() const;

  /** */
  double time() const;

protected:
  type _delta;
};


// -- 
// Timer class: contains three objects of the previous type
// --
/** Timer.
Class that defines stop/start timers to get 
timing informations (CPU time, System time, Wall time or wall clock).
The value that stores a timer is defined after a call to stop method
and represents the delay (interval of time) between the calls to start
and stop.
@author T. Gautier
@version $Id: utils_timer.h,v 1.1.1.1 2006/01/11 15:24:34 thierry Exp $
*/
class Timer : protected HighResTimer, 
              protected WallTimer,
              protected CpuTimer,
              protected SysTimer,
              protected LogicalTimer {
public :

  /** Type for timer
   */
  struct type {
    type& operator-=( const type& a );
    type& operator+=( const type& a );
    HighResTimer::type t_hr;
    WallTimer::type    t_r;
    CpuTimer::type     t_u;
    SysTimer::type     t_s;
    LogicalTimer::type t_l;
  };

  /** Clear timers. Reset internal value of the object.
  */ 
  void clear(); 

  /** Start timers. 
  */
  void start();

  /** Stop timers.
     The different times spent between the start and the stop method calls
     is given by methods usertime, systime, realtime.
  */
  void stop();

  /** Return the total amount of second spent in user mode
    (CPU time).
  */
  double usertime() const;

  /** Return the total amount of second spent in system mode
    (System time)
  */
  double systime () const;

  /** Return the total amount of second spent by
    a wall clock (or real time). High Resolution timer
  */
  double hrrealtime () const;

  /** Return the total amount of second spent by
    a wall clock (or real time).
  */
  double realtime () const;

  /** Return the number of clock tick.
  */
  double tickcount () const;

  /** Return the number of start/stop.
  */
  double callcount () const;

  /** 
     Return a seed value to initialize random generator.
  */
  static long seed();

  /**
    Output to a stream. 
    operator<< is also defined.
  */
  std::ostream& print( std::ostream& ) const;

private:
};


/*
 * inline definitions
 */
inline void HighResTimer::clear()
{ _delta =0; }

inline void HighResTimer::start()
{ _delta = HighResTimer::gettick(); }

inline void HighResTimer::stop()
{ _delta = HighResTimer::gettick() - _delta - HighResTimer::latency; }

inline double HighResTimer::gettime()
{ return double(HighResTimer::gettick()) * inv_dtick_per_s; }

inline void WallTimer::clear()
{ _delta =0; }

inline double WallTimer::gettime()
{ 
   struct timeval tmp2 ;
   gettimeofday (&tmp2, 0) ;

   // real time 
   return (double) tmp2.tv_sec + ((double) tmp2.tv_usec) * 1e-6;
}

inline void WallTimer::start()
{ 
  _delta = WallTimer::gettime();
}

inline void WallTimer::stop()
{ 
  _delta = WallTimer::gettime() - _delta;
}

inline double WallTimer::time() const
{ 
  return _delta;
}


inline void CpuTimer::clear()
{ _delta =0; }

inline double CpuTimer::gettime()
{
#if defined (_WIN32)
  FILETIME creationtime;
  FILETIME exittime;
  FILETIME kerneltime;
  FILETIME usertime;

  GetProcessTimes(GetCurrentProcess(), &creationtime, &exittime, &kerneltime, &usertime);
  return (double)((((ULONGLONG) usertime.dwHighDateTime) << 32) + usertime.dwLowDateTime)/10000000;

#else
   struct rusage tmp;
   getrusage (RUSAGE_SELF, &tmp) ;
   return double(tmp.ru_utime.tv_sec) + double(tmp.ru_utime.tv_usec) * 1e-6;
#endif
   // user time
}

inline void CpuTimer::start()
{
  _delta = CpuTimer::gettime();
}

inline void CpuTimer::stop()
{
  _delta = CpuTimer::gettime() - _delta;
}

inline double CpuTimer::time() const
{
  return _delta;
}


inline void SysTimer::clear()
{ _delta =0; }

inline double SysTimer::gettime()
{
#if defined (_WIN32)
  FILETIME creationtime;
  FILETIME exittime;
  FILETIME kerneltime;
  FILETIME usertime;

  GetProcessTimes(GetCurrentProcess(), &creationtime, &exittime, &kerneltime, &usertime);
  return (double)((((ULONGLONG) kerneltime.dwHighDateTime) << 32) + kerneltime.dwLowDateTime)/10000000;

#else
   struct rusage tmp;
   getrusage (RUSAGE_SELF, &tmp) ;
   // user time
   return double(tmp.ru_stime.tv_sec) + double(tmp.ru_stime.tv_usec) * 1e-6;
#endif
}

inline void SysTimer::start()
{
  _delta = SysTimer::gettime();
}

inline void SysTimer::stop()
{
  _delta = SysTimer::gettime() - _delta;
}

inline double SysTimer::time() const
{
  return _delta;
}

 
inline void LogicalTimer::clear()
{ _delta =0; }

inline void LogicalTimer::start()
{
  _delta =0;
}

inline void LogicalTimer::stop()
{
  _delta =1;
}

inline LogicalTimer::type LogicalTimer::tick() const
{
  return _delta;
}

inline double LogicalTimer::time() const
{
  return double(_delta);
}

inline Timer::type& Timer::type::operator-=( const Timer::type& a )
{
  t_hr -= a.t_hr;
  t_r -= a.t_r;
  t_u -= a.t_u;
  t_s -= a.t_s;
  t_l -= a.t_l;
  return *this;
}

inline Timer::type& Timer::type::operator+=( const Timer::type& a )
{
  t_hr += a.t_hr;
  t_r += a.t_r;
  t_u += a.t_u;
  t_s += a.t_s;
  t_l += a.t_l;
  return *this;
}


inline void Timer::clear()
{ 
  HighResTimer::clear();
  WallTimer::clear();
  CpuTimer::clear();
  SysTimer::clear();
  LogicalTimer::clear();
}

inline void Timer::start()
{
  HighResTimer::start();
  WallTimer::start();
  CpuTimer::start();
  SysTimer::start();
  LogicalTimer::start();
}

inline void Timer::stop()
{
  HighResTimer::stop();
  WallTimer::stop();
  CpuTimer::stop();
  SysTimer::stop();
  LogicalTimer::stop();
}

inline double Timer::usertime() const 
{ return CpuTimer::time(); }

inline double Timer::systime () const 
{ return SysTimer::time(); }

inline double Timer::hrrealtime () const 
{ return HighResTimer::time(); }

inline double Timer::realtime () const 
{ return WallTimer::time(); }

inline double Timer::tickcount () const 
{ return HighResTimer::tick(); }

inline double Timer::callcount () const 
{ return LogicalTimer::time(); }

} // - namespace

inline std::ostream& operator<<( std::ostream& o, const ka::Timer& T)
{ return T.print(o);}


#endif 
