/* KAAPI public interface */
// =========================================================================
// (c) INRIA, projet MOAIS, 2006
// Author: T. Gautier
// Status: ok
//
// =========================================================================
#ifndef _UTILS_STATINFO_H_
#define _UTILS_STATINFO_H_
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>

namespace Util {

// -------------------------------------------------------------------------
/** \brief Compute some statistics about series of values
    \ingroup Misc
    This class acts like as a stream on which series of values may be inserted.
    The object computes basic statistics (count,total,average,standard deviation, min, max)
    of a series of double floating point values.
 */
class Statistic {
public:
  /** Type of value
  */
  typedef double value_type;
  
  /** Default cstor
      All basic statistics are set to 0. Min is set to the maximal value
      of value_type and max set to the minimal value.
   */
  Statistic( ) 
    : _t_min(0.0), 
      _t_max(0.0), 
      _t_total(0.0), 
      _t_var2(0.0), 
      _count(0)
  {}
  
  /** Clear stat
  */
  void clear() 
  { 
    _t_min   = 0.0;
    _t_max   = 0.0;
    _t_total = 0.0;
    _t_var2  = 0.0; 
    _count   = 0; 
  }

  /** Add a new value to the counter
  */
  void insert (const value_type v)
  {
    if (_count ==0) { 
      _t_max    = v;
      _t_min    = v;
      _t_total  = v;
      _t_var2   = v*v;
    } else {
      _t_max    = std::max(_t_max, v);
      _t_min    = std::min(_t_min, v);
      _t_total += v;
      _t_var2  += v*v;
    }
    ++_count;
  }
  
  /** Accumulate two statistic counters
  */
  void merge( const Statistic& v)
  {
    _t_min = std::min( _t_min, v._t_min);
    _t_max = std::max( _t_max, v._t_max);
    _t_total += v._t_total;
    _t_var2  += v._t_var2;
    _count   += v._count;
  }

  /** Return the sum of the values
  */
  value_type total() const 
  { return _t_total; }

  /** Return the average of the values
  */
  value_type average() const 
  { return _t_total / (value_type)_count; }

  /** Return the variance of the values
  */
  value_type variance() const
  { value_type tav = _t_total/(value_type)_count;
    return (_t_var2/_count - tav*tav);
  }

  /** Return the standard deviation of the values
  */
  value_type standard_deviation() const
  { return ::sqrt(variance()); }
  
  /** Return the number of values
  */
  size_t count() const 
  { return (size_t)_count; }

  /** Return the min of the values
  */
  value_type min() const 
  { return _t_min; }

  /** Return the max of the values
  */
  value_type max() const 
  { return _t_max; }

  /** print */
  std::ostream& print( std::ostream& s_out ) const;

public: /* any change should be reported in network_gstream.h */
  value_type  _t_min;
  value_type  _t_max;
  value_type  _t_total;
  value_type  _t_var2;
  value_type  _count;
};



// -------------------------------------------------------------------------
/** \brief Extend basic statistic to compute dynamic values
    \ingroup Misc
    Such object extend the basic statistic object in order to compute
    the velocity of the variation of the serie of dated values.
 */
class TimeStatistic : public Statistic {
public:
  /** Default cstor
      All basic statistics are set to 0. Min is set to the maximal value
      of value_type and max set to the minimal value.
   */
  TimeStatistic( ) 
  { clear(); }
  
  /** Clear stat
  */
  void clear() 
  { 
    Statistic::clear();
    _t_last = 0;
    _v_last = 0;
    _velocity_last[0] = _velocity_last[1] = _velocity_last[2] = 0;
  }


  /** Add a new value to the counter
  */
  void insert (double date, const value_type v)
  {
    double t_wall = date - _t_last;
    _velocity_last[2] = _velocity_last[1];
    _velocity_last[1] = _velocity_last[0];
    _velocity_last[0] = (v - _v_last)/t_wall;
    Statistic::insert(_velocity_last[0]);
    _t_last = date;
    _v_last = v;
  }
  
  /** Accumulate two statistic counter
  */
  void merge( const Statistic& v)
  {
    Statistic::merge(v);
  }

  /** Return the last velocity
  */
  value_type last_velocity() const 
  { return _velocity_last[0]; }

  /** Return the velocity: average on previous values
  */
  value_type velocity() const;

  /** print 
  */
  std::ostream& print( std::ostream& s_out ) const;

public: /* any change should be reported in network_gstream.h */
  value_type  _t_last;
  value_type  _v_last;
  value_type  _velocity_last[3];
};




// -------------------------------------------------------------------------
/** \brief Distribution of values
    \ingroup Misc
    Such object allows to compute the distribution of values with respect
    to a partition of the real domain.
 */
class DistributionStatistic {
public:
  typedef double value_type;
  
  /** Default constructor
  */
  DistributionStatistic( ) 
   : _count_threshold(0), _threshold(0), _count_values(0)
  { }

  /** Constructor
      Construct a DistributionStatistic object that counts number of inserted
      values each interval ] -oo, threshold[0] [, [ threshold[0],threshold[1] [, until
      [threshold[count_threshold], +oo[.
      The array of threshold is copied into the object.
      \param count_thresholds [in] the number of thresholds in thresholds parameter
      \param thresholds [in] the array of thresholds that defines intervals
  */
  DistributionStatistic( uint32_t count_thresholds, const double thresholds[] );
  
  /** Constructor of recopy
  */
  DistributionStatistic(const DistributionStatistic& ds);
  
  /** Destructor
  */
  ~DistributionStatistic();
  
  /** Assignment
  */
  DistributionStatistic& operator=(const DistributionStatistic& ds);
  
  /** Clear distribution
  */
  void clear() 
  { 
    for (unsigned int i=0; i<=_count_threshold; ++i) 
      _count_values[i] = 0;
  }

  /** initialize
      Initialize a DistributionStatistic object that counts number of inserted
      values each interval [ -oo, threshold[0] [, [ threshold[0],threshold[1] [, until
      [threshold[count_threshold], +oo[.
      The array of threshold is copied into the object.
      \param count_threshold [in]  the number of thresholds in thresholds parameter
      \param threshold [in] the array of thresholds that defines intervals
  */
  void initialize( uint32_t count_threshold, const double threshold[] );
  

  /** Insert a new values
  */
  void insert (const value_type v)
  {
    for (unsigned int i=0; i<_count_threshold; ++i)
    {
      if (v < _threshold[i]) {
        ++_count_values[i];
        return;
      }
    }
    ++_count_values[_count_threshold];
  }

  
  /** Return the number of thresholds of the distribution
      \retval return the number of thresholds of the distribution
  */
  uint32_t size() const 
  { return _count_threshold; }

  /** Return the number of values in interval [threshold[i-1], threshold[i][
      The value of threshold[-1] is -oo, the value of threshold[size()] is +oo.
      \param i [in] the index of the interval, 0 <= i <= size()
      \retval return the number of value in [threshold[i-1], threshold[i][
  */
  uint32_t distribution(int i) const 
  { return _count_values[i]; }

  /** Return the value of the i-th threshold
      \param i [in] the index of the threshold value, 0 <= i <= size()
      \retval return the value of threshold[i]
  */
  value_type threshold(int i) const 
  { return _threshold[i]; }

  /** print 
  */
  std::ostream& print( std::ostream& s_out ) const;

  /** Access to the array of thresholds
  */
  const value_type* get_thresholds() const
  { return _threshold; }

  /** Access to the array of thresholds
  */
  value_type* get_thresholds()
  { return _threshold; }

  /** Access to the array of values
  */
  const uint32_t* get_values() const
  { return _count_values; }

  /** Access to the array of values
  */
  uint32_t* get_values()
  { return _count_values; }

protected:
  uint32_t  _count_threshold;
  value_type*  _threshold;
  uint32_t* _count_values;
};


/*
 * inline definitions
 *
 */
} // - namespace Util

inline std::ostream& operator<< ( std::ostream& o, const Util::Statistic& sc )
{ sc.print(o); return o; }
inline std::ostream& operator<< ( std::ostream& o, const Util::TimeStatistic& sc )
{ sc.print(o); return o; }
inline std::ostream& operator<< ( std::ostream& o, const Util::DistributionStatistic& sc )
{ sc.print(o); return o; }

#endif
