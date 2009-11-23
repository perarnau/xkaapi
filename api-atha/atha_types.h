/* KAAPI public interface */
// =========================================================================
// (c) INRIA, projet MOAIS, 2006-2009
// Author: T. Gautier, X. Besseron
// Status: ok
//
//
// =========================================================================
#ifndef _ATHA_TYPES_H_
#define _ATHA_TYPES_H_

#include "kaapi.h"
#include "atha_error.h"
#include "atha_debug.h"
#include <typeinfo>
#include <iosfwd>
#include <sys/uio.h>
#include <arpa/inet.h>

#if defined(KAAPI_USE_LINUX)
#include <netinet/in.h>
#endif

namespace atha {

//@{

/** Return an system wide identifier of the type of an expression
*/
#define kaapi_get_swid(EXPR) kaapi_hash_value(typeid(EXPR).name())
//@}


// -------------------------------------------------------------------------
/** Compatibility
*/
typedef kaapi_uint32_t GlobalId;


// -------------------------------------------------------------------------
/** \name Pointer
    \author T. Gautier 
    \brief Type for a global pointer
    \ingroup Misc
    
    Util::Pointer is the type to represent a generic pointer into one of
    the address space of a set of process on differents architecture. 
    The size in bits of the Util::Pointer is at least enough to represent any 
    of the LocalPointer of the architecture.
    The current implementation is to represent any pointer as a 64 bits long
    integer. Thus a remote pointer is close to an 64 bits integer.
*/
class Pointer {
public:
  /** Default constructor 
  */
  Pointer() : ptr(0) {}

  /** Constructor from an 64 bits unsigned integer
  */
  Pointer(kaapi_uint64_t v) : ptr(v) {}

  /** return 64 bits unsigned integer
  */
  operator kaapi_uint64_t() const { return ptr; }

  /** Test operators 
  */
  //@{
  /** Return true if remote pointer is equal to an integer
  */
  bool operator==(const unsigned long long v) const
  { return (ptr == v); }
  
  /** Return true if remote pointer is not equal to an integer
  */
  bool operator!=(const unsigned long long v) const
  { return (ptr != v); }

  /** Return true if remote pointer is equal to an integer
  */
  bool operator==(const unsigned long v) const
  { return (ptr == v); }
  
  /** Return true if remote pointer is not equal to an integer
  */
  bool operator!=(const unsigned long v) const
  { return (ptr != v); }

  /** Return true if remote pointer is equal to an integer
  */
  bool operator==(const unsigned int v) const
  { return (ptr == v); }
  
  /** Return true if remote pointer is not equal to an integer
  */
  bool operator!=(const unsigned int v) const
  { return (ptr != v); }

  /** Return true if remote pointer is equal to a local pointer
  */
  bool operator==(const void* lp) const
  { return (ptr == (kaapi_uintptr_t)lp); }
  
  /** Return true if remote pointer is not equal to a local pointer
  */
  bool operator!=(const void* lp) const
  { return (ptr != (kaapi_uintptr_t)lp); }
  
  /** Return true if two remote pointers are equals.
  */
  bool operator==(const Pointer& rp) const
  { return (ptr == rp.ptr); }
  
  /** Return true if two remote pointers are not equals.
  */
  bool operator!=(const Pointer& rp) const
  { return (ptr != rp.ptr); }

  /** Return true if two remote pointers are equals.
  */
  bool operator<(const Pointer& rp) const
  { return ptr < rp.ptr; }
  //@}
  
  
  /** Arithmetic operators
  */
  //@{
  /** 
  */
  Pointer& operator+=( const unsigned long long v )
  { ptr += v; return *this; }
  Pointer& operator+=( const long long v )
  { ptr += v; return *this; }
  Pointer& operator+=( const unsigned long v )
  { ptr += v; return *this; }
  Pointer& operator+=( const long v )
  { ptr += v; return *this; }
  Pointer& operator+=( const unsigned int v )
  { ptr += v; return *this; }
  Pointer& operator+=( const int v )
  { ptr += v; return *this; }

  /** 
  */
  Pointer operator+( const unsigned long long v ) const
  { Pointer retval = ptr + v; return retval; }
  Pointer operator+( const long long v ) const
  { Pointer retval = ptr + v; return retval; }
  Pointer operator+( const unsigned long v ) const
  { Pointer retval = ptr + v; return retval; }
  Pointer operator+( const long v ) const
  { Pointer retval = ptr + v; return retval; }
  Pointer operator+( const unsigned int v ) const
  { Pointer retval = ptr + v; return retval; }
  Pointer operator+( const int v ) const
  { Pointer retval = ptr + v; return retval; }

  /** 
  */
  Pointer& operator-=( const unsigned long long v )
  { ptr -= v; return *this; }
  Pointer& operator-=( const long long v )
  { ptr -= v; return *this; }
  Pointer& operator-=( const unsigned long v )
  { ptr -= v; return *this; }
  Pointer& operator-=( const long v )
  { ptr -= v; return *this; }
  Pointer& operator-=( const unsigned int v )
  { ptr -= v; return *this; }
  Pointer& operator-=( const int v )
  { ptr -= v; return *this; }

  /** 
  */
  Pointer operator-( const unsigned long long v ) const
  { Pointer retval = ptr - v; return retval; }
  Pointer operator-( const long long v ) const
  { Pointer retval = ptr - v; return retval; }
  Pointer operator-( const unsigned long v ) const
  { Pointer retval = ptr - v; return retval; }
  Pointer operator-( const long v ) const
  { Pointer retval = ptr - v; return retval; }
  Pointer operator-( const unsigned int v ) const
  { Pointer retval = ptr - v; return retval; }
  Pointer operator-( const int v ) const
  { Pointer retval = ptr - v; return retval; }

  /** 
  */
  unsigned long long operator-( const Pointer& p ) const
  { unsigned long long retval = ptr - p.ptr; return retval; }
  //@}
  
  /** 
  */
  Pointer& operator=(void* l)
  { 
    ptr = (kaapi_uintptr_t)l;
    return *this; 
  }

  /** 
  */
  void from_local( void* l)
  {
    ptr = (kaapi_uintptr_t)l;
  }

  /** Return local pointer */
  void* to_local()
  {
    return (void*)(kaapi_uintptr_t)ptr;
  }

  /** Return local pointer */
  const void* to_local() const
  { 
    return (void*)(kaapi_uintptr_t)ptr; 
  }  
public:
  kaapi_uint64_t ptr;
};

inline Pointer operator+( const unsigned long long v, const Pointer& p )
{ Pointer retval = v + p.ptr; return retval; }
inline Pointer operator+( const long long v, const Pointer& p )
{ Pointer retval = v + p.ptr; return retval; }
inline Pointer operator+( const unsigned long v, const Pointer& p )
{ Pointer retval = v + p.ptr; return retval; }
inline Pointer operator+( const long v, const Pointer& p )
{ Pointer retval = v + p.ptr; return retval; }
inline Pointer operator+( const unsigned int v, const Pointer& p )
{ Pointer retval = v + p.ptr; return retval; }
inline Pointer operator+( const int v, const Pointer& p )
{ Pointer retval = v + p.ptr; return retval; }

inline Pointer operator-( const unsigned long long v, const Pointer& p )
{ Pointer retval = v - p.ptr; return retval; }
inline Pointer operator-( const long long v, const Pointer& p )
{ Pointer retval = v - p.ptr; return retval; }
inline Pointer operator-( const unsigned long v, const Pointer& p )
{ Pointer retval = v - p.ptr; return retval; }
inline Pointer operator-( const long v, const Pointer& p )
{ Pointer retval = v - p.ptr; return retval; }
inline Pointer operator-( const unsigned int v, const Pointer& p )
{ Pointer retval = v - p.ptr; return retval; }
inline Pointer operator-( const int v, const Pointer& p )
{ Pointer retval = v - p.ptr; return retval; }


// -------------------------------------------------------------------------
/** \name get_hostname
    \brief Return the canonical name of the machine
    \ingroup Util
*/
extern const std::string& get_hostname();


// -------------------------------------------------------------------------
// --------- Conversion of long double -------------------------------------

#define MANTISSA_MAXSIZE 14 // bytes ( 14 minimum for ieee quadruple)
typedef kaapi_int32_t exptype;

enum LongDoubleFormat {
  IEEE_DOUBLE      = 0,
  IEEE_EXTENDED_12 = 1,
  IEEE_EXTENDED_16 = 2,
  IEEE_QUADRUPLE   = 3,
  PPC_QUADWORD     = 4,
  NB_FORMAT        = 5 // leave in last position = number of format
};


class Sign {
  friend class LongDouble;
  private:
    bool _positive; 
    
    void get(LongDoubleFormat f, unsigned char *r);
    void set(LongDoubleFormat f, unsigned char *r);
};


class Exponent {
  friend class LongDouble;
  private:
    exptype _exp;  
    
    void get(LongDoubleFormat f, unsigned char *r);
    void set(LongDoubleFormat f, unsigned char *r);
};


class Mantissa {
  friend class LongDouble;
  private:
    unsigned char _mant[MANTISSA_MAXSIZE];
    
    void get(LongDoubleFormat f, unsigned char *r);
    void set(LongDoubleFormat f, unsigned char *r);
    void clear();
    void shift_left();
    void shift_right(bool b);
};

enum LongDoubleType {
  LD_NORMAL,
  LD_INFINITY,
  LD_NAN,
  LD_ZERO
};

class LongDouble {
  private:
    LongDoubleType _type;
    Sign _sign;
    Exponent _exp;
    Mantissa _mant;

    static LongDoubleFormat local_format;
    static LongDoubleFormat compute_local_format();
    
    void get(LongDoubleFormat f, unsigned char *r);
    void set(LongDoubleFormat f, unsigned char *r);

    void debug(const char *s);
  
  public:
    static LongDoubleFormat get_local_format();
    static size_t get_size(LongDoubleFormat f);
    
    static void conversion(LongDoubleFormat from, LongDoubleFormat to, unsigned char *r);
    static void conversion_from_local_to(LongDoubleFormat to, unsigned char *r);
    static void conversion_to_local_from(LongDoubleFormat to, unsigned char *r);
};

inline LongDoubleFormat LongDouble::get_local_format()
{
  return local_format;
}


// ----------------------------------------------------------------------
/** \brief Defines the characteristics of the processor usefull for heterogeneous data transfer.
    \ingroup Serialization
*/
class Architecture {
public:
  kaapi_uint8_t _archi;

  Architecture(kaapi_uint8_t a = 0) : _archi(a) {};

  operator kaapi_uint8_t() const {
    return _archi;
  }

  kaapi_uint8_t operator()() const {
    return _archi;
  }
  
  void set_local();
  
  bool is_bigendian();

  int get_sizeof_long();

  int get_sizeof_bool();

/* TODO
  LongDoubleFormat get_formatof_longdouble();
*/
};


// --------------------------------------------------------------------
/** Interface to alloctor used by allocate method
*/
class InterfaceAllocator {
public:
  virtual ~InterfaceAllocator();
  virtual void* allocate( size_t ) =0;
};

/** Interface to dealloctor used by deallocate method
*/
class InterfaceDeallocator {
public:
  virtual ~InterfaceDeallocator();
  virtual void deallocate( void* ) =0;
};


// --------------------------------------------------------------------
/*   
 *
 */
inline bool Architecture::is_bigendian()
{ return ((_archi & 0x01) == 0x01); }

inline int Architecture::get_sizeof_long()
{
  if ((_archi & 0x02) == 0x02) {
    return 8;
  } else {
    return 4;
  }
}

inline int Architecture::get_sizeof_bool()
{
  if ((_archi & 0x04) == 0x04) {
    return 4;
  } else {
    return 1;
  }
}
    
/* TODO
inline LongDoubleFormat Architecture::get_formatof_longdouble()
{ return (LongDoubleFormat)(_archi >> 3); }
*/

}
#endif
