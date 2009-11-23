/* KAAPI internal interface */
// =========================================================================
// (c) INRIA, projet MOAIS, 2006
// Author: T. Gautier, X. Besseron
// This file defines Format object in order to be able to send
// message between heterogeneous architectures.
//
// =========================================================================
#ifndef _ATHA_FORMAT_H_
#define _ATHA_FORMAT_H_

#include "atha_error.h"
#include "atha_types.h"
#include "atha_stream.h"
#include <vector>

namespace atha {

// --------------------------------------------------------------------
/* About data types between heterogeneous computers:
 *
 * Recognized data types are:
 * boolean,
 * [unsigned] char, [unsigned] short, [unsigned] int, [unsigned] long, [unsigned] long long,
 * float, double, long double.
 *
 * Conversion between little-endian and big-endian are made.
 * (It uses WORDS_BIGENDIAN in kaapi_config.h to determine endianess)
 *
 * For boolean:
 * The least significant byte is kept.
 * (Booleans are 4-bytes long on G5 with g++3)
 *
 * For [unsigned] char (1),
 *     [unsigned] short (2),
 *     [unsigned] int (4),
 *     [unsigned] long long (8),
 *     float (4 bytes with ieee single format),
 *     double (8 bytes with ieee double format):
 * the size and the format is the same everywhere.
 * 
 * For [unsigned] long (4 or 8):
 *     the guarantee precision is the lowest precision of all the computers:
 *     8 bytes if all computers use 8-bytes long,
 *     4 bytes else.
 *
 * For long double:
 *   the recognized long double formats are:
 *     - IEEE_DOUBLE                                  --> ppc with g++3
 *     - IEEE_EXTENDED (with sizeof(long double)=12)  --> i686 and compatible
 *     - IEEE_EXTENDED (with sizeof(long double)=16)  --> itanium
 *     - IEEE_QUADRUPLE                               --> sparc and ??? (not fully tested)
 *     - PPC_QUADWORD                                 --> ppc with g++4
 * (about ieee formats, see http://babbage.cs.qc.edu/courses/cs341/IEEE-754references.html)
 * 
 *   the guarantee precision is the lowest precision of all the computers:
 *     - with PPC_QUADWORD, garantee precision is a ieee double precision.
 *   so
 *     min, max, epsilon values of the lowest precision computer are preserved.
 *     nan are preserved.
 *
 *   if possible
 *      conversion between normalized and denormalized is made
 *   else
 *      +or- "too big value"     -->   +or- infinity
 *      +or- "too small value"   -->   +or- 0
 *
 *
 * Limitation:
 *   - conversion from PPC_QUADWORD transform sNAN in qNAN
 * 
 * 
 */



// --------------------------------------------------------------------
/** \brief Definition of the format of a data type
    \ingroup Serialization
 A Format is the definition of the format for a data structure,
 currently only XDR like format is supported. MPI like format will
 be supported as a compilation of XDR like format.
*/
class Format {
public:
  // - signature for arguments of the following methods
  //@{
  /** type of a format identifier
  */
  typedef kaapi_uint32_t Id;
  //@}

  /** cstor with parameter to define a format.
    The caller should give an unique identifier for the type, as it could
    be return by kaapi_get_swid() macro.
    @param fmid an unique system wide identifier
    @param sz the size of the object class the format represents
    @param name the name of the format
    @see kaapi_get_swid
  */
  Format( Id fmid, size_t sz, const std::string& name);

  /** Virtual dstor
  */
  virtual ~Format();

  /** return the format with identifier 'id'
  */
  static Format* get_format( Id id );

  /** return the identifier of the format
  */
  Id get_id( ) const
  { return _id; }

  /** Return the size in bytes of a C++ data with format
  */
  virtual size_t get_size( ) const
  { return _size; }

  /** Description of the output
  */
  virtual void write( OStream& o, const void* d, size_t count) const =0;

  /** Description of the input
  */
  virtual void read( IStream& o, void* d, size_t count) const =0;

  /** Should be equivalent to malloc(sizeof(Class))
  */
  virtual void* allocate(InterfaceAllocator* a, size_t count) const =0;
  void* allocate(size_t count) const
  { return allocate(0, count ); }


  /** Should be equivalent to free(sizeof(Class))
  */
  virtual void  deallocate(InterfaceDeallocator* a, void* d, size_t count) const =0;
  void  deallocate(void* d, size_t count) const
  { deallocate(0, d, count); }

  /** Should be equivalent to new(d) Class
  */
  virtual void  cstor(InterfaceAllocator* a, void* d, size_t count) const =0;
  void  cstor(void* d, size_t count) const
  { cstor(0, d, count); }

  /** Should be equivalent to new(d) Class(src)
  */
  virtual void  cstor(InterfaceAllocator* a, void* d, const void* src, size_t count) const =0;
  void  cstor(void* d, const void* src, size_t count) const
  { cstor( 0, d, src, count); }

  /** Should be equivalent to d->~Class()
  */
  virtual void  dstor(InterfaceDeallocator* a, void* d, size_t count) const =0;
  void  dstor(void* d, size_t count) const
  { dstor(0, d, count); }

  /** Should be equivalent to d = new Class
  */
  virtual void* create(InterfaceAllocator* a, size_t count) const =0;
  void* create(size_t count) const
  { return create(0, count); }

  /** Should be equivalent to delete d
  */
  virtual void  destroy(InterfaceDeallocator* a, void* d, size_t count) const =0;
  void  destroy(void* d, size_t count) const
  { destroy(0, d, count); }

  /** Should be equivalent to a[i] = b[i] for i=0,count
  */
  virtual void  copy(InterfaceAllocator* a, void* dest, const void* src, size_t count) const =0;
  void  copy(void* dest, const void* src, size_t count) const
  { copy(0, dest, src, count); }

  /** Display a formatted text output of the data t
  */
  virtual void print( std::ostream& o, const void* t ) const;

  /** Set the format object to the attr of this closure
  */
  void set_attr_format(const Format* f);

  /** Return the format object to the attr of this closure
  */
  const Format* get_attr_format( ) const;

  /** Return a pointer to the attribut
   */
  void* get_attr( void* base ) const;

  // - Set the size in bytes of a C++ data with format
  void set_size( size_t s )
  { _size =s; }

  /** Set a name
   */
  void set_name( const std::string& n );

  /** Get a name
  */
  const std::string& get_name() const;

  /** Return true if it is continuous type
  */
  bool is_continuous() const;

  /** Set the is_continuous flag to true
  */
  void set_continuous() const;

  /** Compile a format
  */
  void compile() const;

  /** Compile all registered not yet compiled formats
  */
  static void compile_all();

protected:
  static void init_format();
  friend class Init;
  friend class StreamFormat;

protected:
  mutable Id    _id;
  std::string   _name;
  Format*       _prev_init;
  size_t        _size;
  const Format* _attr_fmt;
  size_t        _attr_offset;

  mutable bool  _iscompiled;
  mutable bool  _iscontiguous;

private:
  static std::map<Id,Format*> _all_fm; // map from Id to format object
  static Format* _base_fmt;            // base format to initialize
private:
  /** Cannot be used
  */
  Format( const Format& fmt );

  /** Cannot be used
  */
  Format& operator=( const Format& fmt );

#ifdef KAAPI_USE_XDR
protected:
  mutable size_t _std_size; // standard size used to send on the network
                            // 0 if not a base type
  mutable bool _isreal;     // true if it is a float, a double or a long double
  mutable bool _isunsigned; // true if it is unsigned

public:
  inline size_t get_std_size() const{
    return _std_size;
  }

  inline void set_std_size(size_t s) const {
    _std_size = s;
  }

  inline bool is_real() const {
    return _isreal;
  }

  inline void set_real(bool b) const {
    _isreal = b;
  }

  inline bool is_unsigned() const {
    return _isunsigned;
  }

  inline void set_unsigned(bool b) const {
    _isunsigned = b;
  }

#endif
};


// --------------------------------------------------------------------
/** \name FormatOperationCumul
    \brief Format for a cumul operation
    \ingroup Serialization
*/
class FormatOperationCumul : public Format {
public:
  /** Type of the cumul function: dest += src
  */
  typedef void (*CumulFunction)(void* dest, const void* src);
  
  /** cstor with parameter to define a format.
    The caller should give an unique identifier for the type, as it could
    be return by kaapi_get_swid() macro.
    @param fmid an unique system wide identifier
    @param sz the size of the object class the format represents
    @param name the name of the format
    @see kaapi_get_swid
  */
  FormatOperationCumul( Id fmid, size_t sz, const std::string& name);

  /** Description of the cumul
  */
  virtual void cumul( void* d, const void* src ) const =0;

};


// --------------------------------------------------------------------
/** \name WrapperFormat<T>
    \brief Wrapper of standard operations over class T to a format
    \ingroup Serialization
*/
template<class T>
class WrapperFormat : public Format {
public:

  void write( OStream& s, const void* val, size_t count ) const;

  void read( IStream& s, void* val, size_t count ) const;

  void* allocate( InterfaceAllocator* a, size_t count ) const
  { return a == 0 ? _kaapi_malloc(count*sizeof(T)) : a->allocate(count*sizeof(T)) ; }
  void* allocate( size_t count ) const
  { return _kaapi_malloc(count*sizeof(T)); }

  void deallocate( InterfaceDeallocator* a, void* d, size_t /*count*/  ) const
  { if (a ==0) _kaapi_free(d); else a->deallocate(d); }
  void deallocate( void* d, size_t /*count*/  ) const
  { _kaapi_free(d); }

  void cstor( InterfaceAllocator* /*a*/, void* d, size_t count ) const
  { T* dd = (T*)d;
    for (size_t i=0; i<count; ++i)
      new (&dd[i]) T;
  }
  void cstor( void* d, size_t count ) const
  { T* dd = (T*)d;
    for (size_t i=0; i<count; ++i)
      new (&dd[i]) T;
  }

  void cstor( InterfaceAllocator* /*a*/, void* d, const void* s, size_t count ) const
  { T* dd = (T*)d; const T* ss = (const T*)s;
    for (size_t i=0; i<count; ++i)
      new (&dd[i]) T(ss[i]);
  }
  void cstor(  void* d, const void* s, size_t count ) const
  { T* dd = (T*)d; const T* ss = (const T*)s;
    for (size_t i=0; i<count; ++i)
      new (&dd[i]) T(ss[i]);
  }

  void dstor( InterfaceDeallocator* /*a*/, void* d, size_t count ) const
  { T* dd = (T*)d;
    for (size_t i=0; i<count; ++i)
      dd[i].~T();
  }
  void dstor( void* d, size_t count ) const
  { T* dd = (T*)d;
    for (size_t i=0; i<count; ++i)
      dd[i].~T();
  }

  void* create( InterfaceAllocator* a, size_t count ) const
  { T*d = (T*)(a == 0 ? _kaapi_malloc(count*sizeof(T)) : a->allocate(count*sizeof(T)));
    for (size_t i=0; i<count; ++i)
      new (&d[i]) T;
    return d;
  }
  void* create( size_t count ) const
  { T*d = (T*)_kaapi_malloc(count*sizeof(T));
    for (size_t i=0; i<count; ++i)
      new (&d[i]) T;
    return d;
  }

  void destroy( InterfaceDeallocator* a, void* d, size_t count ) const
  { T* dd = (T*)d; 
    for (size_t i=0; i<count; ++i) 
      dd[i].~T(); 
    if (a ==0) _kaapi_free(d); else a->deallocate(d); 
  }
  void destroy( void* d, size_t count ) const
  { T* dd = (T*)d; 
    for (size_t i=0; i<count; ++i) 
      dd[i].~T(); 
    _kaapi_free(d); 
  }

  void copy( InterfaceAllocator* , void* dest, const void* src, size_t count ) const
  { T* ddest = (T*)dest;
    const T* dsrc = (const T*)src;
    for (size_t i=0; i<count; ++i)
      ddest[i] = dsrc[i];
  }
  void copy( void* dest, const void* src, size_t count ) const
  { T* ddest = (T*)dest;
    const T* dsrc = (const T*)src;
    for (size_t i=0; i<count; ++i)
      ddest[i] = dsrc[i];
  }

  WrapperFormat()
    : Format( kaapi_get_swid(T), sizeof(T), typeid(T).name() )
  {
  }

  static const WrapperFormat<T> theformat;
  static const WrapperFormat<T>* const format;
  static const WrapperFormat<T>* get_format();
};

template<class T>
const WrapperFormat<T> WrapperFormat<T>::theformat;

template<class T>
const WrapperFormat<T>* const WrapperFormat<T>::format
  = &WrapperFormat<T>::theformat;

template<class T>
inline const WrapperFormat<T>* WrapperFormat<T>::get_format()
{
  return &WrapperFormat<T>::theformat;
}



// --------------------------------------------------------------------
/** \name WrapperFormatOperationCumul<T>
    \brief Wrapper of standard cumul operations over class T 
    \ingroup Serialization
*/
template<class T, class OpCumul>
class WrapperFormatOperationCumul : public FormatOperationCumul {
public:

  void write( OStream& , const void* , size_t ) const
  { }

  void read( IStream& , void* , size_t ) const
  { }

  void* allocate( InterfaceAllocator* a, size_t count ) const
  { return a == 0 ? _kaapi_malloc(count*sizeof(OpCumul)) : a->allocate(count*sizeof(OpCumul)) ; }

  void deallocate( InterfaceDeallocator* a, void* d, size_t ) const
  { if (a ==0) _kaapi_free(d); else a->deallocate(d); }

  void cstor( InterfaceAllocator* , void* d, size_t count ) const
  { OpCumul* dd = (OpCumul*)d;
    for (size_t i=0; i<count; ++i)
      new (&dd[i]) OpCumul;
  }

  void cstor( InterfaceAllocator* , void* d, const void* s, size_t count ) const
  { OpCumul* dd = (OpCumul*)d; const OpCumul* ss = (const OpCumul*)s;
    for (size_t i=0; i<count; ++i)
      new (&dd[i]) OpCumul(ss[i]);
  }

  void dstor( InterfaceDeallocator* , void* d, size_t count ) const
  { OpCumul* dd = (OpCumul*)d;
    for (size_t i=0; i<count; ++i)
      dd[i].~OpCumul();
  }

  void* create( InterfaceAllocator* a, size_t count ) const
  { OpCumul* d = (OpCumul*)(a == 0 ? _kaapi_malloc(count*sizeof(OpCumul)) : a->allocate(count*sizeof(OpCumul)));
    for (size_t i=0; i<count; ++i)
      new (&d[i]) OpCumul;
    return d;
  }

  void destroy( InterfaceDeallocator* a, void* d, size_t count ) const
  { OpCumul* dd = (OpCumul*)d; 
    for (size_t i=0; i<count; ++i) 
      dd[i].~OpCumul(); 
    if (a ==0) _kaapi_free(d); else a->deallocate(d); 
  }

  void copy( InterfaceAllocator* , void* dest, const void* src, size_t count ) const
  { OpCumul* ddest = (OpCumul*)dest;
    const OpCumul* dsrc = (const OpCumul*)src;
    for (size_t i=0; i<count; ++i)
      ddest[i] = dsrc[i];
  }

  template<class X>
  static void CallCumulOperation( void (OpCumul::* /*method*/)(T&, const X&), void* dest, const void* src )
  {
    static OpCumul theop;
    theop( *(T*)dest, *(const X*)src );
  }
  template<class X>
  static void CallCumulOperation( void (OpCumul::* /*method*/)(T&, const X&) const, void* dest, const void* src )
  {
    static OpCumul theop;
    theop( *(T*)dest, *(const X*)src );
  }
  void cumul( void* d, const void* src ) const
  { 
    CallCumulOperation( &OpCumul::operator(), d, src );
  }

  WrapperFormatOperationCumul()
   : FormatOperationCumul( kaapi_get_swid(OpCumul), sizeof(OpCumul), typeid(OpCumul).name() )
  {
  }

  static const WrapperFormatOperationCumul<T,OpCumul> theformat;
  static const WrapperFormatOperationCumul<T,OpCumul>* const format;
  static const WrapperFormatOperationCumul<T,OpCumul>* get_format();
};

template<class T, class OpCumul>
const WrapperFormatOperationCumul<T,OpCumul> WrapperFormatOperationCumul<T,OpCumul>::theformat;

template<class T, class OpCumul>
const WrapperFormatOperationCumul<T,OpCumul>* const WrapperFormatOperationCumul<T,OpCumul>::format
  = &WrapperFormatOperationCumul<T,OpCumul>::theformat;

template<class T, class OpCumul>
inline const WrapperFormatOperationCumul<T,OpCumul>* WrapperFormatOperationCumul<T,OpCumul>::get_format()
{
  return &WrapperFormatOperationCumul<T,OpCumul>::theformat;
}


// --------------------------------------------------------------------
#ifndef KAAPI_USE_IRIX
#define KAAPI_SPECIALIZED_FORMAT(TYPE)\
template<>\
class WrapperFormat<TYPE> : virtual public Format {\
public:\
  void write( OStream& s, const void* val, size_t count ) const\
  { s.write( format, OStream::DA, val, count ); }\
  void read( IStream& s, void* val, size_t count ) const\
  { s.read( format, OStream::DA, val, count ); }\
  void* allocate( InterfaceAllocator* a, size_t count ) const\
  { return a == 0 ? _kaapi_malloc(count*sizeof(TYPE)) : a->allocate(count*sizeof(TYPE)) ; }\
  void* allocate( size_t count ) const\
  { return _kaapi_malloc(count*sizeof(TYPE)); }\
  void deallocate( InterfaceDeallocator* a, void* d, size_t ) const\
  { if (a ==0) _kaapi_free(d); else a->deallocate(d); }\
  void deallocate( void* d, size_t ) const\
  { _kaapi_free(d); }\
  void cstor(InterfaceAllocator* ,  void* d, size_t count ) const\
  { TYPE*dd=(TYPE*)d; for (size_t i=0; i<count; ++i) new (&dd[i]) TYPE(0); }\
  void cstor(void* d, size_t count ) const\
  { TYPE*dd=(TYPE*)d; for (size_t i=0; i<count; ++i) new (&dd[i]) TYPE(0); }\
  void cstor( InterfaceAllocator* , void* d, const void* s, size_t count ) const\
  { TYPE* dd = (TYPE*)d; const TYPE* ss = (TYPE*)s;\
    for (size_t i=0; i<count; ++i) \
      new (&dd[i]) TYPE(ss[i]);\
  }\
  void cstor( void* d, const void* s, size_t count ) const\
  { TYPE* dd = (TYPE*)d; const TYPE* ss = (TYPE*)s;\
    for (size_t i=0; i<count; ++i) \
      new (&dd[i]) TYPE(ss[i]);\
  }\
  void dstor( InterfaceDeallocator* , void*, size_t ) const\
  { }\
  void dstor( void*, size_t ) const\
  { }\
  void* create( InterfaceAllocator* a, size_t count ) const\
  { TYPE*d=(TYPE*)(a == 0 ? _kaapi_malloc(count*sizeof(TYPE)) : a->allocate(count*sizeof(TYPE)));\
    for (size_t i=0; i<count; ++i) new (&d[i]) TYPE(0); return d; }\
  void* create( size_t count ) const\
  { TYPE*d=(TYPE*)_kaapi_malloc(count*sizeof(TYPE));\
    for (size_t i=0; i<count; ++i) new (&d[i]) TYPE(0); return d; }\
  void destroy( InterfaceDeallocator* a, void* d, size_t ) const\
  { if (a ==0) _kaapi_free(d); else a->deallocate(d); }\
  void destroy( void* d, size_t ) const\
  { _kaapi_free(d); }\
  void copy( InterfaceAllocator* , void* dest, const void* src, size_t count ) const\
  { TYPE* ddest = (TYPE*)dest; \
    const TYPE* dsrc = (const TYPE*)src;\
    for (size_t i=0; i<count; ++i) \
      ddest[i] = dsrc[i];\
  }\
  void copy( void* dest, const void* src, size_t count ) const\
  { TYPE* ddest = (TYPE*)dest; \
    const TYPE* dsrc = (const TYPE*)src;\
    for (size_t i=0; i<count; ++i) \
      ddest[i] = dsrc[i];\
  }\
  void print( std::ostream& o, const void* t ) const\
  { o << *(const TYPE*)t; }\
  WrapperFormat(const std::string& name) : Format(kaapi_get_swid(TYPE),sizeof(TYPE),name) {}\
  static const WrapperFormat<TYPE>& theformat;\
  static const WrapperFormat<TYPE>* const format;\
  static const WrapperFormat<TYPE>* get_format()\
  { return format; }\
};

#else
#define KAAPI_SPECIALIZED_FORMAT(TYPE)\
template<>\
class WrapperFormat<TYPE> : virtual public Format {\
public:\
  void write( OStream& s, const void* val, size_t count ) const\
  { s.write( format, OStream::DA, val, count ); }\
  void read( IStream& s, void* val, size_t count ) const\
  { s.read( format, OStream::DA, val, count ); }\
  void* allocate( InterfaceAllocator* a, size_t count ) const\
  { return a == 0 ? _kaapi_malloc(count*sizeof(TYPE)) : a->allocate(count*sizeof(TYPE)) ; }\
  void* allocate( size_t count ) const\
  { return _kaapi_malloc(count*sizeof(TYPE)); }\
  void deallocate( InterfaceDeallocator* a, void* d, size_t ) const\
  { if (a ==0) _kaapi_free(d); else a->deallocate(d); }\
  void deallocate( void* d, size_t ) const\
  { _kaapi_free(d); }\
  void cstor(InterfaceAllocator* ,  void* d, size_t count ) const\
  { TYPE*dd=(TYPE*)d; for (size_t i=0; i<count; ++i) new (&dd[i]) TYPE(0); }\
  void cstor(void* d, size_t count ) const\
  { TYPE*dd=(TYPE*)d; for (size_t i=0; i<count; ++i) new (&dd[i]) TYPE(0); }\
  void cstor( InterfaceAllocator* , void* d, const void* s, size_t count ) const\
  { TYPE* dd = (TYPE*)d; const TYPE* ss = (TYPE*)s;\
    for (size_t i=0; i<count; ++i) \
      new (&dd[i]) TYPE(ss[i]);\
  }\
  void cstor( void* d, const void* s, size_t count ) const\
  { TYPE* dd = (TYPE*)d; const TYPE* ss = (TYPE*)s;\
    for (size_t i=0; i<count; ++i) \
      new (&dd[i]) TYPE(ss[i]);\
  }\
  void dstor( InterfaceDeallocator* , void*, size_t ) const\
  { }\
  void dstor( void*, size_t ) const\
  { }\
  void* create( InterfaceAllocator* a, size_t count ) const\
  { TYPE*d=(TYPE*)(a == 0 ? _kaapi_malloc(count*sizeof(TYPE)) : a->allocate(count*sizeof(TYPE)));\
    for (size_t i=0; i<count; ++i) new (&d[i]) TYPE(0); return d; }\
  void* create( size_t count ) const\
  { TYPE*d=(TYPE*)_kaapi_malloc(count*sizeof(TYPE));\
    for (size_t i=0; i<count; ++i) new (&d[i]) TYPE(0); return d; }\
  void destroy( InterfaceDeallocator* a, void* d, size_t ) const\
  { if (a ==0) _kaapi_free(d); else a->deallocate(d); }\
  void destroy( void* d, size_t ) const\
  { _kaapi_free(d); }\
  void copy( InterfaceAllocator* , void* dest, const void* src, size_t count ) const\
  { TYPE* ddest = (TYPE*)dest; \
    const TYPE* dsrc = (const TYPE*)src;\
    for (size_t i=0; i<count; ++i) \
      ddest[i] = dsrc[i];\
  }\
  void copy( void* dest, const void* src, size_t count ) const\
  { TYPE* ddest = (TYPE*)dest; \
    const TYPE* dsrc = (const TYPE*)src;\
    for (size_t i=0; i<count; ++i) \
      ddest[i] = dsrc[i];\
  }\
  void print( std::ostream& o, const void* t ) const\
  { o << *(const TYPE*)t; }\
  WrapperFormat(const std::string& name) : Format(kaapi_get_swid(TYPE),sizeof(TYPE),name) {}\
  static const WrapperFormat<TYPE>& theformat;\
  static const WrapperFormat<TYPE>* const format;\
  static const WrapperFormat<TYPE>* get_format()\
  { return format; }\
};
#endif

// --------------------------------------------------------------------
#if (__GNUC__ < 4 ) && !defined(KAAPI_USE_DARWIN) && !defined(KAAPI_USE_ARCH_IA64) && !defined(KAAPI_USE_ARCH_ITA)
#define KAAPI_DECL_SPECIALIZED_FORMAT(TYPE,OBJ)\
template<>\
const WrapperFormat<TYPE>& WrapperFormat<TYPE>::theformat = OBJ;\
template<>\
const WrapperFormat<TYPE>* const WrapperFormat<TYPE>::format \
  = &WrapperFormat<TYPE>::theformat;
#else
#define KAAPI_DECL_SPECIALIZED_FORMAT(TYPE,OBJ)\
const WrapperFormat<TYPE>& WrapperFormat<TYPE>::theformat = OBJ;\
const WrapperFormat<TYPE>* const WrapperFormat<TYPE>::format \
  = &WrapperFormat<TYPE>::theformat;
#endif


// --------------------------------------------------------------------
#define KAAPI_DECL_SPEC_FORMAT(T,NN)\
const WrapperFormat<T> NN(\
       KAAPI_VALTOSTR(T)\
     );\
     KAAPI_DECL_SPECIALIZED_FORMAT(T,NN)



// --------------------------------------------------------------------
/* Used to specialized on Type without << and >> for OStream/IStream
   and ostream
*/
#ifndef KAAPI_USE_IRIX
#define KAAPI_SPECIALIZED_VOIDFORMAT(TYPE)\
template<>\
class WrapperFormat<TYPE> : virtual public Format {\
public:\
  void write( OStream& , const void* , size_t ) const\
  { }\
  void read( IStream& , void* , size_t ) const\
  { }\
  void* allocate( InterfaceAllocator* a, size_t count ) const\
  { return a == 0 ? _kaapi_malloc(count*sizeof(TYPE)) : a->allocate(count*sizeof(TYPE)) ; }\
  void* allocate( size_t count ) const\
  { return _kaapi_malloc(count*sizeof(TYPE)); }\
  void deallocate( InterfaceDeallocator* a, void* d, size_t ) const\
  { if (a ==0) _kaapi_free(d); else a->deallocate(d); }\
  void deallocate( void* d, size_t ) const\
  { _kaapi_free(d); }\
  void cstor(InterfaceAllocator* ,  void* d, size_t count ) const\
  { TYPE*dd=(TYPE*)d; for (size_t i=0; i<count; ++i) new (&dd[i]) TYPE; }\
  void cstor(void* d, size_t count ) const\
  { TYPE*dd=(TYPE*)d; for (size_t i=0; i<count; ++i) new (&dd[i]) TYPE; }\
  void cstor( InterfaceAllocator* , void* d, const void* s, size_t count ) const\
  { TYPE* dd = (TYPE*)d; const TYPE* ss = (TYPE*)s;\
    for (size_t i=0; i<count; ++i) \
      new (&dd[i]) TYPE(ss[i]);\
  }\
  void cstor( void* d, const void* s, size_t count ) const\
  { TYPE* dd = (TYPE*)d; const TYPE* ss = (TYPE*)s;\
    for (size_t i=0; i<count; ++i) \
      new (&dd[i]) TYPE(ss[i]);\
  }\
  void dstor( InterfaceDeallocator* , void*, size_t ) const\
  { }\
  void dstor( void*, size_t ) const\
  { }\
  void* create( InterfaceAllocator* a, size_t count ) const\
  { TYPE*d=(TYPE*)(a == 0 ? _kaapi_malloc(count*sizeof(TYPE)) : a->allocate(count*sizeof(TYPE)));\
    for (size_t i=0; i<count; ++i) new (&d[i]) TYPE; return d; }\
  void* create( size_t count ) const\
  { TYPE*d=(TYPE*)_kaapi_malloc(count*sizeof(TYPE));\
    for (size_t i=0; i<count; ++i) new (&d[i]) TYPE; return d; }\
  void destroy( InterfaceDeallocator* a, void* d, size_t ) const\
  { if (a ==0) _kaapi_free(d); else a->deallocate(d); }\
  void destroy( void* d, size_t ) const\
  { _kaapi_free(d); }\
  void copy( InterfaceAllocator* , void* dest, const void* src, size_t count ) const\
  { TYPE* ddest = (TYPE*)dest; \
    const TYPE* dsrc = (const TYPE*)src;\
    for (size_t i=0; i<count; ++i) \
      ddest[i] = dsrc[i];\
  }\
  void copy( void* dest, const void* src, size_t count ) const\
  { TYPE* ddest = (TYPE*)dest; \
    const TYPE* dsrc = (const TYPE*)src;\
    for (size_t i=0; i<count; ++i) \
      ddest[i] = dsrc[i];\
  }\
  void print( std::ostream& , const void* ) const\
  { }\
  WrapperFormat(const std::string& name) : Format(kaapi_get_swid(TYPE),sizeof(TYPE),name) {}\
  static const WrapperFormat<TYPE>& theformat;\
  static const WrapperFormat<TYPE>* const format;\
  static const WrapperFormat<TYPE>* get_format()\
  { return format; }\
};

#else
#define KAAPI_SPECIALIZED_VOIDFORMAT(TYPE)\
template<>\
class WrapperFormat<TYPE> : virtual public Format {\
public:\
  void write( OStream& , const void* , size_t ) const\
  { }\
  void read( IStream& , void* , size_t ) const\
  { }\
  void* allocate( InterfaceAllocator* a, size_t count ) const\
  { return a == 0 ? _kaapi_malloc(count*sizeof(TYPE)) : a->allocate(count*sizeof(TYPE)) ; }\
  void* allocate( size_t count ) const\
  { return _kaapi_malloc(count*sizeof(TYPE)); }\
  void deallocate( InterfaceDeallocator* a, void* d, size_t ) const\
  { if (a ==0) _kaapi_free(d); else a->deallocate(d); }\
  void deallocate( void* d, size_t ) const\
  { _kaapi_free(d); }\
  void cstor(InterfaceAllocator* ,  void* d, size_t count ) const\
  { TYPE*dd=(TYPE*)d; for (size_t i=0; i<count; ++i) new (&dd[i]) TYPE; }\
  void cstor(void* d, size_t count ) const\
  { TYPE*dd=(TYPE*)d; for (size_t i=0; i<count; ++i) new (&dd[i]) TYPE; }\
  void cstor( InterfaceAllocator* , void* d, const void* s, size_t count ) const\
  { TYPE* dd = (TYPE*)d; const TYPE* ss = (TYPE*)s;\
    for (size_t i=0; i<count; ++i) \
      new (&dd[i]) TYPE(ss[i]);\
  }\
  void cstor( void* d, const void* s, size_t count ) const\
  { TYPE* dd = (TYPE*)d; const TYPE* ss = (TYPE*)s;\
    for (size_t i=0; i<count; ++i) \
      new (&dd[i]) TYPE(ss[i]);\
  }\
  void dstor( InterfaceDeallocator* , void*, size_t ) const\
  { }\
  void dstor( void*, size_t ) const\
  { }\
  void* create( InterfaceAllocator* a, size_t count ) const\
  { TYPE*d=(TYPE*)(a == 0 ? _kaapi_malloc(count*sizeof(TYPE)) : a->allocate(count*sizeof(TYPE)));\
    for (size_t i=0; i<count; ++i) new (&d[i]) TYPE; return d; }\
  void* create( size_t count ) const\
  { TYPE*d=(TYPE*)_kaapi_malloc(count*sizeof(TYPE));\
    for (size_t i=0; i<count; ++i) new (&d[i]) TYPE; return d; }\
  void destroy( InterfaceDeallocator* a, void* d, size_t ) const\
  { if (a ==0) _kaapi_free(d); else a->deallocate(d); }\
  void destroy( void* d, size_t ) const\
  { _kaapi_free(d); }\
  void copy( InterfaceAllocator* , void* dest, const void* src, size_t count ) const\
  { TYPE* ddest = (TYPE*)dest; \
    const TYPE* dsrc = (const TYPE*)src;\
    for (size_t i=0; i<count; ++i) \
      ddest[i] = dsrc[i];\
  }\
  void copy( void* dest, const void* src, size_t count ) const\
  { TYPE* ddest = (TYPE*)dest; \
    const TYPE* dsrc = (const TYPE*)src;\
    for (size_t i=0; i<count; ++i) \
      ddest[i] = dsrc[i];\
  }\
  void print( std::ostream& , const void*  ) const\
  { }\
  WrapperFormat(const std::string& name) : Format(kaapi_get_swid(TYPE),sizeof(TYPE),name) {}\
  static const WrapperFormat<TYPE>& theformat;\
  static const WrapperFormat<TYPE>* const format;\
  static const WrapperFormat<TYPE>* get_format()\
  { return format; }\
};
#endif


KAAPI_SPECIALIZED_FORMAT(bool)
KAAPI_SPECIALIZED_FORMAT(char)
KAAPI_SPECIALIZED_FORMAT(signed char)
KAAPI_SPECIALIZED_FORMAT(unsigned char)
KAAPI_SPECIALIZED_FORMAT(int)
KAAPI_SPECIALIZED_FORMAT(unsigned int)
KAAPI_SPECIALIZED_FORMAT(short)
KAAPI_SPECIALIZED_FORMAT(unsigned short)
KAAPI_SPECIALIZED_FORMAT(long)
KAAPI_SPECIALIZED_FORMAT(unsigned long)
KAAPI_SPECIALIZED_FORMAT(long long)
KAAPI_SPECIALIZED_FORMAT(unsigned long long)
KAAPI_SPECIALIZED_FORMAT(float)
KAAPI_SPECIALIZED_FORMAT(double)
KAAPI_SPECIALIZED_FORMAT(long double)




// --------------------------------------------------------------------
/** Format for bytes
*/
class ByteFormat: virtual public Format {
public:
  ByteFormat() : Format(kaapi_get_swid("byte"),1, "byte") {};

  void write( OStream&, const void*, size_t ) const;

  void read( IStream&, void*, size_t ) const;

  void* allocate( InterfaceAllocator* a, size_t count ) const
  { return a == 0 ? _kaapi_malloc(count) : a->allocate(count) ; }
  void* allocate( size_t count ) const
  { return _kaapi_malloc(count); }

  void deallocate( InterfaceDeallocator* a, void* d, size_t ) const
  { if (a ==0) _kaapi_free(d); else a->deallocate(d); }
  void deallocate( void* d, size_t ) const
  { _kaapi_free(d);}

  void cstor( InterfaceAllocator* , void* d, size_t count) const
  { ::memset(d,0, count); }
  void cstor( void* d, size_t count) const
  { ::memset(d,0, count); }


  void cstor( InterfaceAllocator* , void* d, const void* s, size_t count) const
  { ::memcpy(d, s, count); }
  void cstor( void* d, const void* s, size_t count) const
  { ::memcpy(d, s, count); }

  void dstor( InterfaceDeallocator* , void*, size_t ) const
  { }
  void dstor( void*, size_t ) const
  { }

  void* create( InterfaceAllocator* a, size_t count ) const
  { return a == 0 ? _kaapi_malloc(count) : a->allocate(count) ; }
  void* create(  size_t count ) const
  { return _kaapi_malloc(count); }

  void destroy( InterfaceDeallocator* a, void* d, size_t ) const
  { if (a ==0) _kaapi_free(d); else a->deallocate(d); }
  void destroy( void* d, size_t ) const
  { _kaapi_free(d); }

  void copy( InterfaceAllocator* , void* dest, const void* src, size_t count ) const
  { memcpy(dest, src, count); }
  void copy( void* dest, const void* src, size_t count ) const
  { memcpy(dest, src, count); }

  void print( std::ostream& o, const void* t ) const
  { o << short(*(const char*)t); }
};

/* -----------------------------------
*/
#ifdef MACOSX_EDITOR
#pragma mark ----- other high level types
#endif
// --------------------------------------------------------------------
/** Null Format
*/
class NullFormat: virtual public Format {
public:
  NullFormat() : Format(0,0,"Null") {};

  void write( OStream&, const void*, size_t ) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }

  void read( IStream&, void*, size_t ) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }

  void* allocate( InterfaceAllocator*, size_t ) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") );
    return 0;
  }
  void* allocate( size_t ) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") );
    return 0;
  }

  void deallocate( InterfaceDeallocator*, void*, size_t) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }
  void deallocate( void*, size_t) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }

  void cstor( InterfaceAllocator*, void*, size_t) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }
  void cstor( void*, size_t) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }

  void cstor( InterfaceAllocator*, void*, const void*, size_t) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }
  void cstor( void*, const void*, size_t) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }

  void dstor( InterfaceDeallocator*, void*, size_t) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }
  void dstor( void*, size_t) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }

  void* create( InterfaceAllocator*, size_t ) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") );
    return 0;
  }
  void* create( size_t ) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") );
    return 0;
  }

  void destroy( InterfaceDeallocator*, void*, size_t ) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }
  void destroy( void*, size_t ) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }

  void copy( InterfaceAllocator*, void*, const void*, size_t ) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }
  void copy( void*, const void*, size_t ) const
  { Exception_throw( RuntimeError("bad invocation on NullFormat") ); }

};


// --------------------------------------------------------------------
#define KAAPI_DECL_EXT_FORMAT(TYPE,OBJ)\
  extern const WrapperFormat<TYPE> OBJ;
namespace FormatDef {
  extern const NullFormat Null;
  KAAPI_DECL_EXT_FORMAT(bool, Bool)
  KAAPI_DECL_EXT_FORMAT(char, Char)
  extern const ByteFormat Byte;
  KAAPI_DECL_EXT_FORMAT(signed char, SChar)
  KAAPI_DECL_EXT_FORMAT(unsigned char, UChar)
  KAAPI_DECL_EXT_FORMAT(int, Int)
  KAAPI_DECL_EXT_FORMAT(unsigned int, UInt)
  KAAPI_DECL_EXT_FORMAT(short, Short)
  KAAPI_DECL_EXT_FORMAT(unsigned short, UShort)
  KAAPI_DECL_EXT_FORMAT(long, Long)
  KAAPI_DECL_EXT_FORMAT(unsigned long, ULong)
  KAAPI_DECL_EXT_FORMAT(long long, LLong)
  KAAPI_DECL_EXT_FORMAT(unsigned long long, ULLong)
  KAAPI_DECL_EXT_FORMAT(float, Float)
  KAAPI_DECL_EXT_FORMAT(double, Double)
  KAAPI_DECL_EXT_FORMAT(long double, LDouble)
};


/*
 * inline definition
 *
 */
inline Format* Format::get_format( Id id )
{ 
  std::map<Id,Format*>::iterator curr = _all_fm.find(id); 
  if (curr == _all_fm.end()) return 0;
  return curr->second;
}

inline void* Format::get_attr( void* base ) const
{ return _attr_fmt ==0 ? 0 : ((char*)base)+_attr_offset; }

inline void Format::set_attr_format(const Format* f)
{ _attr_fmt = f; }

inline const Format* Format::get_attr_format( ) const
{ return _attr_fmt; }

inline bool Format::is_continuous() const
{ return _iscontiguous; }

inline void Format::set_continuous() const
{ _iscontiguous = true; }

} // namespace atha


/*
 * inline definition in global namespace
 *
 */
/* -----------------------------------
*/
#ifdef MACOSX_EDITOR
#pragma mark ----- Output
#endif
inline atha::OStream& operator<< (atha::OStream& m, const volatile bool& v )
{ m.write( atha::WrapperFormat<bool>::format, (const bool*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile char& v )
{ m.write( atha::WrapperFormat<char>::format, (const char*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile signed char& v )
{ m.write( atha::WrapperFormat<signed char>::format, (const signed char*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile unsigned char& v )
{ m.write( atha::WrapperFormat<unsigned char>::format, (const unsigned char*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile short& v )
{ m.write( atha::WrapperFormat<short>::format, (const short*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile unsigned short& v )
{ m.write( atha::WrapperFormat<unsigned short>::format, (const unsigned short*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile int& v )
{ m.write( atha::WrapperFormat<int>::format, (const int*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile unsigned int& v )
{ m.write( atha::WrapperFormat<unsigned int>::format, (const unsigned int*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile long& v )
{ m.write( atha::WrapperFormat<long>::format, (const long*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile unsigned long& v )
{ m.write( atha::WrapperFormat<unsigned long>::format, (const unsigned long*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile long long& v )
{ m.write( atha::WrapperFormat<long long>::format, (const long long*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile unsigned long long& v )
{ m.write( atha::WrapperFormat<unsigned long long>::format, (const unsigned long long*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile float& v )
{ m.write( atha::WrapperFormat<float>::format, ( const float*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile double& v )
{ m.write( atha::WrapperFormat<double>::format, (const double*)&v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const volatile long double& v )
{ m.write( atha::WrapperFormat<long double>::format,(const long double*) &v, 1);
  return m;
}
//--
inline atha::OStream& operator<< (atha::OStream& m, const bool& v )
{ m.write( atha::WrapperFormat<bool>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const char& v )
{ m.write( atha::WrapperFormat<char>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const signed char& v )
{ m.write( atha::WrapperFormat<signed char>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const unsigned char& v )
{ m.write( atha::WrapperFormat<unsigned char>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const short& v )
{ m.write( atha::WrapperFormat<short>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const unsigned short& v )
{ m.write( atha::WrapperFormat<unsigned short>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const int& v )
{ m.write( atha::WrapperFormat<int>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const unsigned int& v )
{ m.write( atha::WrapperFormat<unsigned int>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const long& v )
{ m.write( atha::WrapperFormat<long>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const unsigned long& v )
{ m.write( atha::WrapperFormat<unsigned long>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const long long& v )
{ m.write( atha::WrapperFormat<long long>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const unsigned long long& v )
{ m.write( atha::WrapperFormat<unsigned long long>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const float& v )
{ m.write( atha::WrapperFormat<float>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const double& v )
{ m.write( atha::WrapperFormat<double>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const long double& v )
{ m.write( atha::WrapperFormat<long double>::format, &v, 1);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& m, const std::string& v )
{
  kaapi_uint32_t sz = v.size();
  m.write( atha::WrapperFormat<kaapi_uint32_t>::format, atha::OStream::IA, &sz, 1);
  m.write( atha::WrapperFormat<char>::format, atha::OStream::IA, (const char*)(v.c_str()), sz);
  return m;
}

inline atha::OStream& operator<< (atha::OStream& o, const atha::Pointer& s)
{
  o.write( &atha::FormatDef::Byte, &s, sizeof(atha::Pointer) );
  return o;
}

template<class T>
inline atha::OStream& operator<< (atha::OStream& m, const std::vector<T>& v )
{
  kaapi_uint32_t sz =v.size();
  m.write( atha::WrapperFormat<kaapi_uint32_t>::format, atha::OStream::IA, &sz, 1);
  for (unsigned i=0; i<sz; ++i)
    m.write( atha::WrapperFormat<T>::format, atha::OStream::IA, (const T*)&v[i], 1);
  return m;
}

template<class Fst, class Snd>
inline  atha::OStream& operator<< (atha::OStream& m, const std::pair<Fst,Snd>& p )
{
    return m << p.first << p.second ;
}


/* -----------------------------------
*/
#ifdef MACOSX_EDITOR
#pragma mark ----- Input
#endif
inline atha::IStream& operator>> (atha::IStream& m, bool& v )
{ m.read( atha::WrapperFormat<bool>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, char& v )
{ m.read( atha::WrapperFormat<char>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, signed char& v )
{ m.read( atha::WrapperFormat<signed char>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, unsigned char& v )
{ m.read( atha::WrapperFormat<unsigned char>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, short& v )
{ m.read( atha::WrapperFormat<short>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, unsigned short& v )
{ m.read( atha::WrapperFormat<unsigned short>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, int& v )
{ m.read( atha::WrapperFormat<int>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, unsigned int& v )
{ m.read( atha::WrapperFormat<unsigned int>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, long& v )
{ m.read( atha::WrapperFormat<long>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, unsigned long& v )
{ m.read( atha::WrapperFormat<unsigned long>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, long long& v )
{ m.read( atha::WrapperFormat<long long>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, unsigned long long& v )
{ m.read( atha::WrapperFormat<unsigned long long>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, float& v )
{ m.read( atha::WrapperFormat<float>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, double& v )
{ m.read( atha::WrapperFormat<double>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, long double& v )
{ m.read( atha::WrapperFormat<long double>::format, &v, 1);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& m, std::string& v )
{
  kaapi_uint32_t sz =0;
  m.read( atha::WrapperFormat<kaapi_uint32_t>::format, atha::OStream::IA, &sz, 1);
  v.resize(sz);
  m.read( atha::WrapperFormat<char>::format, atha::OStream::IA, (char*)&v[0], sz);
  return m;
}

inline atha::IStream& operator>> (atha::IStream& i, atha::Pointer& s)
{
  i.read( &atha::FormatDef::Byte, &s, sizeof(atha::Pointer) );
  return i;
}

template<class T>
inline atha::IStream& operator>> (atha::IStream& m, std::vector<T>& v )
{
  kaapi_uint32_t sz = 0;
  m.read( atha::WrapperFormat<kaapi_uint32_t>::format, atha::IStream::IA, &sz, 1);
  v.resize(sz);
  for (unsigned i=0; i<sz; ++i)
    m.read( atha::WrapperFormat<T>::format, atha::IStream::IA, (T*)&v[i], 1);
  return m;
}

template<class Fst, class Snd>
inline  atha::IStream& operator>> (atha::IStream& m, std::pair<Fst,Snd>& p )
{
    return m >> p.first >> p.second ;
}

template<class T>
inline void atha::WrapperFormat<T>::write
 ( atha::OStream& s, const void* val, size_t count ) const
{
  const T* ref = (const T*)val;
  for (size_t i=0; i<count; ++i)
    s << ref[i];
}

template<class T>
inline void atha::WrapperFormat<T>::read
  ( atha::IStream& s, void* val, size_t count ) const
{
  T* ref = (T*)val;
  for (size_t i=0; i<count; ++i)
    s >> ref[i];
}
#endif
