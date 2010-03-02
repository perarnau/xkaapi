/* KAAPI public interface */
/*
** athapascan-2.h
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
#ifndef _KAAPI_CPP_H_
#define _KAAPI_CPP_H_

#include "kaapi.h"
#include "ka_error.h"
#include "ka_timer.h"
#include <vector>
#include <typeinfo>

namespace ka{}

namespace ka {

  /* take a constant... should be adjusted */
  enum { STACK_ALLOC_THRESHOLD = KAAPI_MAX_DATA_ALIGNMENT };  

  // --------------------------------------------------------------------
  typedef kaapi_uint8_t  ka_uint8_t;
  typedef kaapi_uint16_t ka_uint16_t;
  typedef kaapi_uint32_t ka_uint32_t;
  typedef kaapi_uint64_t ka_uint64_t;

  typedef kaapi_int8_t   ka_int8_t;
  typedef kaapi_int16_t  ka_int16_t;
  typedef kaapi_int32_t  ka_int32_t;
  typedef kaapi_int64_t  ka_int64_t;

  /* Kaapi C++ thread <-> Kaapi C stack */
  class Thread;
  
  /* for next networking part */
  class IStream;
  class OStream;
  class ODotStream;
  class SyncGuard;
  

  /** Defined in order to used automatically generated recopy in Universal Access Mode Type constructor :
      - to convert TypeEff -> TypeInTask.
      - and to convert TypeInTask -> TypeFormal.
  */
  struct Access {
    Access( const Access& a ) : a(a.a)
    { }
    template<typename pointer>
    explicit Access( pointer& p )
    { kaapi_access_init(&a, p.ptr()); }
    operator kaapi_access_t&() 
    { return a; }
    kaapi_access_t a;    
  };
  

  // --------------------------------------------------------------------
  /** link C++ format -> kaapi format */
  class Format : public kaapi_format_t {
  public:
    Format( 
        const std::string& name,
        size_t             size,
        void             (*cstor)( void* dest),
        void             (*dstor)( void* dest),
        void             (*cstorcopy)( void* dest, const void* src),
        void             (*copy)( void* dest, const void* src),
        void             (*assign)( void* dest, const void* src),
        void             (*print)( FILE* file, const void* src)
    );
  };

  /** format for update function */
  class FormatUpdateFnc : public Format {
  public:
    FormatUpdateFnc( 
      const std::string& name,
      int (*update_mb)(void* data, const struct kaapi_format_t* fmtdata,
                       const void* value, const struct kaapi_format_t* fmtvalue )
    );
  };
  
  // --------------------------------------------------------------------  
  template <class T>
  class WrapperFormat {
  public:
    static const Format* format;
    static const Format* get_format() { return &theformat; }
    static const Format theformat;
    static void cstor( void* dest) { new (dest) T; }
    static void dstor( void* dest) { T* d = (T*)dest; d->T::~T(); } 
    static void cstorcopy( void* dest, const void* src) { T* s = (T*)src; new (dest) T(*s); } 
    static void copy( void* dest, const void* src) { T* d = (T*)dest; T* s = (T*)src; *d = *s; } 
    static void assign( void* dest, const void* src) { T* d = (T*)dest; T* s = (T*)src; *d = *s; } 
    static void print( FILE* file, const void* src) { } 
  };
  
  template <class UpdateFnc>
  class WrapperFormatUpdateFnc : public FormatUpdateFnc {
  protected:
    template<class UF, class T, class Y>
    static bool Caller( bool (UF::*)( T&, const Y& ), void* d, const void* v )
    {
      static UpdateFnc ufc;
      T* data = static_cast<T*>(d);
      const Y* value = static_cast<const Y*>(v);
      return ufc( *data, *value );
    }
    
  public:
    static int update_kaapi( void* data, const kaapi_format_t* fmtdata, const void* value, const kaapi_format_t* fmtvalue )
    {
      return Caller( &UpdateFnc::operator(), data, value ) ? 1 : 0;
    }
    static const FormatUpdateFnc* format;
    static const FormatUpdateFnc theformat;
  };

  template <class T>
  const Format WrapperFormat<T>::theformat( typeid(T).name(),
    sizeof(T),
    WrapperFormat<T>::cstor, 
    WrapperFormat<T>::dstor, 
    WrapperFormat<T>::cstorcopy, 
    WrapperFormat<T>::copy, 
    WrapperFormat<T>::assign, 
    WrapperFormat<T>::print 
  );
  template <class T> const Format* WrapperFormat<T>::format = &WrapperFormat<T>::theformat;
  template <> const Format* WrapperFormat<kaapi_int8_t>::format;
  template <> const Format* WrapperFormat<kaapi_int16_t>::format;
  template <> const Format* WrapperFormat<kaapi_int32_t>::format;
  template <> const Format* WrapperFormat<kaapi_int64_t>::format;
  template <> const Format* WrapperFormat<kaapi_uint8_t>::format;
  template <> const Format* WrapperFormat<kaapi_uint16_t>::format;
  template <> const Format* WrapperFormat<kaapi_uint32_t>::format;
  template <> const Format* WrapperFormat<kaapi_uint64_t>::format;
  template <> const Format* WrapperFormat<float>::format;
  template <> const Format* WrapperFormat<double>::format;
  template <>
  class WrapperFormat<Access> {
  public:
    static const Format* get_format() { return &theformat; }
    static const Format theformat;
    static void cstor( void* dest) {}
    static void dstor( void* dest) {} 
    static void cstorcopy( void* dest, const void* src) { const Access* s = (const Access*)src; new (dest) Access(*s); } 
    static void copy( void* dest, const void* src) {} 
    static void assign( void* dest, const void* src) {} 
    static void print( FILE* file, const void* src) {} 
  };
  

  template <class UpdateFnc>
  const FormatUpdateFnc WrapperFormatUpdateFnc<UpdateFnc>::theformat (
    typeid(UpdateFnc).name(),
    &WrapperFormatUpdateFnc<UpdateFnc>::update_kaapi
  );
  template <class UpdateFnc>
  const FormatUpdateFnc* WrapperFormatUpdateFnc<UpdateFnc>::format = &WrapperFormatUpdateFnc<UpdateFnc>::theformat;


  // --------------------------------------------------------------------
  class Community {
  protected:
    friend class System;
    Community( void* com )
    { }

  public:
    Community( const Community& com );

    /* */
    void commit();

    /* */
    void leave();

    /* */
    bool is_leader() const;
  };

  // --------------------------------------------------------------------
  class System {
  public:
    static Community join_community( int& argc, char**& argv )
      throw (RuntimeError,RestartException,ServerException);

    static Community initialize_community( int& argc, char**& argv )
      throw (RuntimeError,RestartException,ServerException);

    static Thread* get_current_thread();
    static int getRank();
    static void terminate();

  public:
  };

  // --------------------------------------------------------------------
  inline Thread* System::get_current_thread()
  {
    return (Thread*)kaapi_self_stack();
  }

  // --------------------------------------------------------------------
  /* same method exists in thread interface */
  template<class T>
  T* Alloca(size_t size)
  {
     void* data = kaapi_stack_pushdata( kaapi_self_stack(), sizeof(T)*size );
     return new (data) T[size];
  }

  // --------------------------------------------------------------------
  struct SetStack {};
  extern SetStack SetInStack;

  // --------------------------------------------------------------------
  struct SetHeap {};
  extern SetHeap SetInHeap;

  // --------------------------------------------------------------------
  class SetStickyC{};
  extern SetStickyC SetSticky;

  // --------------------------------------------------------------------
  /* typenames for access mode */
  struct VALUE_MODE  { enum {value = KAAPI_ACCESS_MODE_V}; };
  struct READ        { enum {value = KAAPI_ACCESS_MODE_R}; };
  struct WRITE       { enum {value = KAAPI_ACCESS_MODE_W}; };
  struct READWRITE   { enum {value = KAAPI_ACCESS_MODE_RW}; };
  struct C_WRITE     { enum {value = KAAPI_ACCESS_MODE_CW}; };
  struct READ_P      { enum {value = KAAPI_ACCESS_MODE_R  |KAAPI_ACCESS_MASK_MODE_P}; };
  struct WRITE_P     { enum {value = KAAPI_ACCESS_MODE_W  |KAAPI_ACCESS_MASK_MODE_P}; };
  struct READWRITE_P { enum {value = KAAPI_ACCESS_MODE_RW |KAAPI_ACCESS_MASK_MODE_P}; };
  struct C_WRITE_P   { enum {value = KAAPI_ACCESS_MODE_CW |KAAPI_ACCESS_MASK_MODE_P}; };

  /* internal name */
  typedef VALUE_MODE  ACCESS_MODE_V;
  typedef READ        ACCESS_MODE_R;
  typedef WRITE       ACCESS_MODE_W;
  typedef READWRITE   ACCESS_MODE_RW;
  typedef C_WRITE     ACCESS_MODE_CW;
  typedef READ_P      ACCESS_MODE_RP;
  typedef WRITE_P     ACCESS_MODE_WP;
  typedef READWRITE_P ACCESS_MODE_RPWP;
  typedef C_WRITE_P   ACCESS_MODE_CWP;

  struct TYPE_INTASK {}; /* internal purpose to define representation of a type in a task */
  struct TYPE_INPROG {}; /* internal purpose to define representation of a type in the user program */

  /* fwd declarations */
  template<class T>
  class pointer;
  template<class T>
  class pointer_rpwp;
  template<class T>
  class pointer_rw;
  template<class T>
  class pointer_rp;
  template<class T>
  class pointer_r;
  template<class T>
  class pointer_wp;
  template<class T>
  class pointer_w;
  template<class T>
  class pointer_cwp;
  template<class T>
  class pointer_cw;


  // --------------------------------------------------------------------
  template<class T>
  struct base_pointer {
    base_pointer() 
#if defined(KAAPI_DEBUG)
     : _ptr(0)
#endif
    {}
    base_pointer( T* p ) : _ptr(p)
    {}
    T* ptr() const { return _ptr; }
  protected:
    mutable T* _ptr;
  };

  /* capture write */
  template<class T>
  class value_ref {
  public:
    value_ref(T* p) : _ptr(p){}
    void operator=( const T& value ) { *_ptr = value; }
  protected:
    T* _ptr;
  };
  

#define KAAPI_POINTER_ARITHMETIC_METHODS\
    Self_t& operator++() { ++base_pointer<T>::_ptr; return *this; }\
    Self_t operator++(int) { return base_pointer<T>::_ptr++; }\
    Self_t& operator--() { --base_pointer<T>::_ptr; return *this; }\
    Self_t operator--(int) { return base_pointer<T>::_ptr--; }\
    Self_t operator+(int i) const { return base_pointer<T>::_ptr+i; }\
    Self_t operator+(long i) const { return base_pointer<T>::_ptr+i; }\
    Self_t operator+(difference_type i) const { return base_pointer<T>::_ptr+i; }\
    Self_t& operator+=(int i) { base_pointer<T>::_ptr+=i; return *this; }\
    Self_t& operator+=(long i) { base_pointer<T>::_ptr+=i; return *this; }\
    Self_t& operator+=(difference_type i) { base_pointer<T>::_ptr+=i; return *this; }\
    Self_t operator-(int i) const { return base_pointer<T>::_ptr-i; }\
    Self_t operator-(long i) const { return base_pointer<T>::_ptr-i; }\
    Self_t operator-(difference_type i) const { return base_pointer<T>::_ptr-i; }\
    Self_t& operator-=(int i) { return base_pointer<T>::_ptr-=i; }\
    Self_t& operator-=(long i) { return base_pointer<T>::_ptr-=i; }\
    Self_t& operator-=(difference_type i) { base_pointer<T>::_ptr-=i; return *this; }\
    difference_type operator-(const Self_t& p) const { return base_pointer<T>::_ptr-p._ptr; }
  
  // --------------------------------------------------------------------
  /* Information notes.
     - Access mode types (ka::W, ka::WP, ka::RW..) are defined to be used 
     in signature definition of tasks. They cannot be used to declare 
     variables or used as effective parameters during a spawn.
     - Effective parameters should be pointer in order to force verification
     of the parameter passing rules between effective parameters and formal parameters.
     They are closed to the Shared types of the previous Athapascan API but 
     may be used like normal pointer (arithmetic + deferencing of pointers).
  */

  // --------------------------------------------------------------------
  template<class T>
  class pointer : public base_pointer<T> {
  public:
    typedef T value_type;
    typedef size_t difference_type;
    typedef pointer<T> Self_t;
    pointer() : base_pointer<T>() {}
    pointer( value_type* ptr ) : base_pointer<T>(ptr) {}
    operator value_type*() { return base_pointer<T>::ptr(); }

    KAAPI_POINTER_ARITHMETIC_METHODS
  };


  // --------------------------------------------------------------------
  template<class T>
  class pointer_rpwp : public base_pointer<T> {
  public:
    typedef T value_type;
    typedef size_t difference_type;
    typedef pointer_rpwp<T> Self_t;

    pointer_rpwp() : base_pointer<T>() {}
    pointer_rpwp( value_type* ptr ) : base_pointer<T>(ptr) {}
    explicit pointer_rpwp( kaapi_access_t& ptr ) : base_pointer<T>(kaapi_data(value_type, &ptr)) {}
    operator value_type*() { return base_pointer<T>::ptr(); }

    KAAPI_POINTER_ARITHMETIC_METHODS
  };


  // --------------------------------------------------------------------
  template<class T>
  class pointer_rw: public base_pointer<T> {
  public:
    typedef T value_type;
    typedef size_t difference_type;
    typedef pointer_w<T> Self_t;

    pointer_rw() : base_pointer<T>() {}
    pointer_rw( value_type* ptr ) : base_pointer<T>(ptr) {}
    explicit pointer_rw( kaapi_access_t& ptr ) : base_pointer<T>(kaapi_data(value_type, &ptr)) {}
    pointer_rw( const pointer_rpwp<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_rw( const pointer<T>& ptr ) : base_pointer<T>(ptr) {}
    operator value_type*() { return base_pointer<T>::ptr(); }
    value_type& operator*() { return *base_pointer<T>::ptr(); }
    value_type& operator[](int i) { return base_pointer<T>::ptr()[i]; }
    value_type& operator[](difference_type i) { return base_pointer<T>::ptr()[i]; }

    KAAPI_POINTER_ARITHMETIC_METHODS
  };
  

  // --------------------------------------------------------------------
  template<class T>
  class pointer_rp : public base_pointer<T> {
  public:
    typedef T value_type;
    typedef size_t difference_type;
    typedef pointer_rp<T> Self_t;

    pointer_rp() : base_pointer<T>() {}
    pointer_rp( value_type* ptr ) : base_pointer<T>(ptr) {}
    explicit pointer_rp( kaapi_access_t& ptr ) : base_pointer<T>(kaapi_data(value_type, &ptr)) {}
    pointer_rp( const pointer_rpwp<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_rp( const pointer<T>& ptr ) : base_pointer<T>(ptr) {}
    operator value_type*() { return base_pointer<T>::ptr(); }

    KAAPI_POINTER_ARITHMETIC_METHODS
  };


  // --------------------------------------------------------------------
  template<class T>
  class pointer_r : public base_pointer<T> {
  public:
    typedef T value_type;
    typedef size_t difference_type;
    typedef pointer_r<T> Self_t;

    pointer_r() : base_pointer<T>() {}
    pointer_r( value_type* ptr ) : base_pointer<T>(ptr) {}
    explicit pointer_r( kaapi_access_t& ptr ) : base_pointer<T>(kaapi_data(value_type, &ptr)) {}
    pointer_r( const pointer_rpwp<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_r( const pointer<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_r( const pointer_rp<T>& ptr ) : base_pointer<T>(ptr) {}
    operator value_type*() { return base_pointer<T>::ptr(); }
    const T& operator*() const { return *base_pointer<T>::ptr(); }
    const T& operator[](int i) const { return base_pointer<T>::ptr()[i]; }
    const T& operator[](long i) const { return base_pointer<T>::ptr()[i]; }
    const T& operator[](difference_type i) const { return base_pointer<T>::ptr()[i]; }

    KAAPI_POINTER_ARITHMETIC_METHODS
  };


  // --------------------------------------------------------------------
  template<class T>
  class pointer_wp : public base_pointer<T> {
  public:
    typedef T value_type;
    typedef size_t difference_type;
    typedef pointer_wp<T> Self_t;
    
    pointer_wp() : base_pointer<T>() {}
    pointer_wp( value_type* ptr ) : base_pointer<T>(ptr) {}
    explicit pointer_wp( kaapi_access_t& ptr ) : base_pointer<T>(kaapi_data(value_type, &ptr)) {}
    pointer_wp( const pointer_rpwp<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_wp( const pointer<T>& ptr ) : base_pointer<T>(ptr) {}
    operator value_type*() { return base_pointer<T>::ptr(); }

    KAAPI_POINTER_ARITHMETIC_METHODS
  };


  // --------------------------------------------------------------------
  template<class T>
  class pointer_w : public base_pointer<T> {
  public:
    typedef T value_type;
    typedef size_t difference_type;
    typedef pointer_w<T> Self_t;

    pointer_w() : base_pointer<T>() {}
    pointer_w( value_type* ptr ) : base_pointer<T>(ptr) {}
    explicit pointer_w( kaapi_access_t& ptr ) : base_pointer<T>(kaapi_data(value_type, &ptr)) {}
    pointer_w( const pointer_rpwp<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_w( const pointer<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_w( const pointer_wp<T>& ptr ) : base_pointer<T>(ptr) {}
    operator value_type*() { return base_pointer<T>::ptr(); }
    value_ref<T> operator*() { return value_ref<T>(base_pointer<T>::ptr()); }
    value_ref<T> operator[](int i) { return value_ref<T>(base_pointer<T>::ptr()+i); }
    value_ref<T> operator[](long i) { return value_ref<T>(base_pointer<T>::ptr()+i); }
    value_ref<T> operator[](difference_type i) { return value_ref<T>(base_pointer<T>::ptr()+i); }

    KAAPI_POINTER_ARITHMETIC_METHODS
  };


 // --------------------------------------------------------------------
  template<class T>
  class pointer_cwp : public base_pointer<T> {
  public:
    typedef T value_type;
    typedef size_t difference_type;
    typedef pointer_cwp<T> Self_t;
    
    pointer_cwp() : base_pointer<T>() {}
    pointer_cwp( value_type* ptr ) : base_pointer<T>(ptr) {}
    explicit pointer_cwp( kaapi_access_t& ptr ) : base_pointer<T>(kaapi_data(value_type, &ptr)) {}
    pointer_cwp( const pointer_rpwp<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_cwp( const pointer<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_cwp( const pointer_cwp<T>& ptr ) : base_pointer<T>(ptr) {}
    operator value_type*() { return base_pointer<T>::ptr(); }

    KAAPI_POINTER_ARITHMETIC_METHODS
  };


  // --------------------------------------------------------------------
  template<class T>
  class pointer_cw: public base_pointer<T> {
  public:
    typedef T value_type;
    typedef size_t difference_type;
    typedef pointer_cw<T> Self_t;

    pointer_cw() : base_pointer<T>() {}
    pointer_cw( value_type* ptr ) : base_pointer<T>(ptr) {}
    explicit pointer_cw( kaapi_access_t& ptr ) : base_pointer<T>(kaapi_data(value_type, &ptr)) {}
    pointer_cw( const pointer_rpwp<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_cw( const pointer<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_cw( const pointer_cwp<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_cw( const pointer_cw<T>& ptr ) : base_pointer<T>(ptr) {}
    operator value_type*() { return base_pointer<T>::ptr(); }

    value_type& operator*() { return *base_pointer<T>::ptr(); }
    value_type& operator[](int i) { return base_pointer<T>::ptr()[i]; }
    value_type& operator[](long i) { return base_pointer<T>::ptr()[i]; }
    value_type& operator[](difference_type i) { return base_pointer<T>::ptr()[i]; }
    
    KAAPI_POINTER_ARITHMETIC_METHODS
  };
  

  // --------------------------------------------------------------------
  /** Trait for universal access mode type.
      The universal access mode type allows to 1/ compact code by providing generic information; 2/ define
      for any user level type, the way the type may be accessed in task, its task internal representation as
      well as its program level representation.
      Moreover, the UAMParam is used during conversion of type when effective parameter is binded to the internal
      parameter and when the internal parameter is binded to formal parameter during the call of the task body.
       *  TraitUAMType<T>::UAMParam<MODE>::type_t gives the representation of the type for all kinds of access mode.
       *  TraitUAMType<T>::UAMParam<TYPE_INTASK>::type_t gives the representation of the type in a task.
       *  TraitUAMType<T>::UAMParam<TYPE_INPROG>::type_t gives the representation of the type in the user program.
       
      During the creation of tasks with a parameter fi of type F, the recopy constructor is called like this:
        new (&task->fi) TraitUAMType<F>::UAMParam<TYPE_INTASK>::type_t( ei )
      where the effective parameter ei is of type TraitUAMType<F>::UAMParam<EffectiveAccessMode>::type_t
        
      Once the task is created and scheduled, the runtime will invokes the user defined function and required
      to convert TraitUAMType<F>::UAMParam<TYPE_INTASK>::type_t to TraitUAMType<F>::UAMParam<FORMAL_MODE>::type_t.
      This is called using the explicit constructor of recopy of TraitUAMType<F>::UAMParam<FORMAL_MODE>::type_t
      from TraitUAMType<F>::UAMParam<TYPE_INTASK>::type_t.
  */
  template<typename T>
  struct TraitUAMTypeFormat { typedef T type_t; };
  template<typename T, typename Mode>
  struct TraitUAMTypeParam { typedef T type_t; };
  template<typename T>
  struct TraitUAMTypeParam<const T&, TYPE_INTASK> { typedef T type_t; };

  template<typename T>
  struct TraitUAMType {
    typedef typename TraitUAMTypeFormat<T>::type_t typeformat_t;

    template<typename Mode>
    struct UAMParam {
      typedef TraitUAMType<T>                            uamparam_t;
      typedef typename TraitUAMTypeParam<T,Mode>::type_t type_t;
      typedef Mode                                       mode_t;
    };

    /* tells where are the pointer in the structure */
    static const int count_pointer = 0;
    static const int offset_pointer[];
  };

  /* This specialization describes how to represent Kaapi pointer
  */
  template<typename T>
  struct TraitUAMTypeFormat<pointer<T> > { typedef T type_t; };
  template<typename T>
  struct TraitUAMTypeParam<pointer<T>, TYPE_INTASK> { typedef Access type_t; };
  template<typename T>
  struct TraitUAMTypeParam<pointer<T>, TYPE_INPROG> { typedef pointer<T> type_t; };
  template<typename T>
  struct TraitUAMTypeParam<pointer<T>, ACCESS_MODE_R> { typedef pointer_r<T> type_t; };
  template<typename T>
  struct TraitUAMTypeParam<pointer<T>, ACCESS_MODE_W> { typedef pointer_w<T> type_t; };
  template<typename T>
  struct TraitUAMTypeParam<pointer<T>, ACCESS_MODE_RW> { typedef pointer_rw<T> type_t; };
  template<typename T>
  struct TraitUAMTypeParam<pointer<T>, ACCESS_MODE_CW> { typedef pointer_cw<T> type_t; };
  template<typename T>
  struct TraitUAMTypeParam<pointer<T>, ACCESS_MODE_RP> { typedef pointer_rp<T> type_t; };
  template<typename T>
  struct TraitUAMTypeParam<pointer<T>, ACCESS_MODE_WP> { typedef pointer_wp<T> type_t; };
  template<typename T>
  struct TraitUAMTypeParam<pointer<T>, ACCESS_MODE_RPWP> { typedef pointer_rpwp<T> type_t; };
  template<typename T>
  struct TraitUAMTypeParam<pointer<T>, ACCESS_MODE_CWP> { typedef pointer_cwp<T> type_t; };


  // --------------------------------------------------------------------
  /* Helpers to declare type in signature of task */
  template<typename UserType> struct Value {};
  template<typename UserType, typename AccessMode> struct Composite {};
  template<typename UserType> struct RPWP {};
  template<typename UserType> struct RP {};
  template<typename UserType> struct R  {};
  template<typename UserType> struct WP {};
  template<typename UserType> struct W {};
  template<typename UserType> struct RW {};
  
  template<typename UserType>
  struct DefaultAdd {
    void operator()( UserType& result, const UserType& value ) const
    { result += value; }
  };
  
  template<typename UserType/*, class OpCumul = DefaultAdd<UserType>*/ > struct CWP {};
  template<typename UserType/*, class OpCumul = DefaultAdd<UserType>*/ > struct CW {};

  // --------------------------------------------------------------------  
  /* Trait used in each type of parameter in the signature to retreive the
     UAMType and its access mode.
  */
  template<typename UserType>
  struct TraitUAMParam {
    typedef TraitUAMType<UserType> uamttype_t;
    typedef ACCESS_MODE_V   mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<Value<UserType> > {
    typedef TraitUAMType<UserType> uamttype_t;
    typedef ACCESS_MODE_V   mode_t;
  };

  template<typename UserType, typename AccessMode>
  struct TraitUAMParam<Composite<UserType, AccessMode> > {
    typedef TraitUAMType<UserType> uamttype_t;
    typedef AccessMode             mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<RPWP<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_RPWP       mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<RW<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_RW         mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<RP<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_RP         mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<R<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_R          mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<WP<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_WP         mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<W<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_W          mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<CWP<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_CWP        mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<CW<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_CW         mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<pointer<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_RPWP       mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<pointer_rpwp<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_RPWP       mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<pointer_rw<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_RW         mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<pointer_rp<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_RP         mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<pointer_r<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_R          mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<pointer_wp<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_WP         mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<pointer_w<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_W          mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<pointer_cwp<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_CWP        mode_t;
  };

  template<typename UserType>
  struct TraitUAMParam<pointer_cw<UserType> > {
    typedef TraitUAMType<pointer<UserType> > uamttype_t;
    typedef ACCESS_MODE_CW         mode_t;
  };

  // --------------------------------------------------------------------  
  class DefaultAttribut {
  public:
    kaapi_task_t* operator()( kaapi_stack_t*, kaapi_task_t* clo) const
    { return clo; }
  };
  extern DefaultAttribut SetDefault;
  
  /* */
  class UnStealableAttribut {
  public:
    kaapi_task_t* operator()( kaapi_stack_t*, kaapi_task_t* clo) const
    { clo->flag |= KAAPI_TASK_STICKY; return clo; }
  };
  inline UnStealableAttribut SetUnStealable()
  { return UnStealableAttribut(); }

  /* like default attribut: not yet distributed computation */
  class SetLocalAttribut {
  public:
    kaapi_task_t* operator()( kaapi_stack_t*, kaapi_task_t* clo) const
    { 
      kaapi_task_setflags( clo, KAAPI_TASK_STICKY );
      return clo; 
    }
  };
  extern SetLocalAttribut SetLocal;

  /* do nothing... not yet distributed implementation */
  class AttributSetSite {
    int _site;
  public:
    AttributSetSite( int s ) : _site(s) {}
    kaapi_task_t* operator()( kaapi_stack_t*, kaapi_task_t* clo) const
    { return clo; }
  };

  inline AttributSetSite SetSite( int s )
  { return AttributSetSite(s); }
  
  /* do nothing */
  class SetStaticSchedAttribut {
    int _npart;
    int _niter;
  public:
    SetStaticSchedAttribut( int n, int m  ) 
     : _npart(n), _niter(m) {}
    template<class A1_CLO>
    kaapi_task_t* operator()( kaapi_stack_t*, A1_CLO*& clo) const
    { return clo; }
  };
  inline SetStaticSchedAttribut SetStaticSched(int npart, int iter = 1 )
  { return SetStaticSchedAttribut(npart, iter); }


  // --------------------------------------------------------------------
#if defined(KAAPI_DEBUG)
  /* for better understand error message */
  template<int i>
  struct FOR_ARG {};

  /* for better understand error message */
  template<class TASK>
  struct FOR_TASKNAME {};
  
  /* ME: effectif -> MF: formal */
  template<class ME, class MF, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE {
//    static void IS_COMPATIBLE();
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_V, ACCESS_MODE_V, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_R, ACCESS_MODE_R, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK> /* this rule is only valid for terminal fork... */
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_W, ACCESS_MODE_W, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_CW, ACCESS_MODE_CW, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_RPWP, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_RW, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_WP, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_W, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_CWP, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_CW, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_R, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_RP, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RP, ACCESS_MODE_RP, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RP, ACCESS_MODE_R, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_WP, ACCESS_MODE_WP, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_WP, ACCESS_MODE_W, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_CWP, ACCESS_MODE_CW, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_CWP, ACCESS_MODE_CWP, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
#endif

  /* ICI: signature avec kaapi_stack & kaapi_task as first parameter ?
     Quel interface C++ pour les tâches adaptatives ?
  */
  
  template<int i>
  struct Task {};

} // end of namespace atha: following definition sould be in global namespace in 
  // order to be specialized easily

  // --------------------------------------------------------------------
  
  template<class TASK>
  struct TaskBodyCPU : public TASK {};

  // --------------------------------------------------------------------
  template<class TASK>
  struct TaskBodyGPU : public TASK {};


namespace ka {

  // --------------------------------------------------------------------
  template<class TASK>
  struct KaapiTask0 {
    static void body( kaapi_task_t* task, kaapi_stack_t* stack )
    { 
      static TASK dummy;
      dummy();
    }
  };

#include "ka_api_clo.h"

  // --------------------------------------------------------------------
  /* New API: thread.Spawn<TASK>([ATTR])( args )
     Spawn<TASK>([ATTR])(args) with be implemented on top of 
     System::get_current_thread()->Spawn<TASK>([ATTR])( args ).
  */
  class Thread {
  public:

    template<class T>
    T* Alloca(size_t size)
    {
       void* data = kaapi_stack_pushdata( &_stack, sizeof(T)*size );
       return new (data) T[size];
    }

    template<class TASK, class Attr>
    class Spawner {
    public:
      Spawner( kaapi_stack_t* s, const Attr& a ) : _stack(s), _attr(a) {}

      /**
      **/      
      void operator()()
      { 
        kaapi_task_t* clo = kaapi_stack_toptask( _stack);
        kaapi_task_initdfg( _stack, clo, KaapiTask0<TASK>::body, 0 );
        _attr(_stack, clo);
        kaapi_stack_pushtask( _stack);    
      }

#include "ka_api_fork.h"

    protected:
      kaapi_stack_t* _stack;
      const Attr&    _attr;
    };
        
    template<class TASK>
    Spawner<TASK, DefaultAttribut> Spawn() { return Spawner<TASK, DefaultAttribut>(&_stack, DefaultAttribut()); }

    template<class TASK, class Attr>
    Spawner<TASK, Attr> Spawn(const Attr& a) { return Spawner<TASK, Attr>(&_stack, a); }

  protected:
    kaapi_stack_t _stack;
    friend class SyncGuard;
  };

  
  
  // --------------------------------------------------------------------
  /** Top level Spawn */
  template<class TASK>
  Thread::Spawner<TASK, DefaultAttribut> Spawn() { return Thread::Spawner<TASK, DefaultAttribut>(kaapi_self_stack(), DefaultAttribut()); }

  template<class TASK, class Attr>
  Thread::Spawner<TASK, Attr> Spawn(const Attr& a) { return Thread::Spawner<TASK, Attr>(kaapi_self_stack(), a); }



  // --------------------------------------------------------------------
  /** Wait execution of all forked tasks of the running task */
  extern void Sync();



  // --------------------------------------------------------------------
  /* Main task */
  template<class TASK>
  struct MainTask {
    int    argc;
    char** argv;
    static void body_cpu( kaapi_task_t* task, kaapi_stack_t* stack )
    {
      MainTask<TASK>* args = kaapi_task_getargst( task, MainTask<TASK>);
      TASK()( args->argc, args->argv );
    }
    static void body_gpu( kaapi_task_t* task, kaapi_stack_t* stack )
    {
      MainTask<TASK>* args = kaapi_task_getargst( task, MainTask<TASK>);
      TASK()( args->argc, args->argv );
    }
    void operator()( kaapi_task_t* task, kaapi_stack_t* stack )
    {
      MainTask<TASK>* args = kaapi_task_getargst( task, MainTask<TASK>);
      TASK()( args->argc, args->argv );
    }
    static kaapi_format_t* registerformat()
    {
      if (MainTask::fmid != 0) return &MainTask::format;
      MainTask::fmid = kaapi_format_taskregister( 
            &MainTask::getformat, 
#if defined(KAAPI_VERY_COMPACT_TASK)
            -1,
#else
            0,
#endif
            &MainTask::body_cpu, 
            typeid(MainTask).name(),
            sizeof(MainTask),
            0,
            0,
            0,
            0
        );
     int (TASK::*f_defaultcpu)(...) = (int (TASK::*)(...))&TASK::operator();  /* inherited from Signature */
     int (TASK::*f_cpu)(...) = (int (TASK::*)(...))&TaskBodyCPU<TASK>::operator();
     if (f_cpu == f_defaultcpu) {
       MainTask::format.entrypoint[KAAPI_PROC_TYPE_CPU] = 0;
     }
     else {
       MainTask::format.entrypoint[KAAPI_PROC_TYPE_CPU] = &MainTask::body_cpu;
     }
     int (MainTask::*f_defaultgpu)(...) = (int (MainTask::*)(...))&MainTask::operator();  /* inherited from Signature */
     int (MainTask::*f_gpu)(...) = (int (MainTask::*)(...))&TaskBodyGPU<MainTask>::operator();
     if (f_gpu == f_defaultgpu) {
       MainTask::format.entrypoint[KAAPI_PROC_TYPE_GPU] = 0;
     }
     else {
       MainTask::format.entrypoint[KAAPI_PROC_TYPE_GPU] = &MainTask::body_gpu;
     }
      return &MainTask::format;
    }  
    static const kaapi_task_bodyid_t bodyid;
    static kaapi_format_t    format;
    static kaapi_format_id_t fmid;
    static kaapi_format_t* getformat()
    { return &format; }
  };
  
  template<class TASK>
  kaapi_format_t    MainTask<TASK>::format;
  template<class TASK>
  kaapi_format_id_t MainTask<TASK>::fmid =0;

  template<class TASK>
  const kaapi_task_bodyid_t MainTask<TASK>::bodyid = registerformat()->bodyid;
  
  template<class TASK>
  struct SpawnerMain
  {
    SpawnerMain() 
    { }

    void operator()( int argc, char** argv)
    {
      kaapi_stack_t* stack = kaapi_self_stack();
      kaapi_task_t* clo = kaapi_stack_toptask( stack);
      kaapi_task_initdfg( stack, clo, MainTask<TASK>::bodyid, kaapi_stack_pushdata(stack, sizeof(MainTask<TASK>)) );
      kaapi_task_setflags( clo, KAAPI_TASK_STICKY );
      MainTask<TASK>* arg = kaapi_task_getargst( clo, MainTask<TASK>);
      arg->argc = argc;
      arg->argv = argv;
      kaapi_stack_pushtask( stack);    
    }
  };

  template<class TASK>
  SpawnerMain<TASK> SpawnMain()
  { 
    return SpawnerMain<TASK>();
  }
    


  // --------------------------------------------------------------------
  extern std::ostream& logfile();

  // --------------------------------------------------------------------
  class SyncGuard {
      Thread       *_thread;
      kaapi_frame_t _frame;
  public:
      SyncGuard() : _thread( System::get_current_thread() )
      {
        kaapi_stack_save_frame( &_thread->_stack, &_frame );
      }
      ~SyncGuard()
      {
        kaapi_sched_sync( &_thread->_stack );
        kaapi_stack_restore_frame( &_thread->_stack, &_frame );
      }
  };
} // namespace ka

#ifndef _KAAPIPLUSPLUS_NOT_IN_NAMESPACE
using namespace ka;
#endif

#endif

