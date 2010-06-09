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

  struct kaapi_bodies_t {
    kaapi_bodies_t( kaapi_task_body_t cpu_body, kaapi_task_body_t gpu_body );
    kaapi_task_body_t cpu_body;
    kaapi_task_body_t gpu_body;
    kaapi_task_body_t default_body;
  };
  
  /* Kaapi C++ thread <-> Kaapi C thread */
  class Thread;

  /* Kaapi C++ threadgroup <-> Kaapi C threadgroup */
  class ThreadGroup;
  
  /* for next networking part */
  class IStream;
  class OStream;
  class ODotStream;
  class SyncGuard;
  

  // --------------------------------------------------------------------
  /** link C++ format -> kaapi format */
  class Format {
  public:
    Format( 
        const std::string& name = "empty",
        size_t             size = 0,
        void             (*cstor)( void* dest) = 0,
        void             (*dstor)( void* dest) = 0,
        void             (*cstorcopy)( void* dest, const void* src) = 0,
        void             (*copy)( void* dest, const void* src) = 0,
        void             (*assign)( void* dest, const void* src) = 0,
        void             (*print)( FILE* file, const void* src) = 0
    );
  struct kaapi_format_t* get_c_format();
  const struct kaapi_format_t* get_c_format() const;

  public:
    /* should be protected and friend with wrapperformat */
    Format( kaapi_format_t* f );
    void reinit( kaapi_format_t* f ) const;

  protected:
    mutable struct kaapi_format_t* fmt;
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

  /** format for update function */
  class FormatTask : public Format {
  public:
    FormatTask( 
      const std::string&          name,
      size_t                      size,
      int                         count,
      const kaapi_access_mode_t   mode_param[],
      const kaapi_offset_t        offset_param[],
      const kaapi_format_t*       fmt_param[]
    );
  };
  
  // --------------------------------------------------------------------  
  template <class T>
  class WrapperFormat : public Format {
    WrapperFormat(kaapi_format_t* f);
  public:
    WrapperFormat();
    static const WrapperFormat<T> format;
    static const Format* get_format();
    static const kaapi_format_t* get_c_format();
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
    static const FormatUpdateFnc format;
    static const FormatUpdateFnc* get_format();
  };

  template <class T>
  WrapperFormat<T>::WrapperFormat()
   : Format(  typeid(T).name(),
              sizeof(T),
              WrapperFormat<T>::cstor, 
              WrapperFormat<T>::dstor, 
              WrapperFormat<T>::cstorcopy, 
              WrapperFormat<T>::copy, 
              WrapperFormat<T>::assign, 
              WrapperFormat<T>::print 
    )
  {}

  template <class T>
  WrapperFormat<T>::WrapperFormat(kaapi_format_t* f)
   : Format( f )
  {}

  template <class T>
  const WrapperFormat<T> WrapperFormat<T>::format;
  template <class T>
  const Format* WrapperFormat<T>::get_format() 
  { 
    return &format; 
  }
  template <class T>
  const kaapi_format_t* WrapperFormat<T>::get_c_format() 
  { 
    return format.Format::get_c_format(); 
  }  


  template <> const WrapperFormat<char> WrapperFormat<char>::format;
  template <> const WrapperFormat<short> WrapperFormat<short>::format;
  template <> const WrapperFormat<int> WrapperFormat<int>::format;
  template <> const WrapperFormat<long> WrapperFormat<long>::format;
  template <> const WrapperFormat<unsigned char> WrapperFormat<unsigned char>::format;
  template <> const WrapperFormat<unsigned short> WrapperFormat<unsigned short>::format;
  template <> const WrapperFormat<unsigned int> WrapperFormat<unsigned int>::format;
  template <> const WrapperFormat<unsigned long> WrapperFormat<unsigned long>::format;
  template <> const WrapperFormat<float> WrapperFormat<float>::format;
  template <> const WrapperFormat<double> WrapperFormat<double>::format;
  
  template <class UpdateFnc>
  const FormatUpdateFnc WrapperFormatUpdateFnc<UpdateFnc>::format (
    typeid(UpdateFnc).name(),
    &WrapperFormatUpdateFnc<UpdateFnc>::update_kaapi
  );

  template <class UpdateFnc>
  const FormatUpdateFnc* WrapperFormatUpdateFnc<UpdateFnc>::get_format()
  {
    return &format;
  }


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
    return (Thread*)kaapi_self_thread();
  }

  // --------------------------------------------------------------------
  /* same method exists in thread interface */
  template<class T>
  T* Alloca(size_t size)
  {
     void* data = kaapi_thread_pushdata( kaapi_self_thread(), sizeof(T)*size );
     return new (data) T[size];
  }

  template<class T>
  T* Alloca()
  {
     void* data = kaapi_thread_pushdata( kaapi_self_thread(), sizeof(T) );
     return new (data) T[1];
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
    void ptr(T* p) { _ptr = p; }
  protected:
    mutable T* _ptr;
  };

  /* capture write */
  template<class T>
  class value_ref {
  public:
    value_ref(T* p) : _ptr(p){}
    operator T&() { return *_ptr; }
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
    typedef pointer_rw<T> Self_t;

    pointer_rw() : base_pointer<T>() {}
    pointer_rw( value_type* ptr ) : base_pointer<T>(ptr) {}
    explicit pointer_rw( kaapi_access_t& ptr ) : base_pointer<T>(kaapi_data(value_type, &ptr)) {}
    pointer_rw( const pointer_rpwp<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_rw( const pointer<T>& ptr ) : base_pointer<T>(ptr) {}
    operator value_type*() { return base_pointer<T>::ptr(); }
    T* operator->() { return base_pointer<T>::ptr(); }
    value_type& operator*() { return *base_pointer<T>::ptr(); }
    value_type& operator[](int i) { return base_pointer<T>::ptr()[i]; }
    value_type& operator[](long i) { return base_pointer<T>::ptr()[i]; }
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
    pointer_r( const value_type* ptr ) : base_pointer<T>((T*)ptr) {}
    explicit pointer_r( kaapi_access_t& ptr ) : base_pointer<T>(kaapi_data(value_type, &ptr)) {}
    pointer_r( const pointer_rpwp<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_r( const pointer<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_r( const pointer_rp<T>& ptr ) : base_pointer<T>(ptr) {}
    const T* operator->() { return base_pointer<T>::ptr(); }
    operator value_type*() { return base_pointer<T>::ptr(); }
    const T& operator*() const { return *base_pointer<T>::ptr(); }
    const T* operator->() const { return base_pointer<T>::ptr(); }
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
    T* operator->() { return base_pointer<T>::ptr(); }
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
  template<class T>
  struct TraitNoDeleteTask {
    enum { value = false };
  };

  /* user may specialize this trait to avoid spawn of delete for some object */
  template<> struct TraitNoDeleteTask<char> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<short> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<int> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<long> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<long long> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<unsigned char> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<unsigned short> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<unsigned int> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<unsigned long> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<unsigned long long> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<float> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<double> { enum { value = true}; };

  template<class T>
  class auto_pointer : public pointer<T> {
  public:
    typedef T value_type;
    typedef size_t difference_type;
    typedef auto_pointer<T> Self_t;
    auto_pointer() : pointer<T>() {}
    ~auto_pointer();
    auto_pointer( value_type* ptr ) : pointer<T>(ptr) {}
    operator value_type*() { return pointer<T>::ptr(); }

    KAAPI_POINTER_ARITHMETIC_METHODS
  };


  // --------------------------------------------------------------------
  /** Defined in order to used automatically generated recopy in Universal Access Mode Type constructor :
      - to convert TypeEff -> TypeInTask.
      - and to convert TypeInTask -> TypeFormal.
  */
  struct Access {
    Access( const Access& a ) : a(a.a)
    { }
    template<typename pointer>
    explicit Access( pointer* p )
    { kaapi_access_init(&a, p); }
    template<typename pointer>
    explicit Access( const pointer* p )
    { kaapi_access_init(&a, (void*)p); }
    template<typename T>
    explicit Access( const base_pointer<T>& p )
    { kaapi_access_init(&a, p.ptr()); }
    operator kaapi_access_t&() 
    { return a; }
    kaapi_access_t a;    
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
  template<typename T>
  struct TraitUAMTypeFormat<const T&> { typedef T type_t; };
  template<typename T>
  struct TraitUAMTypeFormat<pointer<T> > { typedef T type_t; };

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
  struct TraitUAMParam<auto_pointer<UserType> > {
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

  /* to be able to use point as arg of spawn */
  template<typename UserType>
  struct TraitUAMParam<const UserType*> {
    typedef TraitUAMType<pointer_rp<UserType> > uamttype_t;
    typedef ACCESS_MODE_RPWP         mode_t;
  };

  /* to be able to use point as arg of spawn */
  template<typename UserType>
  struct TraitUAMParam<UserType*> {
    typedef TraitUAMType<pointer_rpwp<UserType> > uamttype_t;
    typedef ACCESS_MODE_RPWP         mode_t;
  };


  // --------------------------------------------------------------------  
  class DefaultAttribut {
  public:
    kaapi_task_t* operator()( kaapi_thread_t*, kaapi_task_t* clo) const
    { return clo; }
  };
  extern DefaultAttribut SetDefault;
  
  /* */
  class UnStealableAttribut {
  public:
    kaapi_task_t* operator()( kaapi_thread_t*, kaapi_task_t* clo) const
    { 
      //kaapi_task_setflags( clo, KAAPI_TASK_STICKY );
      return clo;
    }
  };
  inline UnStealableAttribut SetUnStealable()
  { return UnStealableAttribut(); }

  /* like default attribut: not yet distributed computation */
  class SetLocalAttribut {
  public:
    kaapi_task_t* operator()( kaapi_thread_t*, kaapi_task_t* clo) const
    { 
      //kaapi_task_setflags( clo, KAAPI_TASK_STICKY );
      return clo; 
    }
  };
  extern SetLocalAttribut SetLocal;

  /* do nothing... not yet distributed implementation */
  class AttributSetPartition {
    int _partition;
  public:
    AttributSetPartition( int s ) : _partition(s) {}
    int get_partition() const { return _partition; }
    kaapi_task_t* operator()( kaapi_threadgroup_t*, kaapi_task_t* clo) const
    { return clo; }
  };

  inline AttributSetPartition SetPartition( int s )
  { return AttributSetPartition(s); }
  
  /* do nothing */
  class SetStaticSchedAttribut {
    int _npart;
    int _niter;
  public:
    SetStaticSchedAttribut( int n, int m  ) 
     : _npart(n), _niter(m) {}
    template<class A1_CLO>
    kaapi_task_t* operator()( kaapi_thread_t*, A1_CLO*& clo) const
    { return clo; }
  };
  inline SetStaticSchedAttribut SetStaticSched(int npart, int iter = 1 )
  { return SetStaticSchedAttribut(npart, iter); }


  // --------------------------------------------------------------------
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

  template <typename type >
  class counting_iterator : public std::iterator< std::random_access_iterator_tag,     /* category */
                                           const type  /* element type */                                            
                                        >
  {
  public:
      typedef type value_type;
      typedef ptrdiff_t difference_type;
      typedef const type& reference;
      typedef const type* pointer;

      counting_iterator()
       : _rep(0)
      {}
      explicit counting_iterator(value_type x) 
       : _rep(x) 
      {}
      value_type const& base() const
      { return _rep; }

      counting_iterator& operator++() 
      { 
        ++_rep;
        return *this;
      }
      counting_iterator operator++(int) 
      { 
        counting_iterator retval = *this;
        ++_rep;
        return retval; 
      }
      counting_iterator& operator--() 
      { 
        --_rep;
        return *this;
      }
      counting_iterator operator--(int) 
      { 
        counting_iterator retval = *this;
        --_rep;
        return retval; 
      }
      difference_type operator-(const counting_iterator& it) const
      { return _rep - it._rep; }
      counting_iterator operator+(value_type v) const
      { return counting_iterator(_rep +v); }
      counting_iterator operator-(value_type v) const
      { return counting_iterator(_rep -v); }
      counting_iterator operator[](int i) const
      { return counting_iterator(_rep +i); }
      bool operator==(const counting_iterator& rhs) 
      { return (_rep==rhs._rep); }

      bool operator!=(const counting_iterator& rhs) 
      { return (_rep!=rhs._rep); }

      reference operator*() 
      { return _rep; }

      pointer* operator->() 
      { return &_rep; }

  private:
      value_type _rep;
  };

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
    static TASK dummy;
    static void body( kaapi_task_t* task, kaapi_stack_t* stack )
    { 
      dummy();
    }
  };
  template<class TASK>
  TASK KaapiTask0<TASK>::dummy;

#include "ka_api_clo.h"

  // --------------------------------------------------------------------
  /* New API: thread.Spawn<TASK>([ATTR])( args )
     Spawn<TASK>([ATTR])(args) with be implemented on top of 
     System::get_current_thread()->Spawn<TASK>([ATTR])( args ).
  */
  class Thread {
  private:
    Thread() {}
  public:

    template<class T>
    T* Alloca(size_t size)
    {
       void* data = kaapi_thread_pushdata( &_thread, sizeof(T)*size );
       return new (data) T[size];
    }

    template<class T>
    T* Alloca()
    {
       void* data = kaapi_thread_pushdata( &_thread, sizeof(T) );
       return new (data) T[1];
    }

    template<class TASK, class Attr>
    class Spawner {
    public:
      Spawner( kaapi_thread_t* t, const Attr& a ) : _thread(t), _attr(a) {}

      /**
      **/      
      void operator()()
      { 
        kaapi_task_t* clo = kaapi_thread_toptask( _thread );
        kaapi_task_initdfg( clo, KaapiTask0<TASK>::body, 0 );
        _attr(_thread, clo);
        kaapi_thread_pushtask( _thread);    
      }

#include "ka_api_spawn.h"

    protected:
      kaapi_thread_t* _thread;
      const Attr&     _attr;
    };

    template<class TASK>
    Spawner<TASK, DefaultAttribut> Spawn() { return Spawner<TASK, DefaultAttribut>(&_thread, DefaultAttribut()); }

    template<class TASK, class Attr>
    Spawner<TASK, Attr> Spawn(const Attr& a) { return Spawner<TASK, Attr>(&_thread, a); }

  protected:
    kaapi_thread_t _thread;
    friend class SyncGuard;
  };


  // --------------------------------------------------------------------
  /* API: 
     * threadgroup.Spawn<TASK>(SetPartition(i) [, ATTR])( args )
     * threadgroup[i]->Spawn<TASK>
  */
  class ThreadGroup {
  private:
    ThreadGroup() {}
  public:

    ThreadGroup(size_t size) 
     : _size(size), _created(false)
    {
    }
    void resize(size_t size)
    { _size = size; }
    
    size_t size() const
    { return _size; }

    /* begin to partition task */
    void begin_partition()
    {
      if (!_created) { kaapi_threadgroup_create( &_threadgroup, _size ); _created = true; }
      kaapi_threadgroup_begin_partition( _threadgroup );
      kaapi_set_threadgroup(_threadgroup);
    }

    /* internal class required for spawn method */
    class AttributComputeDependencies {
    public:
      AttributComputeDependencies( kaapi_threadgroup_t thgrp, int thid ) : _threadgroup(thgrp), _threadindex(thid) {}
      void operator()(kaapi_thread_t* thread, kaapi_task_t* task)
      { kaapi_threadgroup_computedependencies( _threadgroup, _threadindex, task ); }
    public:
      kaapi_threadgroup_t _threadgroup;
      int                 _threadindex;
    };

    /** Spawner for ThreadGroup */
    template<class TASK>
    class Spawner {
    public:
      Spawner( AttributComputeDependencies attr, kaapi_thread_t* t ) 
       : _attr(attr), _thread(t) {}

      /**
      **/      
      void operator()()
      { 
        kaapi_task_t* clo = kaapi_thread_toptask( _thread );
        kaapi_task_initdfg( clo, KaapiTask0<TASK>::body, 0 );
        _attr(_thread, clo);
        kaapi_thread_pushtask( _thread);    
      }

#include "ka_api_spawn.h"

    protected:
      AttributComputeDependencies _attr;
      kaapi_thread_t*             _thread;
    };  
    
    /* Interface: threadgroup.Spawn<TASK>(SetPartition(i) [, ATTR])( args ) */
    template<class TASK>
    Spawner<TASK> Spawn(const AttributSetPartition& a) 
    { return Spawner<TASK>(
                  AttributComputeDependencies(_threadgroup, a.get_partition()),
                  kaapi_threadgroup_thread(_threadgroup, a.get_partition())
              ); 
    }


    /** Executor of one task for ThreadGroup */
    template<class TASKGENERATOR>
    class Executor {
    public:
      Executor(ThreadGroup* thgrp): _threadgroup(thgrp) {}
      /** 0 args **/
      void operator()()
      {
        _threadgroup->begin_partition();
        TASKGENERATOR();
        _threadgroup->end_partition();
        _threadgroup->execute();
      }
#include "ka_api_execgraph.h"
    protected:
      ThreadGroup*  _threadgroup;
    };  
    
    /* Interface: threadgroup.ExecGraph<TASKGENERATOR>()( args ) */
    template<class TASKGENERATOR>
    Executor<TASKGENERATOR> ExecGraph() 
    { return Executor<TASKGENERATOR>( this ); }


    /** ForEach */
    template<class TASKGENERATOR, typename Iterator>
    class ForEachDriver {
    public:
      ForEachDriver(ThreadGroup* thgrp, Iterator beg, Iterator end)
       : _threadgroup(thgrp), _beg(beg), _end(end), step(0), total(0)
      {}
      /** 0 args **/
      void operator()()
      {
        if (_beg == _end) return;
        _threadgroup->begin_partition();
        tpart = kaapi_get_elapsedtime();
        TASKGENERATOR();
        tpart = kaapi_get_elapsedtime()-tpart;
        _threadgroup->end_partition();
        _threadgroup->save();
        while (_beg != _end)
        {
          t0 = kaapi_get_elapsedtime();
          _threadgroup->start_execute();
          _threadgroup->wait_execute();
          t1 = kaapi_get_elapsedtime();
          if (step >0) total += t1-t0;
          std::cout << step << ":: Time: " << t1 - t0 << std::endl;
          ++step;
          if (++_beg != _end) _threadgroup->restore();
        }
        _threadgroup->end_execute(); /* free data structure */
      }
#include "ka_api_execforeach.h"
    protected:
      ThreadGroup*  _threadgroup;
      Iterator      _beg;
      Iterator      _end;
      int           step;
      double        tpart;
      double        t0,t1,total;
    };  

    /** ForEach */
    template<class TASKGENERATOR>
    class ForEachDriverTrampoline {
    public:
      ForEachDriverTrampoline(ThreadGroup* thgrp) : _threadgroup(thgrp) {}
      /** First set of args for the iteration space **/
      template<typename Iterator>
      ForEachDriver<TASKGENERATOR,Iterator> operator()( Iterator beg, Iterator end)
      { 
        return ForEachDriver<TASKGENERATOR, Iterator>( _threadgroup, beg, end );
      }
    protected:
      ThreadGroup*  _threadgroup;
    };  
    
    /* Interface: threadgroup.ExecGraph<TASKGENERATOR>()( args ) */
    template<class TASKGENERATOR>
    ForEachDriverTrampoline<TASKGENERATOR> ForEach() 
    { return ForEachDriverTrampoline<TASKGENERATOR>( this ); }



    /* Interface: threadgroup[i].Spawn<TASK>()( args )
       Il faudrait ici des methods spawn sur un object, cf testspawn.cpp
       + threadgroup[i] == pointeur sur _thread + _threadgroup
    */
    Thread* operator[](int i)    
    {
      kaapi_assert_debug( (i>=0) && (i<(int)_size) );
      return (Thread*)kaapi_threadgroup_thread(_threadgroup, i);
    }

    /* begin to partition task */
    void end_partition()
    {
      kaapi_threadgroup_end_partition( _threadgroup );
      kaapi_set_threadgroup(0);      
    }

    /* execute the threads */
    void execute()
    {
      kaapi_threadgroup_begin_execute( _threadgroup );
      kaapi_threadgroup_end_execute  ( _threadgroup );
    }

    /* asynchronous start */
    void start_execute()
    {
      kaapi_threadgroup_begin_step( _threadgroup );
    }

    /* synchronous call to wait end of execution */
    void wait_execute()
    {
      kaapi_threadgroup_end_step( _threadgroup );
    }

    /* synchronous call say to kernel that this partion is finished to be executed */
    void end_execute()
    {
      kaapi_threadgroup_end_execute( _threadgroup );
    }

    /* save */
    void save()
    {
      kaapi_threadgroup_save( _threadgroup );
    }

    /* restore */
    void restore()
    {
      kaapi_threadgroup_restore( _threadgroup );
    }

    void print()
    { kaapi_threadgroup_print(stdout, _threadgroup); }
  protected:
    size_t              _size;
    bool                _created;
    kaapi_threadgroup_t _threadgroup;
  };

  
  
  // --------------------------------------------------------------------
  /** Top level Spawn */
  template<class TASK>
  Thread::Spawner<TASK, DefaultAttribut> Spawn() 
  { return Thread::Spawner<TASK, DefaultAttribut>(kaapi_self_thread(), DefaultAttribut()); }

  template<class TASK>
  ThreadGroup::Spawner<TASK> Spawn(const AttributSetPartition& a) 
  { return ThreadGroup::Spawner<TASK>(
                ThreadGroup::AttributComputeDependencies(kaapi_self_threadgroup(), a.get_partition()),
                kaapi_threadgroup_thread(kaapi_self_threadgroup(), a.get_partition())
            ); 
  }

  template<class TASK, class Attr>
  Thread::Spawner<TASK, Attr> Spawn(const Attr& a) 
  { return Thread::Spawner<TASK, Attr>(kaapi_self_thread(), a); }


  template<class TASK>
  struct RegisterBodyCPU {
    static void doit() __attribute__((constructor)) 
    {
      static volatile int isinit __attribute__((unused))= DoRegisterBodyCPU<TASK>( &TASK::dummy_method_to_have_formal_param_type ); 
    }
    RegisterBodyCPU() 
    { 
      doit();
    }
  };

  template<class TASK>
  struct RegisterBodyGPU {
    static void doit() __attribute__((constructor))
    { 
      static volatile int isinit __attribute__((unused))= DoRegisterBodyGPU<TASK>( &TASK::dummy_method_to_have_formal_param_type ); 
    }
    RegisterBodyGPU()
    {
      doit();
    }
  };

  template<class TASK>
  struct RegisterBodies {
    static void doit() __attribute__((constructor))
    { 
      static volatile int isinit1 __attribute__((unused))= DoRegisterBodyCPU<TASK>( &TASK::dummy_method_to_have_formal_param_type ); 
      static volatile int isinit2 __attribute__((unused))= DoRegisterBodyGPU<TASK>( &TASK::dummy_method_to_have_formal_param_type ); 
    }
    RegisterBodies()
    {
      doit();
    }  
  };

  // --------------------------------------------------------------------
  /** Wait execution of all forked tasks of the running task */
  extern void Sync();



  // --------------------------------------------------------------------
  /* Main task */
  template<class TASK>
  struct MainTaskBodyArgcv {
    static void body( int argc, char** argv )
    {
      TASK()( argc, argv );
    }
  };
  template<class TASK>
  struct MainTaskBodyNoArgcv {
    static void body( int argc, char** argv )
    {
      TASK()( );
    }
  };
  
  template<class TASK>
  struct SpawnerMain
  {
    SpawnerMain() 
    { }

    void operator()( int argc, char** argv)
    {
      kaapi_thread_t* thread = kaapi_self_thread();
      kaapi_task_t* clo = kaapi_thread_toptask( thread );
      kaapi_task_initdfg( clo, kaapi_taskmain_body, kaapi_thread_pushdata(thread, sizeof(kaapi_taskmain_arg_t)) );
      kaapi_taskmain_arg_t* arg = kaapi_task_getargst( clo, kaapi_taskmain_arg_t);
      arg->argc = argc;
      arg->argv = argv;
      arg->mainentry = &MainTaskBodyArgcv<TASK>::body;
      kaapi_thread_pushtask( thread);    
    }

    void operator()( )
    {
      kaapi_thread_t* thread = kaapi_self_thread();
      kaapi_task_t* clo = kaapi_thread_toptask( thread );
      kaapi_task_initdfg( clo, kaapi_taskmain_body, kaapi_thread_pushdata(thread, sizeof(kaapi_taskmain_arg_t)) );
      kaapi_taskmain_arg_t* arg = kaapi_task_getargst( clo, kaapi_taskmain_arg_t);
      arg->argc = 0;
      arg->argv = 0;
      arg->mainentry = &MainTaskBodyNoArgcv<TASK>::body;
      kaapi_thread_pushtask( thread );    
    }
  };

  template<class TASK>
  SpawnerMain<TASK> SpawnMain()
  { 
    return SpawnerMain<TASK>();
  }
    

  // --------------------------------------------------------------------
  template<class T>
  struct TaskDelete : public Task<1>::Signature<RW<T> > { };
} // namespace a1

  template<class T>
  struct TaskBodyCPU<ka::TaskDelete<T> > : public ka::TaskDelete<T> {
    void operator() ( ka::Thread* thread, ka::pointer_rw<T> res )
    {
      delete *res;
    }
  };
  
namespace ka {
  
  template<bool noneedtaskdelete>
  struct SpawnDelete {
    template<class T> static void doit( auto_pointer<T>& ap ) { Spawn<TaskDelete<T> >(ap); ap.ptr(0); }
  };
  template<>
  struct SpawnDelete<true> {
    template<class T> static void doit( auto_pointer<T>& ap ) 
    { 
#if !defined(KAAPI_NDEBUG)
      ap.ptr(0);
#endif
    }
  };

  template<class T>
  auto_pointer<T>::~auto_pointer()
  { SpawnDelete<TraitNoDeleteTask<T>::value>::doit(*this); }


  // --------------------------------------------------------------------
  extern std::ostream& logfile();

  // --------------------------------------------------------------------
  class SyncGuard {
    kaapi_thread_t*         _thread;
    kaapi_frame_t           _frame;
  public:
    SyncGuard();

    ~SyncGuard();
  };
  
  // --------------------------------------------------------------------
  // Should be defined for real distributed computation
  class OStream {
  public:
    enum Mode {
      IA = 1,
      DA = 2
    };
    size_t write( const Format*, int, const void*, size_t ) { return 0; }
  };

  class IStream {
  public:
    enum Mode {
      IA = 1,
      DA = 2
    };
    size_t read( const Format*, int, void*, size_t ) { return 0; }
  };
  
  // --------------------------------------------------------------------
  struct InitKaapiCXX {
    InitKaapiCXX();
  };
  static InitKaapiCXX stroumph;
  
} // namespace ka


/* compatibility toolkit */
inline ka::OStream& operator<< (ka::OStream& s_out, char c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, short c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, int c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, long c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, long long c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, unsigned char c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, unsigned short c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, unsigned int c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, unsigned long c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, unsigned long long c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, float c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, double c )
{ return s_out; }

inline ka::IStream& operator>> (ka::IStream& s_in, char& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, short& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, int& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, long& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, long long& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, unsigned char& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, unsigned short& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, unsigned int& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, unsigned long& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, unsigned long long& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, float& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, double& c )
{ return s_in; }


#if !defined(_KAAPIPLUSPLUS_NOT_IN_GLOBAL_NAMESPACE)
using namespace ka;
#endif

#endif

