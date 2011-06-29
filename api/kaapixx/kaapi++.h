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
#ifndef _KAAPI_CPP_H_
#define _KAAPI_CPP_H_

#include "kaapi.h"
#include "ka_error.h"
#include "ka_timer.h"
#include <vector>
#include <typeinfo>
#include <iterator>
#include <stdexcept>

/** Version number for the API
    - v1: new API for DFG with pointer interface.
*/
#define KAAPIXX_API_VERSION 1
namespace ka {

  /** Log file for this process
      \ingroup atha
      If not set_logfile, then return the std::cout stream.
  */
  extern std::ostream& logfile();

//@{
/** Return an system wide identifier of the type of an expression
*/
#define kaapi_get_swid(EXPR) kaapi_hash_value(typeid(EXPR).name())
//@}

  /* take a constant... should be adjusted */
  enum { STACK_ALLOC_THRESHOLD = KAAPI_MAX_DATA_ALIGNMENT };  

  // --------------------------------------------------------------------
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
  
  /* Kaapi C++ StealContext <-> Kaapi C kaapi_stealcontext_t */
  class StealContext;

  /* Kaapi C++ Request <-> Kaapi C kaapi_request_t */
  class Request;

  /* for next networking part */
  class IStream;
  class OStream;
  class ODotStream;
  class SyncGuard;
  

  // --------------------------------------------------------------------
  class gpuStream {
  public:
    gpuStream( kaapi_gpustream_t gs ): stream(gs) {}
    gpuStream( ): stream(0) {}
    kaapi_gpustream_t stream; /* at least for CUDA */
  };


  // --------------------------------------------------------------------
  /** link C++ format -> kaapi format */
  class Format {
  public:
    Format( 
        const char*        name = "empty",
        size_t             size = 0,
        void             (*cstor)( void* dest) = 0,
        void             (*dstor)( void* dest) = 0,
        void             (*cstorcopy)( void* dest, const void* src) = 0,
        void             (*copy)( void* dest, const void* src) = 0,
        void             (*assign)( void*, const kaapi_memory_view_t*, const void*, const kaapi_memory_view_t*) =0,
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
      const char* name,
      int (*update_mb)(void* data, const struct kaapi_format_t* fmtdata,
                       const void* value, const struct kaapi_format_t* fmtvalue )
    );
  };

  // --------------------------------------------------------------------  
  /** format for all kind of task */
  class FormatTask : public Format {
  public:
    /* task with static format */
    FormatTask( 
      const char*                 name,
      size_t                      size,
      int                         count,
      const kaapi_access_mode_t   mode_param[],
      const kaapi_offset_t        offset_param[],
      const kaapi_offset_t        offset_version[],
      const kaapi_format_t*       fmt_param[],
      const kaapi_memory_view_t   view_param[],
      const kaapi_reducor_t       reducor_param[],
      const kaapi_redinit_t       redinit_param[],
      const kaapi_task_binding_t* task_bind
    );

    /* task with dynamic format */
    FormatTask( 
      const char*                 name,
      size_t                      size,
      size_t                    (*get_count_params)(const struct kaapi_format_t*, const void*),
      kaapi_access_mode_t       (*get_mode_param)  (const struct kaapi_format_t*, unsigned int, const void*),
      void*                     (*get_off_param)   (const struct kaapi_format_t*, unsigned int, const void*),
      kaapi_access_t            (*get_access_param)(const struct kaapi_format_t*, unsigned int, const void*),
      void                      (*set_access_param)(const struct kaapi_format_t*, unsigned int, void*, const kaapi_access_t*),
      const struct kaapi_format_t*(*get_fmt_param) (const struct kaapi_format_t*, unsigned int, const void*),
      kaapi_memory_view_t       (*get_view_param)  (const struct kaapi_format_t*, unsigned int, const void*),
      void                      (*set_view_param)  (const struct kaapi_format_t*, unsigned int, void*, const kaapi_memory_view_t*),
      void                      (*reducor )        (const struct kaapi_format_t*, unsigned int, void*, const void*),
      void                      (*redinit )        (const struct kaapi_format_t*, unsigned int, const void*, void*),
      void                      (*get_task_binding)(const struct kaapi_format_t*, const kaapi_task_t*, kaapi_task_binding_t*)
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
    static void assign( 
            void* dest, const kaapi_memory_view_t* view_dest, 
            const void* src, const kaapi_memory_view_t* view_src) 
    { 
      T* d = (T*)dest; T* s = (T*)src; 
#if !defined(KAAPI_NDEBUG)
      if (view_dest->type != view_src->type)
      {
        std::cerr << "*** Format, bad assignment of views with different types " << std::endl;
        exit(1);
      }
#endif
      if (kaapi_memory_view_iscontiguous(view_dest) && kaapi_memory_view_iscontiguous(view_src))
      {
        /* size in byte */
        size_t size = kaapi_memory_view_size(view_dest) / sizeof(T);
        for (size_t i=0; i < size; ++i)
          d[i] = s[i];
      }
      else if (view_src->type == KAAPI_MEMORY_VIEW_2D)
      {
        /* size per dimension is in sizeof(T) unit */
        size_t size_i = view_src->size[0];
        size_t size_j = view_src->size[1];
        for (size_t i=0; i < size_i; ++i)
        {
          for (size_t j=0; j < size_j; ++j)
            d[j] = s[j];
          d += view_dest->lda;
          s += view_src->lda;
        }
      }
      else {
        std::cerr << "*** Format, bad assignment of unimplemented views" << std::endl;
        exit(1);
      }
    } 
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
    static Community join_community()
      throw (std::runtime_error);

    static Community join_community( int& argc, char**& argv )
      throw (std::runtime_error);

    static Community initialize_community( int& argc, char**& argv )
      throw (std::runtime_error);

    static Community initialize( int& argc, char**& argv )
      throw (std::runtime_error)
    { return System::initialize_community(argc, argv); }

    static Thread* get_current_thread();

    static int getRank();

    static void terminate();


    /** The global id of this process
    */
    static uint32_t local_gid;
  public:
    static int saved_argc;
    static char** saved_argv;
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
  /* type for nothing */
  struct THIS_TYPE_IS_USED_ONLY_INTERNALLY  { };

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
  
  template<class Mode>
  struct TYPEMODE2VALUE{};

  template<> struct TYPEMODE2VALUE<ACCESS_MODE_V>    { enum { value = ACCESS_MODE_V::value}; };
  template<> struct TYPEMODE2VALUE<ACCESS_MODE_R>    { enum { value = ACCESS_MODE_R::value}; };
  template<> struct TYPEMODE2VALUE<ACCESS_MODE_W>    { enum { value = ACCESS_MODE_W::value}; };
  template<> struct TYPEMODE2VALUE<ACCESS_MODE_RW>   { enum { value = ACCESS_MODE_RW::value}; };
  template<> struct TYPEMODE2VALUE<ACCESS_MODE_CW>   { enum { value = ACCESS_MODE_CW::value}; };
  template<> struct TYPEMODE2VALUE<ACCESS_MODE_RP>   { enum { value = ACCESS_MODE_RP::value}; };
  template<> struct TYPEMODE2VALUE<ACCESS_MODE_WP>   { enum { value = ACCESS_MODE_WP::value}; };
  template<> struct TYPEMODE2VALUE<ACCESS_MODE_RPWP> { enum { value = ACCESS_MODE_RPWP::value}; };
  template<> struct TYPEMODE2VALUE<ACCESS_MODE_CWP>  { enum { value = ACCESS_MODE_CWP::value}; };

  template<class Mode> 
  struct IsAccessMode { static const bool value = true; };
  template<> struct IsAccessMode<ACCESS_MODE_V> { static const bool value = false; };

  struct TYPE_INTASK {}; /* internal purpose to define representation of a type in a task */
  struct TYPE_INPROG {}; /* internal purpose to define representation of a type in the user program */

  template<class T>
  struct DefaultAdd {
    void operator()( T& result, const T& value)
    { result += value; }
  };
  
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
  template<class T, typename OP=DefaultAdd<T> >
  class pointer_cw;


  // --------------------------------------------------------------------
  template<class T> 
  struct TraitIsStatic { static const bool value = true; };

} // end of namespace ka

#include "ka_api_array.h"

namespace ka {

  template<int dim, class T, Storage2DClass S> 
  struct TraitIsStatic<array<dim,T,S> > { static const bool value = false; };

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

  /* capture cumulative write */
  template<class T,class OP>
  class cumul_value_ref {
  public:
    cumul_value_ref( pointer_cw<T,OP>& p ) : _ptr(p) {}
    void operator+=( const T& value );
    template<typename OP2>
    void cumul( const T& value );
  protected:
    pointer_cw<T,OP>& _ptr; /* the order of field should correspond to order of pointer_cw */
  };
  

#define KAAPI_POINTER_ARITHMETIC_METHODS\
    Self_t& operator++() { ++base_pointer<T>::_ptr; return *this; }\
    Self_t operator++(int) { return base_pointer<T>::_ptr++; }\
    Self_t& operator--() { --base_pointer<T>::_ptr; return *this; }\
    Self_t operator--(int) { return base_pointer<T>::_ptr--; }\
    Self_t operator+(int i) const { return base_pointer<T>::_ptr+i; }\
    Self_t operator+(unsigned int i) const { return base_pointer<T>::_ptr+i; }\
    Self_t operator+(long i) const { return base_pointer<T>::_ptr+i; }\
    Self_t operator+(unsigned long i) const { return base_pointer<T>::_ptr+i; }\
    Self_t& operator+=(int i) { base_pointer<T>::_ptr+=i; return *this; }\
    Self_t& operator+=(unsigned int i) { base_pointer<T>::_ptr+=i; return *this; }\
    Self_t& operator+=(long i) { base_pointer<T>::_ptr+=i; return *this; }\
    Self_t& operator+=(unsigned long i) { base_pointer<T>::_ptr+=i; return *this; }\
    Self_t operator-(int i) const { return base_pointer<T>::_ptr-i; }\
    Self_t operator-(unsigned int i) const { return base_pointer<T>::_ptr-i; }\
    Self_t operator-(long i) const { return base_pointer<T>::_ptr-i; }\
    Self_t operator-(unsigned long i) const { return base_pointer<T>::_ptr-i; }\
    Self_t& operator-=(int i) { return base_pointer<T>::_ptr-=i; }\
    Self_t& operator-=(unsigned int i) { return base_pointer<T>::_ptr-=i; }\
    Self_t& operator-=(long i) { return base_pointer<T>::_ptr-=i; }\
    Self_t& operator-=(unsigned long i) { return base_pointer<T>::_ptr-=i; }\
    difference_type operator-(const Self_t& p) const { return base_pointer<T>::_ptr-p._ptr; }\
    bool operator==(const Self_t& p) const { return base_pointer<T>::_ptr == p._ptr; }\
    bool operator!=(const Self_t& p) const { return base_pointer<T>::_ptr != p._ptr; }


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
//    operator value_type*() { return base_pointer<T>::ptr(); }

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
//    operator value_type*() { return base_pointer<T>::ptr(); }

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
//    operator value_type*() { return base_pointer<T>::ptr(); }

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
    operator const value_type*() const { return base_pointer<T>::ptr(); }
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
//    operator value_type*() { return base_pointer<T>::ptr(); }

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

    KAAPI_POINTER_ARITHMETIC_METHODS
  };


  // --------------------------------------------------------------------  
  template<class T, typename OP >
  class pointer_cw: public base_pointer<T> {
    friend class cumul_value_ref<T,OP>;
  public:
    typedef T value_type;
    typedef size_t difference_type;
    typedef pointer_cw<T> Self_t;

    pointer_cw() : base_pointer<T>() {}
    pointer_cw( value_type* ptr ) : base_pointer<T>(ptr) {}
    /* to call user define task */
    explicit pointer_cw( kaapi_access_t& ptr ) : base_pointer<T>(kaapi_data(value_type, &ptr)) {}
    pointer_cw( const pointer_rpwp<T>& ptr ) : base_pointer<T>(ptr) {}
    pointer_cw( const pointer_cwp<T>& ptr ) : base_pointer<T>(ptr)  {}
    template<class OP2>
    pointer_cw( const pointer_cw<T, OP2>& ptr ) : base_pointer<T>(ptr) {}

    /* be carrefull here: pointer_cw and cumul_value_ref should be identic */
    cumul_value_ref<T,OP> operator*() 
    { return cumul_value_ref<T,OP>(*this); }
    cumul_value_ref<T,OP>* operator->() 
    { cumul_value_ref<T,OP>(*this); }

    KAAPI_POINTER_ARITHMETIC_METHODS
  };

  template<class T, typename OP >
  void cumul_value_ref<T,OP>::operator+=( const T& value ) 
  { 
    *_ptr.ptr() += value; 
  }
  template<class T, typename OP >
  template<typename OP2>
  void cumul_value_ref<T,OP>::cumul( const T& value ) 
  {
    OP2()(*_ptr.ptr(),value);
  }


  // --------------------------------------------------------------------  
  /* here requires to distinguish pointer to object from pointer to function */
  template<class R> struct __kaapi_is_function { enum { value = false }; };
  template<class R> struct __kaapi_is_function<R (*)()> { enum { value = true }; };
  template<class R> struct __kaapi_is_function<R (*)(...)> { enum { value = true }; };
  template<class R, class T0> 
  struct __kaapi_is_function<R (*)(T0)> { enum { value = true }; };
  template<class R, class T0, class T1> 
  struct __kaapi_is_function<R (*)(T0, T1)> { enum { value = true }; };
  template<class R, class T0, class T1, class T2> 
  struct __kaapi_is_function<R (*)(T0, T1, T2)> { enum { value = true }; };
  template<class R, class T0, class T1, class T2, class T3> 
  struct __kaapi_is_function<R (*)(T0, T1, T2, T3)> { enum { value = true }; };
  template<class R, class T0, class T1, class T2, class T3, class T4>
  struct __kaapi_is_function<R (*)(T0, T1, T2, T3, T4)> { enum { value = true }; };
  template<class R, class T0, class T1, class T2, class T3, class T4, class T5>
  struct __kaapi_is_function<R (*)(T0, T1, T2, T3, T4, T5)> { enum { value = true }; };
  template<class R, class T0, class T1, class T2, class T3, class T4, class T5, class T6>
  struct __kaapi_is_function<R (*)(T0, T1, T2, T3,T4, T5, T6)> { enum { value = true }; };
  template<class R, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7> 
  struct __kaapi_is_function<R (*)(T0, T1, T2, T3, T4, T5, T6, T7)> { enum { value = true }; };
  template<class R, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8> 
  struct __kaapi_is_function<R (*)(T0, T1, T2, T3, T4, T5, T6, T7, T8)> { enum { value = true }; };
  template<class R, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9> 
  struct __kaapi_is_function<R (*)(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9)> { enum { value = true }; };
  template<class R, class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10> 
  struct __kaapi_is_function<R (*)(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10)> { enum { value = true }; };


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
  template<> struct TraitNoDeleteTask<unsigned char> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<unsigned short> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<unsigned int> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<unsigned long> { enum { value = true}; };
#if defined(__APPLE__) && defined(__ppc__) && defined(__GNUC__)
#else  
  template<> struct TraitNoDeleteTask<unsigned long long> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<long long> { enum { value = true}; };
#endif
  template<> struct TraitNoDeleteTask<float> { enum { value = true}; };
  template<> struct TraitNoDeleteTask<double> { enum { value = true}; };

  /* autopointer: like a pointer, except that it spawn delete task
     at the end of the definition cope
  */
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
    operator const value_type*() const { return pointer<T>::ptr(); }

    KAAPI_POINTER_ARITHMETIC_METHODS
  };

  template<class T>
  class auto_variable {
    T* _var;
  public:
    typedef T value_type;
    typedef auto_variable<T> Self_t;
    auto_variable() 
    {
       void* data = kaapi_thread_pushdata( kaapi_self_thread(), sizeof(T) );
       _var = new (data) T;
    }
    auto_variable(const T& value) 
    {
       void* data = kaapi_thread_pushdata( kaapi_self_thread(), sizeof(T) );
       _var = new (data) T(value);
    }
    auto_variable( Thread* thread ) 
    {
       void* data = kaapi_thread_pushdata( (kaapi_thread_t*)thread, sizeof(T) );
       _var = new (data) T;
    }
    auto_variable( Thread* thread, const T& value) 
    {
       void* data = kaapi_thread_pushdata( (kaapi_thread_t*)thread, sizeof(T) );
       _var = new (data) T(value);
    }
    ~auto_variable();
    operator const T&() const { return *_var; }
    operator T&() { return *_var; }
    T& operator=( const T& value) { *_var = value; return *_var; }
    T* operator&() { return _var; }

    /* clear variable: used for debug only */
    void clear() { _var = 0; }    
  };


  // --------------------------------------------------------------------
  /** Defined in order to used automatically generated recopy in Universal Access Mode Type constructor :
      - to convert TypeEff -> TypeInTask.
      - and to convert TypeInTask -> TypeFormal.
  */
  struct Access : public kaapi_access_t {
    Access( const Access& access ) : kaapi_access_t(access)
    { }
    template<typename T>
    explicit Access( const base_pointer<T>& p )
    { kaapi_access_init(this, p.ptr()); }

    template<typename pointer>
    explicit Access( pointer* p )
    { kaapi_access_init(this, p); }

    template<typename pointer>
    explicit Access( const pointer* p )
    { kaapi_access_init(this, (void*)p); }

    operator kaapi_access_t&() 
    { return *this; }
    void operator=( const kaapi_access_t& a)
    { data = a.data; version = a.version; }
  };
  
  // --------------------------------------------------------------------
  /* Helpers to declare type in signature of task */
  template<typename UserType=void> struct Value {};
  template<typename UserType=void> struct RPWP {};
  template<typename UserType=void> struct RP {};
  template<typename UserType=void> struct R  {};
  template<typename UserType=void> struct WP {};
  template<typename UserType=void> struct W {};
  template<typename UserType=void> struct RW {};
  template<typename UserType=void, typename OpCumul = DefaultAdd<UserType> > struct CW {};
  template<typename UserType=void> struct CWP {};


  // --------------------------------------------------------------------
  /** Trait to encode operations / types required to spawn task
       *  TraitFormalParam<T>::type_t gives type of the underlaying C++ object. If not pointer, this is T
       *  TraitFormalParam<T>::formal_t gives type of the formal parameter
       *  TraitFormalParam<T>::mode_t gives type of the access mode of T
       *  TraitFormalParam<T>::type_inclosure_t gives type of C++ object store in the task argument
       *  TraitFormalParam<T>::is_static retuns true or false if the type does not contains a dynamic set 
       of pointer access.
       *  TraitFormalParam<T>::handl2data used when pointer to data for shared object are pointer to pointer
       to data. Should return a type formal_t after interpretation of the type in closure.
       
       *  TraitFormalParam<T>::get_data return the address in the task argument data structure of 
       the i-th data parameter
       *  TraitFormalParam<T>::get_version return the address in the task argument data structure of 
       the i-th version parameter
       *  TraitFormalParam<T>::get_nparam( typeinclosure* ): size_t
       *  TraitFormalParam<T>::is_access( typeinclosure* ): bool
       
      During the creation of task with a formal parameter fi of type Fi, then the task argument data structure
      stores an object of type TraitFormalParam<T>::type_inclosure_t called taskarg->fi.
      Then, to bind the effective parameter ei of type Ei to fi:
      1- the effective argument ei is bind to the type into the closure:
        new (&taskarg->fi) TraitFormalParam<T>::type_inclosure_t( ei )
      2- the object store into the task argument data structure is binded to the formal parameter during
        bootstrap code to execute a task body: taskbody( ..., taskarg->fi, ... )
        It means that the type to the closure should be convertible to the type of the formal parameter.        
  */
  template<typename T>
  struct TraitFormalParam { 
    typedef T                type_t; 
    typedef T                signature_t; 
    typedef T                formal_t; 
    typedef ACCESS_MODE_V    mode_t; 
    typedef T                type_inclosure_t; 
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const T* a ) { return 0; }
    static T&                handle2data( type_inclosure_t* a) { return *a; }
    static void*             get_data   ( type_inclosure_t* a, unsigned int i ) { return a; }
    static void*             get_version( type_inclosure_t* a, unsigned int i ) { return 0; }
    static void              get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) {}
    static void              set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) {}
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
    static void              reducor_fnc(void*, const void*) {}
    static void              redinit_fnc(void*) {}
  };

  template<typename T>
  struct TraitFormalParam<Value<T> > { 
    typedef T                type_t; 
    typedef T                signature_t; 
    typedef T                formal_t; 
    typedef ACCESS_MODE_V    mode_t; 
    typedef T                type_inclosure_t; 
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const T* a ) { return 0; }
    static T&                handle2data( type_inclosure_t* a) { return *a; }
    static void*             get_data   ( type_inclosure_t* a, unsigned int i ) { return a; }
    static void*             get_version( type_inclosure_t* a, unsigned int i ) { return 0; }
    static void              get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) {}
    static void              set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) {}
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
    static void              reducor_fnc(void*, const void*) {}
    static void              redinit_fnc(void*) {}
  };

  template<typename T>
  struct TraitFormalParam<const T&> { 
    typedef T                type_t; 
    typedef T                signature_t; 
    typedef const T&         formal_t; 
    typedef ACCESS_MODE_V    mode_t; 
    typedef T                type_inclosure_t; 
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const T* a ) { return 0; }
    static const T&          handle2data( type_inclosure_t* a) { return *a; }
    static void*             get_data   ( type_inclosure_t* a, unsigned int i ) { return a; }
    static void*             get_version( type_inclosure_t* a, unsigned int i ) { return 0; }
    static void              get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) {}
    static void              set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) {}
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
    static void              reducor_fnc(void*, const void*) {}
    static void              redinit_fnc(void*) {}
  };

  template<typename T>
  struct TraitFormalParam<const T> { 
    typedef T                type_t; 
    typedef const T          signature_t; 
    typedef const T          formal_t; 
    typedef ACCESS_MODE_V    mode_t; 
    typedef T                type_inclosure_t; 
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const T* a ) { return 0; }
    static const T&          handle2data( type_inclosure_t* a) { return *a; }
    static void*             get_data   ( type_inclosure_t* a, unsigned int i ) { return a; }
    static void*             get_version( type_inclosure_t* a, unsigned int i ) { return 0; }
    static void              get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) {}
    static void              set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) {}
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static void              reducor_fnc(void*, const void*) {}
    static void              redinit_fnc(void*) {}
  };

  template<typename T>
  struct TraitFormalParam<pointer<T> > { 
    typedef T                type_t; 
    typedef RPWP<T>          signature_t; 
    typedef ACCESS_MODE_RPWP mode_t; 
    typedef Access           type_inclosure_t; 
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const pointer<T>* a ) { return a->ptr(); }
    static const void*       get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->data; }
    static const void*       get_version( const type_inclosure_t* a, unsigned int i ) { return &a->version; }
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t  get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
  };

  template<typename T>
  struct TraitFormalParam<auto_pointer<T> > { 
    typedef T                type_t; 
    typedef RPWP<T>          signature_t; 
    typedef ACCESS_MODE_RPWP mode_t; 
    typedef Access           type_inclosure_t; 
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const auto_pointer<T>* a ) { return a->ptr(); }
    static const void*       get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->data; }
    static const void*       get_version( const type_inclosure_t* a, unsigned int i ) { return &a->version; }
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t  get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
  };

  template<typename T>
  struct TraitFormalParam<pointer_r<T> > { 
    typedef T                type_t; 
    typedef R<T>             signature_t; 
    typedef pointer_r<T>     formal_t; 
    typedef ACCESS_MODE_R    mode_t; 
    typedef Access           type_inclosure_t;  /* could be only one pointer without version */
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const pointer_r<T>* a ) { return a->ptr(); }
    static pointer_r<T>      handle2data( type_inclosure_t* a) 
    { return (T*)__kaapi_pointer2void((kaapi_handle_t(a->data))->ptr); }
    static const void*       get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->data; }
    static const void*       get_version( const type_inclosure_t* a, unsigned int i ) { return &a->version; }
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
    static void              get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) { *r = *a; }
    static void              set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) { *a = *r; }
    static void              reducor_fnc(void*, const void*) {}
    static void              redinit_fnc(void*) {}
  };

  template<typename T>
  struct TraitFormalParam<pointer_rp<T> > { 
    typedef T                type_t; 
    typedef RP<T>            signature_t; 
    typedef pointer_rp<T>    formal_t; 
    typedef ACCESS_MODE_RP   mode_t; 
    typedef Access           type_inclosure_t;  /* could be only one pointer without version */
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const pointer_rp<T>* a ) { return a->ptr(); }
    static pointer_rp<T>     handle2data( type_inclosure_t* a)
    { return (T*)__kaapi_pointer2void((kaapi_handle_t(a->data))->ptr); }
    static const void*       get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->data; }
    static const void*       get_version( const type_inclosure_t* a, unsigned int i ) { return &a->version; }
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
    static void              get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) { *r = *a; }
    static void              set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) { *a = *r; }
    static void              reducor_fnc(void*, const void*) {}
    static void              redinit_fnc(void*) {}
  };

  template<typename T>
  struct TraitFormalParam<pointer_w<T> > { 
    typedef T                type_t; 
    typedef W<T>             signature_t; 
    typedef pointer_w<T>     formal_t; 
    typedef ACCESS_MODE_W    mode_t; 
    typedef Access           type_inclosure_t;  /* could be only one pointer without version */
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const pointer_w<T>* a ) { return a->ptr(); }
    static pointer_w<T>      handle2data( type_inclosure_t* a) 
    { return (T*)__kaapi_pointer2void((kaapi_handle_t(a->data))->ptr); }
    static const void*       get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->data; }
    static const void*       get_version( const type_inclosure_t* a, unsigned int i ) { return &a->version; }
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
    static void              get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) { *r = *a; }
    static void              set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) { *a = *r; }
    static void              reducor_fnc(void*, const void*) {}
    static void              redinit_fnc(void*) {}
  };

  template<typename T>
  struct TraitFormalParam<pointer_wp<T> > { 
    typedef T                type_t; 
    typedef WP<T>            signature_t; 
    typedef pointer_wp<T>    formal_t; 
    typedef ACCESS_MODE_WP   mode_t; 
    typedef Access           type_inclosure_t;  /* could be only one pointer without version */
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const pointer_wp<T>* a ) { return a->ptr(); }
    static pointer_wp<T>     handle2data( type_inclosure_t* a)
    { return (T*)__kaapi_pointer2void((kaapi_handle_t(a->data))->ptr); }
    static const void*       get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->data; }
    static const void*       get_version( const type_inclosure_t* a, unsigned int i ) { return &a->version; }
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
    static void              get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) { *r = *a; }
    static void              set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) { *a = *r; }
    static void              reducor_fnc(void*, const void*) {}
    static void              redinit_fnc(void*) {}
  };

  template<typename T>
  struct TraitFormalParam<pointer_rw<T> > { 
    typedef T                type_t; 
    typedef RW<T>            signature_t; 
    typedef pointer_rw<T>    formal_t; 
    typedef ACCESS_MODE_RW   mode_t; 
    typedef Access           type_inclosure_t;  /* could be only one pointer without version */
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const pointer_rw<T>* a ) { return a->ptr(); }
    static pointer_rw<T>     handle2data( type_inclosure_t* a) 
    { return (T*)__kaapi_pointer2void((kaapi_handle_t(a->data))->ptr); }
    static const void*       get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->data; }
    static const void*       get_version( const type_inclosure_t* a, unsigned int i ) { return &a->version; }
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
    static void              get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) { *r = *a; }
    static void              set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) { *a = *r; }
    static void              reducor_fnc(void*, const void*) {}
    static void              redinit_fnc(void*) {}
  };

  template<typename T>
  struct TraitFormalParam<pointer_rpwp<T> > {
    typedef T                type_t; 
    typedef RPWP<T>          signature_t; 
    typedef pointer_rpwp<T>  formal_t; 
    typedef ACCESS_MODE_RPWP mode_t; 
    typedef Access           type_inclosure_t; 
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const pointer_rpwp<T>* a ) { return a->ptr(); }
    static pointer_rpwp<T>   handle2data( type_inclosure_t* a)
    { return (T*)__kaapi_pointer2void((kaapi_handle_t(a->data))->ptr); }
    static const void*       get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->data; }
    static const void*       get_version( const type_inclosure_t* a, unsigned int i ) { return &a->version; }
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
    static void              get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) { *r = *a; }
    static void              set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) { *a = *r; }
    static void              reducor_fnc(void*, const void*) {}
    static void              redinit_fnc(void*) {}
  };  

 template<typename T, typename OP>
  struct TraitFormalParam<pointer_cw<T,OP> > { 
    typedef T                type_t; 
    typedef CW<T,OP>         signature_t; 
    typedef pointer_cw<T,OP> formal_t; 
    typedef ACCESS_MODE_CW   mode_t; 
    typedef Access           type_inclosure_t;  /* could be only one pointer without version */
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const pointer_cw<T>* a ) { return a->ptr(); }
    static pointer_cw<T,OP>  handle2data( type_inclosure_t* a)
    { return (T*)__kaapi_pointer2void((kaapi_handle_t(a->data))->ptr); }

    static const void*       get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->data; }
    static const void*       get_version( const type_inclosure_t* a, unsigned int i ) { return &a->version; }
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
    static void              get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) { *r = *a; }
    static void              set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) { *a = *r; }
    static void              reducor_fnc(void* result, const void* value) 
    { T* r = static_cast<T*> (result); 
      const T* v = static_cast<const T*> (value);
      OP()(*r, *v);
    }
    static void              redinit_fnc(void* value) 
    { 
      new (value)T;
    }
  };

  template<typename T>
  struct TraitFormalParam<pointer_cwp<T> > {
    typedef T                type_t; 
    typedef CWP<T>           signature_t; 
    typedef pointer_cwp<T>   formal_t; 
    typedef ACCESS_MODE_CWP  mode_t; 
    typedef Access           type_inclosure_t; 
    static const bool        is_static = TraitIsStatic<T>::value;
    static const void*       ptr( const pointer_cwp<T>* a ) { return a->ptr(); }
    static pointer_cwp<T>    handle2data( type_inclosure_t* a)
    { return (T*)__kaapi_pointer2void((kaapi_handle_t(a->data))->ptr); }
    static const void*       get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->data; }
    static const void*       get_version( const type_inclosure_t* a, unsigned int i ) { return &a->version; }
    static size_t            get_nparam ( const type_inclosure_t* a ) { return 1; }
    static kaapi_memory_view_t get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return kaapi_memory_view_make1d( 1, sizeof(type_inclosure_t) ); }
    static void              set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { }
    static void              get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) { *r = *a; }
    static void              set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) { *a = *r; }
    static void              reducor_fnc(void*, const void*) {}
    static void              redinit_fnc(void* value) { }
  }; 

  template<typename T>
  struct TraitFormalParam<W<T> > : public TraitFormalParam<pointer_w<T> > { };
  template<typename T>
  struct TraitFormalParam<R<T> > : public TraitFormalParam<pointer_r<T> > { };
  template<typename T>
  struct TraitFormalParam<WP<T> > : public TraitFormalParam<pointer_wp<T> > { };
  template<typename T>
  struct TraitFormalParam<RP<T> > : public TraitFormalParam<pointer_rp<T> > { };
  template<typename T>
  struct TraitFormalParam<RW<T> > : public TraitFormalParam<pointer_rw<T> > { };
  template<typename T>
  struct TraitFormalParam<RPWP<T> > : public TraitFormalParam<pointer_rpwp<T> > { };
  template<typename T>
  struct TraitFormalParam<CWP<T> > : public TraitFormalParam<pointer_cwp<T> > { };
  template<typename T, typename OP>
  struct TraitFormalParam<CW<T,OP> > : public TraitFormalParam<pointer_cw<T,OP> > { };


  /* ------ rep of range of contiguous array of data: pointer_XX<array<dim, T> >
  */
  template<int d, typename T, Storage2DClass S>
  struct array_inclosure_t : public array<d,T,S> {
    array_inclosure_t() : array<d,T,S>(), version(0) {}
    array_inclosure_t(const array<d,T,S>& a) : array<d,T,S>(a), version(0) {}
    template<typename P>
    array_inclosure_t( const P& p ) : array<d,T,S>(p) {}
    
    void* version;
  };

  /* alias: ka::range1d<T> in place of array<1,T> */
  template<typename T>
  struct range1d : public array<1,T> {
    range1d( T* beg, T* end ) : array<1,T>(beg, end-beg) {}
    range1d( T* beg, size_t size ) : array<1,T>(beg, size ) {}
    range1d( const array<1,T>& a ) : array<1,T>(a) {}
    range1d operator() (const rangeindex& ri) 
    { return range1d( array<1,T>::operator()(ri) ); }
    range1d operator[] (const rangeindex& ri) 
    { return range1d( array<1,T>::operator[](ri) ); }
    range1d& operator=( const array<1,T>& a )
    { 
      array<1,T>::operator=( a );
      return *this;
    }
  };

  /* alias ka::range2d<T> in place of array<2,T,S> */
  template<typename T,Storage2DClass S=RowMajor>
  struct range2d : public array<2,T,S> {
    range2d( T* beg, size_t n, size_t m, size_t lda ) : array<2,T,S>(beg, n, m, lda ) {}
    range2d( const array<2,T,S>& a ) : array<2,T,S>(a) {}
    range2d operator() (const rangeindex& ri, const rangeindex& rj) 
    { return range2d( array<2,T,S>::operator()(ri,rj) ); }
    range2d& operator=( const array<2,T,S>& a )
    { 
      array<2,T,S>::operator=( a );
      return *this;
    }
  };

  /* ------ formal parameter of type _r, _w and _rw and rpwp over array */
  template<typename T>
  class pointer_r<array<1,T> > : protected array<1,T> {
  public:
    typedef T                      value_type;
    typedef size_t                 difference_type;
    typedef pointer_r<array<1,T> > Self_t;

    pointer_r() : array<1,T>() {}
    pointer_r( const array<1,T>& a ) : array<1,T>(a) {}
    /* cstor call on closure creation */
    explicit pointer_r( array_inclosure_t<1,T,RowMajor>& a ) : array<1,T>(a) {}
    /* use in spawn effective -> in closure */
    operator array_inclosure_t<1,T,RowMajor>() const { return array_inclosure_t<1,T,RowMajor>(*this); }
    
    /* public interface */
    array<1,T>& operator*() { return *this; }
    array<1,T>* operator->() { return this; }
    const array<1,T>& operator*() const { return *this; }
    const array<1,T>* operator->() const { return this; }

    size_t size() const { return array<1,T>::size(); }
    const T* ptr() const { return array<1,T>::ptr(); }
    const T* begin() const { return array<1,T>::ptr(); }
    const T* end() const { return array<1,T>::ptr()+array<1,T>::size(); }

    const T& operator[](int i) const { return array<1,T>::operator[](i); }
    const T& operator[](unsigned i) const { return array<1,T>::operator[](i); }
    const T& operator[](long i) const { return array<1,T>::operator[](i); }
    const T& operator[](difference_type i) const { return array<1,T>::operator[](i); }

    const T& operator()(int i) const { return array<1,T>::operator[](i); }
    const T& operator()(unsigned i) const { return array<1,T>::operator[](i); }
    const T& operator()(long i) const { return array<1,T>::operator[](i); }
    const T& operator()(difference_type i) const { return array<1,T>::operator[](i); }
  };

  /* alias: ka::range1d_r<T> in place of pointer_r<array<1,T> > */
  template<typename T>
  struct range1d_r : public pointer_r<array<1,T> > {
    typedef size_t                   difference_type;
    typedef range1d_r<T>             Self_t;

    range1d_r( range1d<T>& a ) : pointer_r<array<1,T> >(a) {}
    explicit range1d_r( array<1,T>& a ) : pointer_r<array<1,T> >(a) {}
    explicit range1d_r( const array<1,T>& a ) : pointer_r<array<1,T> >(a) {}

    const T& operator[](difference_type i) const { return array<1,T>::operator[](i); }
    const T& operator()(difference_type i) const { return array<1,T>::operator[](i); }

    Self_t operator[] (const rangeindex& r) const 
    { return Self_t( array<1,T>::operator()(r) ); }
    Self_t operator() (const rangeindex& r) const 
    { return Self_t( array<1,T>::operator()(r) ); }
  };

  /* */
  template<typename T>
  class pointer_w<array<1,T> > : protected array<1,T> {
    friend class array_inclosure_t<1,T,RowMajor>;
  public:
    typedef T                        value_type;
    typedef size_t                   difference_type;
    typedef pointer_w<array<1,T> >   Self_t;

    pointer_w() : array<1,T>() {}
    pointer_w( const array_inclosure_t<1,T,RowMajor>& a ) : array<1,T>(a) {}
    /* cstor call on closure creation */
    explicit pointer_w( array<1,T>& a ) : array<1,T>(a) {}
    /* use in spawn effective -> in closure */
    operator array<1,T>() const { return *this; }
    
    /* public interface */
    array<1,T>& operator*() { return *this; }
    array<1,T>* operator->() { return this; }
    const array<1,T>& operator*() const { return *this; }
    const array<1,T>* operator->() const { return this; }

    size_t size() const { return array<1,T>::size(); }
    T* ptr() { return array<1,T>::ptr(); }
    T* begin() { return array<1,T>::ptr(); }
    T* end() { return array<1,T>::ptr()+array<1,T>::size(); }

    T& operator[](int i)  { return array<1,T>::operator[](i); }
    T& operator[](unsigned i)  { return array<1,T>::operator[](i); }
    T& operator[](long i) { return array<1,T>::operator[](i); }
    T& operator[](difference_type i) { return array<1,T>::operator[](i); }

    T& operator()(int i)  { return array<1,T>::operator[](i); }
    T& operator()(unsigned i)  { return array<1,T>::operator[](i); }
    T& operator()(long i) { return array<1,T>::operator[](i); }
    T& operator()(difference_type i) { return array<1,T>::operator[](i); }

    Self_t operator[] (const rangeindex& r) const 
    { return pointer_w( array<1,T>::operator()(r) ); }
    Self_t operator() (const rangeindex& r) const 
    { return pointer_w( array<1,T>::operator()(r) ); }
  };

  /* alias: ka::range1d_w<T> in place of pointer_w<array<1,T> > */
  template<typename T>
  struct range1d_w : public pointer_w<array<1,T> > {
    typedef range1d_w<T>             Self_t;
    typedef size_t                   difference_type;

    range1d_w( range1d<T>& a ) : pointer_w<array<1,T> >(a) {}
    explicit range1d_w( array<1,T>& a ) : pointer_w<array<1,T> >(a) {}

    T& operator[](difference_type i) { return array<1,T>::operator[](i); }
    T& operator()(difference_type i) { return array<1,T>::operator[](i); }

    Self_t operator[] (const rangeindex& r) 
    { return Self_t( array<1,T>::operator()(r) ); }
    Self_t operator() (const rangeindex& r) 
    { return Self_t( array<1,T>::operator()(r) ); }
  };

  /**/
  template<typename T>
  class pointer_rw<array<1,T> > : protected array<1,T> {
    friend class array_inclosure_t<1,T,RowMajor>;
  public:
    typedef T                        value_type;
    typedef size_t                   difference_type;
    typedef pointer_rw<array<1,T> > Self_t;

    pointer_rw() : array<1,T>() {}
    pointer_rw( const array<1,T>& a ) : array<1,T>(a) {}
    /* cstor call on closure creation */
    explicit pointer_rw( array_inclosure_t<1,T,RowMajor>& a ) : array<1,T>(a) {}
    /* use in spawn effective -> in closure */
    operator array_inclosure_t<1,T,RowMajor>() const { return array_inclosure_t<1,T,RowMajor>(*this); }
    
    /* public interface */
    array<1,T>& operator*() { return *this; }
    array<1,T>* operator->() { return this; }
    const array<1,T>& operator*() const { return *this; }
    const array<1,T>* operator->() const { return this; }

    size_t size() const { return array<1,T>::size(); }
    T* ptr() { return array<1,T>::ptr(); }
    T* begin() { return array<1,T>::ptr(); }
    T* end() { return array<1,T>::ptr()+array<1,T>::size(); }

    T& operator[](int i)  { return array<1,T>::operator[](i); }
    T& operator[](unsigned i)  { return array<1,T>::operator[](i); }
    T& operator[](long i) { return array<1,T>::operator[](i); }
    T& operator[](difference_type i) { return array<1,T>::operator[](i); }

    T& operator()(int i)  { return array<1,T>::operator[](i); }
    T& operator()(unsigned i)  { return array<1,T>::operator[](i); }
    T& operator()(long i) { return array<1,T>::operator[](i); }
    T& operator()(difference_type i) { return array<1,T>::operator[](i); }

    Self_t operator[] (const rangeindex& r) const 
    { return pointer_rw( array<1,T>::operator()(r) ); }
    Self_t operator() (const rangeindex& r) const 
    { return pointer_rw( array<1,T>::operator()(r) ); }
  };

  /* alias: ka::range1d_rw<T> in place of pointer_rw<array<1,T> > */
  template<typename T>
  struct range1d_rw : public pointer_rw<array<1,T> > {
    typedef range1d_rw<T>            Self_t;
    typedef size_t                   difference_type;

    range1d_rw( range1d<T>& a ) : pointer_rw<array<1,T> >(a) {}
    range1d_rw( const range1d<T>& a ) : pointer_rw<array<1,T> >(a) {}
    explicit range1d_rw( array<1,T>& a ) : pointer_rw<array<1,T> >(a) {}
    explicit range1d_rw( const array<1,T>& a ) : pointer_rw<array<1,T> >(a) {}

    T& operator[](difference_type i) { return array<1,T>::operator[](i); }
    T& operator()(difference_type i) { return array<1,T>::operator[](i); }

    Self_t operator[] (const rangeindex& r) 
    { return Self_t( array<1,T>::operator()(r) ); }
    Self_t operator() (const rangeindex& r) 
    { return Self_t( array<1,T>::operator()(r) ); }
  };


  template<typename T>
  class pointer_rpwp<array<1,T> > : protected array<1,T> {
    friend class array_inclosure_t<1,T,RowMajor>;
  public:
    typedef T                        value_type;
    typedef size_t                   difference_type;
    typedef pointer_rpwp<array<1,T> > Self_t;

    pointer_rpwp() : array<1,T>() {}
    pointer_rpwp( const array<1,T>& a ) : array<1,T>(a) {}
    /* cstor call on closure creation */
    explicit pointer_rpwp( array_inclosure_t<1,T,RowMajor>& a ) : array<1,T>(a) {}
    /* use in spawn effective -> in closure */
    operator array_inclosure_t<1,T,RowMajor>() const { return array_inclosure_t<1,T,RowMajor>(*this); }
    
    /* public interface */
    array<1,T>& operator*() { return *this; }
    array<1,T>* operator->() { return this; }
    const array<1,T>& operator*() const { return *this; }
    const array<1,T>* operator->() const { return this; }

    size_t size() const { return array<1,T>::size(); }
    const T* ptr() const { return array<1,T>::ptr(); }
    const T* begin() const { return array<1,T>::ptr(); }
    const T* end() const { return array<1,T>::ptr()+array<1,T>::size(); }

    Self_t operator[] (const rangeindex& r) const 
    { return pointer_rpwp( array<1,T>::operator()(r) ); }
    Self_t operator() (const rangeindex& r) const 
    { return pointer_rpwp( array<1,T>::operator()(r) ); }
  };

  /* alias: ka::range1d_rpwp<T> in place of pointer_rw<array<1,T> > */
  template<typename T>
  struct range1d_rpwp : public pointer_rpwp<array<1,T> > {
    typedef range1d_rpwp<T>             Self_t;

    range1d_rpwp( range1d<T>& a ) : pointer_rpwp<array<1,T> >(a) {}
    explicit range1d_rpwp( array<1,T>& a ) : pointer_rpwp<array<1,T> >(a) {}
    explicit range1d_rpwp( const array<1,T>& a ) : pointer_rpwp<array<1,T> >(a) {}

    Self_t operator[] (const rangeindex& r) const 
    { return Self_t( array<1,T>::operator()(r) ); }
    Self_t operator() (const rangeindex& r) const 
    { return Self_t( array<1,T>::operator()(r) ); }
  };


  template<typename T>
  class pointer_cw<array<1,T> > : protected array<1,T> {
    friend class array_inclosure_t<1,T,RowMajor>;
  public:
    typedef T                        value_type;
    typedef size_t                   difference_type;
    typedef pointer_cw<array<1,T> > Self_t;

    pointer_cw() : array<1,T>() {}
    pointer_cw( const array<1,T>& a ) : array<1,T>(a) {}
    /* cstor call on closure creation */
    explicit pointer_cw( array_inclosure_t<1,T,RowMajor>& a ) : array<1,T>(a) {}
    /* use in spawn effective -> in closure */
    operator array_inclosure_t<1,T,RowMajor>() const { return array_inclosure_t<1,T,RowMajor>(*this); }
    
    /* public interface */
    array<1,T>& operator*() { return *this; }
    array<1,T>* operator->() { return this; }
    const array<1,T>& operator*() const { return *this; }
    const array<1,T>* operator->() const { return this; }

    size_t size() const { return array<1,T>::size(); }
    T* ptr() { return array<1,T>::ptr(); }
    T* begin() { return array<1,T>::ptr(); }
    T* end() { return array<1,T>::ptr()+array<1,T>::size(); }

    T& operator[](int i)  { return array<1,T>::operator[](i); }
    T& operator[](unsigned i)  { return array<1,T>::operator[](i); }
    T& operator[](long i) { return array<1,T>::operator[](i); }
    T& operator[](difference_type i) { return array<1,T>::operator[](i); }

    T& operator()(int i)  { return array<1,T>::operator[](i); }
    T& operator()(unsigned i)  { return array<1,T>::operator[](i); }
    T& operator()(long i) { return array<1,T>::operator[](i); }
    T& operator()(difference_type i) { return array<1,T>::operator[](i); }

    Self_t operator[] (const rangeindex& r) const 
    { return pointer_cw( array<1,T>::operator()(r) ); }
    Self_t operator() (const rangeindex& r) const 
    { return pointer_cw( array<1,T>::operator()(r) ); }
  };

  /* alias: ka::range1d_cw<T> in place of pointer_cw<array<1,T> > */
  template<typename T>
  struct range1d_cw : public pointer_cw<array<1,T> > {
    typedef range1d_cw<T>             Self_t;

    range1d_cw( range1d<T>& a ) : pointer_cw<array<1,T> >(a) {}
    explicit range1d_cw( array<1,T>& a ) : pointer_cw<array<1,T> >(a) {}

    Self_t operator[] (const rangeindex& r) const 
    { return Self_t( array<1,T>::operator()(r) ); }
    Self_t operator() (const rangeindex& r) const 
    { return Self_t( array<1,T>::operator()(r) ); }
  };



  /* same for 2d range */
  /* ------ formal parameter of type _r, _w and _rw and rpwp over array */
  template<typename T, Storage2DClass S>
  class pointer_r<array<2,T,S> > : protected array<2,T,S> {
    friend class array_inclosure_t<2,T,S>;
  public:
    typedef T                              value_type;
    typedef size_t                         difference_type;
    typedef typename array<2,T,S>::index_t index_t;
    typedef pointer_r<array<2,T,S> >       Self_t;

    pointer_r() : array<2,T,S>() {}
    pointer_r( const array<2,T,S>& a ) : array<2,T,S>(a) {}
    /* cstor call on closure creation */
    explicit pointer_r( array_inclosure_t<2,T,S>& a ) : array<2,T,S>(a) {}

    /* use in spawn effective -> in closure */
    operator array_inclosure_t<2,T,S>() const { return array_inclosure_t<2,T,S>(*this); }
    
    /* public interface */
    array<2,T,S>& operator*() { return *this; }
    array<2,T,S>* operator->() { return this; }
    const array<2,T,S>& operator*() const { return *this; }
    const array<2,T,S>* operator->() const { return this; }
    Self_t operator() (const rangeindex& ri, const rangeindex& rj) const
    { return Self_t( array<2,T,S>::operator()(ri,rj) ); }
  };

  /* alias: ka::range2d_r<T,S> in place of pointer_r<array<2,T,S> > */
  template<typename T, Storage2DClass S = RowMajor>
  class range2d_r : public pointer_r<array<2,T,S> > {
  public:
    typedef pointer_r<array<2,T,S> >           Self_t;
    typedef typename Self_t::value_type      value_type;
    typedef typename Self_t::difference_type difference_type;
    typedef typename Self_t::index_t         index_t;

    range2d_r( range2d<T>& a ) : pointer_r<array<2,T,S> >(a) {}
    explicit range2d_r(  const array<2,T,S>& a ) : pointer_r<array<2,T,S> >(a) {}

    const T& operator()(int i, int j) const { return array_rep<2,T,S>::operator()(i,j); }
    const T& operator()(unsigned int i, unsigned int j) const { return array_rep<2,T,S>::operator()(i,j); }
    const T& operator()(long i, long j) const { return array_rep<2,T,S>::operator()(i,j); }
    const T& operator()(unsigned long i, unsigned long j) const { return array_rep<2,T,S>::operator()(i,j); }

    const T* ptr() { return array_rep<2,T,S>::ptr(); }
    const T* ptr() const { return array_rep<2,T,S>::ptr(); }
    size_t dim(int i) const { return array_rep<2,T,S>::dim(i); }
    size_t lda() const { return array_rep<2,T,S>::lda(); }

    range2d_r<T,S> operator()(const rangeindex& ri, const rangeindex& rj)  const 
    { return range2d_r<T,S>( range2d_r<T,S>(array<2,T,S>::operator()(ri,rj) ) ); }
  };


  template<typename T, Storage2DClass S>
  class pointer_w<array<2,T,S> > : protected array<2,T,S> {
    friend class array_inclosure_t<2,T,S>;
  public:
    typedef T                            value_type;
    typedef size_t                       difference_type;
    typedef typename array<2,T,S>::index_t index_t;
    typedef pointer_w<array<2,T,S> >       Self_t;

    pointer_w() : array<2,T,S>() {}
    pointer_w( const array_inclosure_t<2,T,S>& a ) : array<2,T,S>(a) {}
    /* cstor call on closure creation */
    explicit pointer_w( array<2,T,S>& a ) : array<2,T,S>(a) {}
    /* use in spawn effective -> in closure */
    operator array<2,T,S>() const { return *this; }
    
    /* public interface */
    array<2,T,S>& operator*() { return *this; }
    array<2,T,S>* operator->() { return this; }
    const array<2,T,S>& operator*() const { return *this; }
    const array<2,T,S>* operator->() const { return this; }
    Self_t operator() (const rangeindex& ri, const rangeindex& rj) 
    { return Self_t( array_rep<2,T,S>::operator()(ri,rj) ); }
  };


  /* alias: ka::range2d_w<T,S> in place of pointer_w<array<2,T,S> > */
  template<typename T, Storage2DClass S = RowMajor>
  struct range2d_w : public pointer_w<array<2,T,S> > 
  {
    typedef pointer_w<array<2,T,S> >           Self_t;
    typedef typename Self_t::value_type      value_type;
    typedef typename Self_t::difference_type difference_type;
    typedef typename Self_t::index_t         index_t;

    range2d_w( range2d<T>& a ) : pointer_w<array<2,T,S> >(a) {}
    explicit range2d_w( const array<2,T,S>& a ) : pointer_w<array<2,T,S> >(a) {}

    T& operator()(int i, int j)  { return array_rep<2,T,S>::operator()(i,j); }
    T& operator()(unsigned int i, unsigned int j)  { return array_rep<2,T,S>::operator()(i,j); }
    T& operator()(long i, long j) { return array_rep<2,T,S>::operator()(i,j); }
    T& operator()(unsigned long i, unsigned long j) { return array_rep<2,T,S>::operator()(i,j); }

    T* ptr() { return array_rep<2,T,S>::ptr(); }
    const T* ptr() const { return array_rep<2,T,S>::ptr(); }
    size_t dim(int i) const { return array_rep<2,T,S>::dim(i); }
    size_t lda() const { return array_rep<2,T,S>::lda(); }

    range2d_w<T,S> operator() (const rangeindex& ri, const rangeindex& rj) 
    { return range2d_w<T,S>( array<2,T,S>::operator()(ri,rj) ); }

    void operator=(const T& value)  { array_rep<2,T,S>::operator=(value); }
  };


  template<typename T, Storage2DClass S>
  class pointer_rw<array<2,T,S> > : protected array<2,T,S> {
    friend class array_inclosure_t<2,T,S>;
  public:
    typedef T                            value_type;
    typedef size_t                       difference_type;
    typedef typename array<2,T,S>::index_t index_t;
    typedef pointer_rw<array<2,T,S> >      Self_t;

    pointer_rw() : array<2,T,S>() {}
    pointer_rw( const array<2,T,S>& a ) : array<2,T,S>(a) {}
    /* cstor call on closure creation */
    explicit pointer_rw( array_inclosure_t<2,T,S>& a ) : array<2,T,S>(a) {}
    /* use in spawn effective -> in closure */
    operator array_inclosure_t<2,T,S>() const { return array_inclosure_t<2,T,S>(*this); }
    
    /* public interface */
    array<2,T,S>& operator*() { return *this; }
    array<2,T,S>* operator->() { return this; }
    const array<2,T,S>& operator*() const { return *this; }
    const array<2,T,S>* operator->() const { return this; }

    Self_t operator() (const rangeindex& ri, const rangeindex& rj) const 
    { return Self_t( array_rep<2,T,S>::operator()(ri,rj) ); }
  };

  /* alias: ka::range2d_rw<T,S> in place of pointer_rw<array<2,T,S> > */
  template<typename T, Storage2DClass S = RowMajor>
  struct range2d_rw : public pointer_rw<array<2,T,S> > 
  {
    typedef pointer_rw<array<2,T,S> >          Self_t;
    typedef typename Self_t::value_type      value_type;
    typedef typename Self_t::difference_type difference_type;
    typedef typename Self_t::index_t         index_t;

    range2d_rw( range2d<T>& a ) : pointer_rw<array<2,T,S> >(a) {}
    explicit range2d_rw( const array<2,T,S>& a ) : pointer_rw<array<2,T,S> >(a) {}

    T& operator()(int i, int j)  { return array_rep<2,T,S>::operator()(i,j); }
    T& operator()(unsigned int i, unsigned int j)  { return array_rep<2,T,S>::operator()(i,j); }
    T& operator()(long i, long j) { return array_rep<2,T,S>::operator()(i,j); }
    T& operator()(unsigned long i, unsigned long j) { return array_rep<2,T,S>::operator()(i,j); }

    T* ptr() { return array_rep<2,T,S>::ptr(); }
    const T* ptr() const { return array_rep<2,T,S>::ptr(); }
    size_t dim(int i) const { return array_rep<2,T,S>::dim(i); }
    size_t lda() const { return array_rep<2,T,S>::lda(); }

    range2d_rw<T,S> operator() (const rangeindex& ri, const rangeindex& rj) const 
    { return range2d_rw<T,S>( array<2,T,S>::operator()(ri,rj) ); }

    void operator=(const T& value) { array_rep<2,T,S>::operator=(value); }
  };


  template<typename T, Storage2DClass S>
  class pointer_rpwp<array<2,T,S> > : protected array<2,T,S> {
    friend class array_inclosure_t<2,T,S>;
  public:
    typedef T                         value_type;
    typedef size_t                    difference_type;
    typedef pointer_rpwp<array<2,T,S> > Self_t;

    pointer_rpwp() : array<2,T,S>() {}
    pointer_rpwp( const array<2,T,S>& a ) : array<2,T,S>(a) {}
    /* cstor call on closure creation */
    explicit pointer_rpwp( array_inclosure_t<2,T,S>& a ) : array<2,T,S>(a) {}
    /* use in spawn effective -> in closure */
    operator array_inclosure_t<2,T,S>() const { return array_inclosure_t<2,T,S>(*this); }
    
    /* public interface */
    array<2,T,S>& operator*() { return *this; }
    array<2,T,S>* operator->() { return this; }
    const array<2,T,S>& operator*() const { return *this; }
    const array<2,T,S>* operator->() const { return this; }

    Self_t operator() (const rangeindex& ri, const rangeindex& rj) const 
    { return Self_t( array_rep<2,T,S>::operator()(ri,rj) ); }
  };

  /* alias: ka::range2d_rpwp<T> in place of pointer_rw<array<2,T,S> > */
  template<typename T, Storage2DClass S = RowMajor>
  struct range2d_rpwp : public pointer_rpwp<array<2,T,S> > 
  {
    typedef pointer_rpwp<array<2,T,S> >        Self_t;
    typedef typename Self_t::value_type      value_type;
    typedef typename Self_t::difference_type difference_type;
    typedef typename Self_t::index_t         index_t;

    range2d_rpwp( range2d<T>& a ) : pointer_rpwp<array<2,T,S> >(a) {}
    explicit range2d_rpwp( const array<2,T,S>& a ) : pointer_rpwp<array<2,T,S> >(a) {}

    T* ptr() { return array_rep<2,T,S>::ptr(); }
    const T* ptr() const { return array_rep<2,T,S>::ptr(); }
    size_t dim(int i) const { return array_rep<2,T,S>::dim(i); }
    size_t lda() const { return array_rep<2,T,S>::lda(); }

    range2d_rpwp<T> operator() (const rangeindex& ri, const rangeindex& rj) const
    { return range2d_rpwp<T>( array<2,T,S>::operator()(ri,rj) ); }
  };


  template<typename T, Storage2DClass S>
  class pointer_cw<array<2,T,S> > : protected array<2,T,S> {
    friend class array_inclosure_t<2,T,S>;
  public:
    typedef T                            value_type;
    typedef size_t                       difference_type;
    typedef typename array<2,T,S>::index_t index_t;
    typedef pointer_cw<array<2,T,S> >      Self_t;

    pointer_cw() : array<2,T,S>() {}
    pointer_cw( const array<2,T,S>& a ) : array<2,T,S>(a) {}
    /* cstor call on closure creation */
    explicit pointer_cw( array_inclosure_t<2,T,S>& a ) : array<2,T,S>(a) {}
    /* use in spawn effective -> in closure */
    operator array_inclosure_t<2,T,S>() const { return array_inclosure_t<2,T,S>(*this); }
    
    /* public interface */
    array<2,T,S>& operator*() { return *this; }
    array<2,T,S>* operator->() { return this; }
    const array<2,T,S>& operator*() const { return *this; }
    const array<2,T,S>* operator->() const { return this; }

    Self_t operator() (const rangeindex& ri, const rangeindex& rj) const 
    { return Self_t( array_rep<2,T,S>::operator()(ri,rj) ); }
  };

  /* alias: ka::range2d_cw<T,S> in place of pointer_cw<array<2,T,S> > */
  template<typename T, Storage2DClass S = RowMajor>
  struct range2d_cw : public pointer_cw<array<2,T,S> > 
  {
    typedef pointer_cw<array<2,T,S> >          Self_t;
    typedef typename Self_t::value_type      value_type;
    typedef typename Self_t::difference_type difference_type;
    typedef typename Self_t::index_t         index_t;

    range2d_cw( range2d<T>& a ) : pointer_cw<array<2,T,S> >(a) {}
    explicit range2d_cw( const array<2,T,S>& a ) : pointer_cw<array<2,T,S> >(a) {}

    T& operator()(int i, int j)  { return array_rep<2,T,S>::operator()(i,j); }
    T& operator()(unsigned int i, unsigned int j)  { return array_rep<2,T,S>::operator()(i,j); }
    T& operator()(long i, long j) { return array_rep<2,T,S>::operator()(i,j); }
    T& operator()(unsigned long i, unsigned long j) { return array_rep<2,T,S>::operator()(i,j); }

    T* ptr() { return array_rep<2,T,S>::ptr(); }
    const T* ptr() const { return array_rep<2,T,S>::ptr(); }
    size_t dim(int i) const { return array_rep<2,T,S>::dim(i); }
    size_t lda() const { return array_rep<2,T,S>::lda(); }

    range2d_cw<T,S> operator() (const rangeindex& ri, const rangeindex& rj) const 
    { return range2d_cw<T,S>( array<2,T,S>::operator()(ri,rj) ); }

    void operator=(const T& value) { array_rep<2,T,S>::operator=(value); }
  };


  /* trait */    
  template<int dim, typename T, Storage2DClass S>
  struct TraitFormalParam<pointer_r<array<dim,T,S> > > { 
    typedef array_inclosure_t<dim,T,S>  type_inclosure_t;
    typedef R<array<dim,T,S> >          signature_t; 
    typedef pointer_r<array<dim,T,S> >  formal_t; 
    typedef ACCESS_MODE_R             mode_t; 
    typedef T                         type_t;
    static const bool                 is_static = false;
    static const void*                ptr( const pointer_r<array<dim,T,S> >* a ) { return a->ptr(); }
    static formal_t                   handle2data( type_inclosure_t* a) 
    { array<dim,T,S> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
    static const void*                get_data   ( const type_inclosure_t* a, unsigned int i ) { return 0; }
    static void                       get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) 
    { r->data = a->ptr(); r->version = a->version; }
    static void                       set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) 
    { a->setptr( (T*)r->data ); a->version = r->version; }
    static size_t                     get_nparam( const type_inclosure_t* a ) 
    { return 1; }
    static kaapi_memory_view_t        get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return a->get_view(); }
    static void                       set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { a->set_view(view); }
    static void                       reducor_fnc(void*, const void*) {}
    static void                       redinit_fnc(void*) {}
   };

  template<int dim, typename T, Storage2DClass S>
  struct TraitFormalParam<pointer_w<array<dim,T,S> > > { 
    typedef array_inclosure_t<dim,T,S>  type_inclosure_t;
    typedef W<array<dim,T,S> >          signature_t; 
    typedef pointer_w<array<dim,T,S> >  formal_t; 
    typedef ACCESS_MODE_W             mode_t; 
    typedef T                         type_t;
    static const bool                 is_static = false;
    static const void*                ptr( const pointer_w<array<dim,T,S> >* a ) { return a->ptr(); }
    static formal_t                   handle2data( type_inclosure_t* a) 
    { array<dim,T,S> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
//    static formal_t                   handle2data( type_inclosure_t* a) 
//    { array<dim,T,S> retval(*a); retval.setptr( *(T**)a->ptr() ); return (formal_t)retval; }
    static const void*                get_data   ( const type_inclosure_t* a, unsigned int i ) { return 0; }
    static void                       get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) 
    { r->data = a->ptr(); r->version = a->version; }
    static void                       set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) 
    { a->setptr( (T*)r->data ); a->version = r->version; }
    static size_t                     get_nparam( const type_inclosure_t* a ) 
    { return 1; }
    static kaapi_memory_view_t        get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return a->get_view(); }
    static void                       set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { a->set_view(view); }
    static void                       reducor_fnc(void*, const void*) {}
    static void                       redinit_fnc(void*) {}
  };

  template<int dim, typename T, Storage2DClass S>
  struct TraitFormalParam<pointer_rw<array<dim,T,S> > > { 
    typedef array_inclosure_t<dim,T,S>  type_inclosure_t;
    typedef RW<array<dim,T,S> >         signature_t; 
    typedef pointer_rw<array<dim,T,S> > formal_t; 
    typedef ACCESS_MODE_RW            mode_t; 
    typedef T                         type_t;
    static const bool                 is_static = false;
    static const void*                ptr( const pointer_rw<array<dim,T,S> >* a ) { return a->ptr(); }
    static formal_t                   handle2data( type_inclosure_t* a) 
    { array<dim,T,S> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
//    static formal_t                   handle2data( type_inclosure_t* a) 
//    { array<dim,T,S> retval(*a); retval.setptr( *(T**)a->ptr() ); return (formal_t)retval; }
    static const void*                get_data   ( const type_inclosure_t* a, unsigned int i ) { return 0; }
    static void                       get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) 
    { r->data = a->ptr(); r->version = a->version; }
    static void                       set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) 
    { a->setptr( (T*)r->data ); a->version = r->version; }
    static size_t                     get_nparam( const type_inclosure_t* a ) 
    { return 1; }
    static kaapi_memory_view_t        get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return a->get_view(); }
    static void                       set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { a->set_view(view); }
    static void                       reducor_fnc(void*, const void*) {}
    static void                       redinit_fnc(void*) {}
  };   

  template<int dim, typename T, Storage2DClass S>
  struct TraitFormalParam<pointer_rpwp<array<dim,T,S> > > { 
    typedef array_inclosure_t<dim,T,S>  type_inclosure_t;
    typedef RPWP<array<dim,T,S> >       signature_t; 
    typedef pointer_rpwp<array<dim,T,S> > formal_t; 
    typedef ACCESS_MODE_RPWP          mode_t; 
    typedef T                         type_t;
    static const bool                 is_static = false;
    static const void*                ptr( const pointer_rpwp<array<dim,T,S> >* a ) { return a->ptr(); }
    static formal_t                   handle2data( type_inclosure_t* a) 
    { array<dim,T,S> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
//    static formal_t                   handle2data( type_inclosure_t* a) 
//    { array<dim,T,S> retval(*a); retval.setptr( *(T**)a->ptr() ); return (formal_t)retval; }
    static const void*                get_data   ( const type_inclosure_t* a, unsigned int i ) { return 0; }
    static void                       get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) 
    { r->data = a->ptr(); r->version = a->version; }
    static void                       set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) 
    { a->setptr( (T*)r->data ); a->version = r->version; }
    static size_t                     get_nparam( const type_inclosure_t* a ) 
    { return 1; }
    static kaapi_memory_view_t        get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return a->get_view(); }
    static void                       set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { a->set_view(view); }
    static void                       reducor_fnc(void*, const void*) {}
    static void                       redinit_fnc(void*) {}
  };   

  template<int dim, typename T, Storage2DClass S>
  struct TraitFormalParam<pointer_cw<array<dim,T,S> > > { 
    typedef array_inclosure_t<dim,T,S>  type_inclosure_t;
    typedef CW<array<dim,T,S> >         signature_t; 
    typedef pointer_cw<array<dim,T,S> > formal_t; 
    typedef ACCESS_MODE_CW            mode_t; 
    typedef T                         type_t;
    static const bool                 is_static = false;
    static const void*                ptr( const pointer_cw<array<dim,T,S> >* a ) { return a->ptr(); }
    static formal_t                   handle2data( type_inclosure_t* a) 
    { array<dim,T,S> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
//    static formal_t                   handle2data( type_inclosure_t* a) 
//    { array<dim,T,S> retval(*a); retval.setptr( *(T**)a->ptr() ); return (formal_t)retval; }
    static const void*                get_data   ( const type_inclosure_t* a, unsigned int i ) { return 0; }
    static void                       get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) 
    { r->data = a->ptr(); r->version = a->version; }
    static void                       set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) 
    { a->setptr( (T*)r->data ); a->version = r->version; }
    static size_t                     get_nparam( const type_inclosure_t* a ) 
    { return 1; }
    static kaapi_memory_view_t        get_view_param( const type_inclosure_t* a, unsigned int i ) 
    { return a->get_view(); }
    static void                       set_view_param( type_inclosure_t* a, unsigned int i, const kaapi_memory_view_t*  view ) 
    { a->set_view(view); }
    static void                       reducor_fnc(void*, const void*) {}
    static void                       redinit_fnc(void*) {}
  };   


  template<int dim, typename T, Storage2DClass S>
  struct TraitFormalParam< R<array<dim,T,S> > > : public TraitFormalParam<pointer_r<array<dim,T,S> > > {};
  template<int dim, typename T, Storage2DClass S>
  struct TraitFormalParam< W<array<dim,T,S> > > : public TraitFormalParam<pointer_w<array<dim,T,S> > > {};
  template<int dim, typename T, Storage2DClass S>
  struct TraitFormalParam< RW<array<dim,T,S> > > : public TraitFormalParam<pointer_rw<array<dim,T,S> > > {};
  template<int dim, typename T, Storage2DClass S>
  struct TraitFormalParam< RPWP<array<dim,T,S> > > : public TraitFormalParam<pointer_rpwp<array<dim,T,S> > > {};
  template<int dim, typename T, Storage2DClass S>
  struct TraitFormalParam< CW<array<dim,T,S> > > : public TraitFormalParam<pointer_cw<array<dim,T,S> > > {};
  
  template<typename T>
  struct TraitFormalParam<range1d<T> > : public TraitFormalParam<array<1,T> > { };
  template<typename T>
  struct TraitFormalParam<range1d_r<T> > : public TraitFormalParam<pointer_r<array<1,T> > > { 
    typedef TraitFormalParam<pointer_r<array<1,T> > > inherited_t;
    typedef typename inherited_t::type_inclosure_t type_inclosure_t;
    typedef range1d_r<T>    formal_t; 
    typedef R<range1d <T> > signature_t; 
    static formal_t                   handle2data( type_inclosure_t* a) 
    { array<1,T> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
//    static formal_t         handle2data( type_inclosure_t* a) 
//    { array<1,T> retval(*a); retval.setptr( *(T**)a->ptr() ); return (formal_t)retval; }
  };
  template<typename T>
  struct TraitFormalParam<range1d_w<T> > : public TraitFormalParam<pointer_w<array<1,T> > > {
    typedef TraitFormalParam<pointer_w<array<1,T> > > inherited_t;
    typedef typename inherited_t::type_inclosure_t type_inclosure_t;
    typedef range1d_w<T>    formal_t; 
    typedef W<range1d<T> >  signature_t; 
    static formal_t         handle2data( type_inclosure_t* a) 
    { array<1,T> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
//    { array<1,T> retval(*a); retval.setptr( *(T**)a->ptr() ); return (formal_t)retval; }
  };
  template<typename T>
  struct TraitFormalParam<range1d_rw<T> > : public TraitFormalParam<pointer_rw<array<1,T> > > {
    typedef TraitFormalParam<pointer_rw<array<1,T> > > inherited_t;
    typedef typename inherited_t::type_inclosure_t type_inclosure_t;
    typedef range1d_rw<T>   formal_t; 
    typedef RW<range1d<T> > signature_t; 
    static formal_t           handle2data( type_inclosure_t* a) 
    { array<1,T> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
//    { array<1,T> retval(*a); retval.setptr( *(T**)a->ptr() ); return (formal_t)retval; }
  };
  template<typename T>
  struct TraitFormalParam<range1d_rpwp<T> > : public TraitFormalParam<pointer_rpwp<array<1,T> > > {
    typedef TraitFormalParam<pointer_rpwp<array<1,T> > > inherited_t;
    typedef typename inherited_t::type_inclosure_t type_inclosure_t;
    typedef range1d_rpwp<T>   formal_t; 
    typedef RPWP<range1d<T> > signature_t; 
    static formal_t           handle2data( type_inclosure_t* a) 
    { array<1,T> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
//    { array<1,T> retval(*a); retval.setptr( *(T**)a->ptr() ); return (formal_t)retval; }
  };
  template<typename T>
  struct TraitFormalParam<range1d_cw<T> > : public TraitFormalParam<pointer_cw<array<1,T> > > {
    typedef TraitFormalParam<pointer_cw<array<1,T> > > inherited_t;
    typedef typename inherited_t::type_inclosure_t type_inclosure_t;
    typedef range1d_cw<T>     formal_t; 
    typedef RPWP<range1d<T> > signature_t; 
    static formal_t           handle2data( type_inclosure_t* a) 
    { array<1,T> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
//    { array<1,T> retval(*a); retval.setptr( *(T**)a->ptr() ); return (formal_t)retval; }
  };
  template<typename T>
  struct TraitFormalParam< R<range1d<T> > > : public TraitFormalParam<range1d_r<T> > {};
  template<typename T>
  struct TraitFormalParam< W<range1d<T> > > : public TraitFormalParam<range1d_w<T> > {};
  template<typename T>
  struct TraitFormalParam< RW<range1d<T> > > : public TraitFormalParam<range1d_rw<T> > {};
  template<typename T>
  struct TraitFormalParam< RPWP<range1d<T> > > : public TraitFormalParam<range1d_rpwp<T> > {};
  template<typename T>
  struct TraitFormalParam< CW<range1d<T> > > : public TraitFormalParam<range1d_cw<T> > {};


  template<typename T, Storage2DClass S>
  struct TraitFormalParam<range2d<T,S> > : public TraitFormalParam<array<2,T,S> > { };

  template<typename T, Storage2DClass S>
  struct TraitFormalParam<range2d_r<T,S> > : public TraitFormalParam<pointer_r<array<2,T,S> > > { 
    typedef TraitFormalParam<pointer_r<array<2,T,S> > > inherited_t;
    typedef typename inherited_t::type_inclosure_t type_inclosure_t;
    typedef range2d_r<T,S>   formal_t; 
    typedef R<range2d<T,S> > signature_t; 
    static formal_t          handle2data( type_inclosure_t* a) 
    { array<2,T,S> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
  };

  template<typename T, Storage2DClass S>
  struct TraitFormalParam<range2d_w<T,S> > : public TraitFormalParam<pointer_w<array<2,T,S> > > {
    typedef TraitFormalParam<pointer_w<array<2,T,S> > > inherited_t;
    typedef typename inherited_t::type_inclosure_t type_inclosure_t;
    typedef range2d_w<T,S>    formal_t; 
    typedef W<range2d<T,S> >  signature_t; 
    static formal_t           handle2data( type_inclosure_t* a) 
    { array<2,T,S> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
  };

  template<typename T, Storage2DClass S>
  struct TraitFormalParam<range2d_rw<T,S> > : public TraitFormalParam<pointer_rw<array<2,T,S> > > {
    typedef TraitFormalParam<pointer_rw<array<2,T,S> > > inherited_t;
    typedef typename inherited_t::type_inclosure_t type_inclosure_t;
    typedef range2d_rw<T,S>   formal_t; 
    typedef RW<range2d<T,S> > signature_t; 
    static formal_t           handle2data( type_inclosure_t* a) 
    { array<2,T,S> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
  };

  template<typename T, Storage2DClass S>
  struct TraitFormalParam<range2d_rpwp<T,S> > : public TraitFormalParam<pointer_rpwp<array<2,T,S> > > {
    typedef TraitFormalParam<pointer_rpwp<array<2,T,S> > > inherited_t;
    typedef typename inherited_t::type_inclosure_t type_inclosure_t;
    typedef range2d_rpwp<T,S>   formal_t; 
    typedef RPWP<range2d<T,S> > signature_t; 
    static formal_t             handle2data( type_inclosure_t* a) 
    { array<2,T,S> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
  };

  template<typename T, Storage2DClass S>
  struct TraitFormalParam<range2d_cw<T,S> > : public TraitFormalParam<pointer_cw<array<2,T,S> > > {
    typedef TraitFormalParam<pointer_cw<array<2,T,S> > > inherited_t;
    typedef typename inherited_t::type_inclosure_t type_inclosure_t;
    typedef range2d_cw<T,S>     formal_t; 
    typedef CW<range2d<T> >   signature_t; 
    static formal_t           handle2data( type_inclosure_t* a) 
    { array<2,T,S> retval((T*)__kaapi_pointer2void(kaapi_handle_t(a->ptr())->ptr), 
                          &(kaapi_handle_t(a->ptr()))->view);
      return (formal_t)retval; 
    }
  };

  template<typename T, Storage2DClass S>
  struct TraitFormalParam< R<range2d<T,S> > > : public TraitFormalParam<range2d_r<T,S> > {};
  template<typename T, Storage2DClass S>
  struct TraitFormalParam< W<range2d<T,S> > > : public TraitFormalParam<range2d_w<T,S> > {};
  template<typename T, Storage2DClass S>
  struct TraitFormalParam< RW<range2d<T,S> > > : public TraitFormalParam<range2d_rw<T,S> > {};
  template<typename T, Storage2DClass S>
  struct TraitFormalParam< RPWP<range2d<T,S> > > : public TraitFormalParam<range2d_rpwp<T,S> > {};
  template<typename T, Storage2DClass S>
  struct TraitFormalParam< CW<range2d<T,S> > > : public TraitFormalParam<range2d_cw<T,S> > {};


  /* ------ rep of array into a closure      
  */
  template<int dim, typename T, Storage2DClass S=RowMajor>
  struct arraytype_inclosure_t;

  template<typename T, Storage2DClass S>
  struct arraytype_inclosure_t<1,T,S> {
    arraytype_inclosure_t<1,T,S>( const array<1,T,S>& a ) : _data(a), _version() {}

    int size() const { return _data.size(); }

    array_rep<1,T,S>              _data;
    array_rep<1,T*,S>             _version; /* only used if the task is stolen */
  private:
    arraytype_inclosure_t<1,T,S>() {}
  };

  template<typename T,Storage2DClass S>
  struct arraytype_inclosure_t<2,T,S> {
    arraytype_inclosure_t<2,T,S>( const array<2,T,S>& a ) : _data(a), _version() { }
    size_t size() const { return _data.dim(0)*_data.dim(1); }

    array_rep<2,T,S>              _data;
    array_rep<2,T*,S>             _version; /* only used if the task is stolen */
  private:
    arraytype_inclosure_t<2,T,S>() {}
  };

  /* ------ specialisation of representation of array */
  template<typename T>
  class array1d_rep_with_write {
  public:
    typedef base_array::index_t index_t;
    typedef T*                  pointer_t;

    class reference_t {
    public:
      reference_t( T& v ) : ref(v) {}
      reference_t operator=( const T& v )
      { ref = v; return *this; }
    private:
      T& ref;
    };

    class const_reference_t {
    public:
      const_reference_t( T& v ) : ref(v) {}
    private:
      T& ref;
    };

    /** */
    array1d_rep_with_write() : _data(0) {}
    /** */
    array1d_rep_with_write(T* ptr, size_t sz) : _data(ptr), _size(sz) {}

    /** */
    int size() const
    { return _size; }

    /** */
    reference_t operator[](index_t i)
    { return reference_t(_data[i]); }
    /** */
    pointer_t operator+(index_t i) const
    { return _data+ i; }
    /** */
    void set (index_t i, const T& value) const 
    { _data[i] = value; }
    /** */
    pointer_t shift_base(index_t shift) 
    { return _data+shift; }
  protected:
    T*  _data;
    int _size;
  };

  template<typename T>
  class array1d_rep_with_read {
  public:
    typedef base_array::index_t index_t;
    typedef T*                  pointer_t;
    typedef const T&            const_reference_t;
    typedef const T&            reference_t;

    /** */
    array1d_rep_with_read() : _data(0) {}
    /** */
    array1d_rep_with_read(T* ptr, size_t sz) : _data(ptr), _size(sz) {}

    /** */
    int size() const
    { return _size; }

    /** */
    const_reference_t operator[](index_t i)
    { return _data[i]; }
    /** */
    const_reference_t operator[](index_t i) const
    { return _data[i]; }
    /** */
    pointer_t operator+(index_t i) const
    { return _data+ i; }
    /** */
    const_reference_t get (index_t i) const 
    { return _data[i]; }
    /** */
    pointer_t shift_base(index_t shift) 
    { return _data+shift; }
  protected:
    T*  _data;
    int _size;
  };

  /* ------ specialisation of representation of array */
  template<typename T>
  class array1d_rep_with_readwrite {
  public:
    typedef base_array::index_t index_t;
    typedef T*                  pointer_t;

    class reference_t {
    public:
      reference_t( T& v ) : ref(v) {}
      reference_t operator=( const T& v )
      { ref = v; return *this; }
      operator T&() { return ref; }
      T* operator ->() { return &ref; }
    private:
      T& ref;
    };

    class const_reference_t {
    public:
      const_reference_t( T& v ) : ref(v) {}
      operator const T&() const { return ref; }
    private:
      T& ref;
    };

    /** */
    int size() const
    { return _size; }

    /** */
    array1d_rep_with_readwrite() : _data(0), _size(0) {}
    /** */
    array1d_rep_with_readwrite(T* ptr, size_t sz) : _data(ptr), _size(sz) {}
    /** */
    reference_t operator[](index_t i)
    { return reference_t(_data[i]); }
    /** */
    const_reference_t operator[](index_t i) const
    { return const_reference_t(_data[i]); }
    /** */
    pointer_t operator+(index_t i) const
    { return _data+ i; }
    /** */
    void set (index_t i, const T& value) const 
    { _data[i] = value; }
    /** */
    pointer_t shift_base(index_t shift) 
    { return _data+shift; }
  protected:
    T*  _data;
    int _size;
  };
  
  /* specialisation of array representation */
  template<typename T>
  class array_rep<1,pointer_r<T>,RowMajor> : public array1d_rep_with_read<T> {
  public:
    array_rep<1,pointer_r<T>,RowMajor>( arraytype_inclosure_t<1,T,RowMajor>& arg_clo )
     : array1d_rep_with_read<T>( arg_clo._data.ptr(), arg_clo._data.size() ) 
    {
    }
  };

  /* specialisation of array representation */
  template<typename T>
  class array_rep<1,pointer_w<T>,RowMajor > : public array1d_rep_with_write<T> {
  public:    
    array_rep<1,pointer_w<T>,RowMajor >( arraytype_inclosure_t<1,T>& arg_clo )
     : array1d_rep_with_write<T>( arg_clo._data.ptr(), arg_clo._data.size() ) 
    {
    }
  };

  /* specialisation of array representation */
  template<typename T>
  class array_rep<1,pointer_rw<T>,RowMajor > : public array1d_rep_with_readwrite<T> {
  public:    
    array_rep<1,pointer_rw<T>,RowMajor >( arraytype_inclosure_t<1,T>& arg_clo )
     : array1d_rep_with_readwrite<T>( arg_clo._data.ptr(), arg_clo._data.size() ) 
    {
    }
  };
  

  template<int dim, typename T>
  struct TraitFormalParam<array<dim, pointer<T> > > { 
    typedef arraytype_inclosure_t<dim,T> type_inclosure_t;
    typedef ACCESS_MODE_RPWP   mode_t; 
    typedef T                  type_t;
    static const bool          is_static = false;
    static const void*         ptr( const array<dim, pointer<T> >* a ) { return a->ptr(); }
  };

  template<int dim, typename T>
  struct TraitFormalParam<array<dim, T> > { 
    typedef arraytype_inclosure_t<dim,T> type_inclosure_t;
    typedef ACCESS_MODE_RPWP   mode_t; 
    typedef T                  type_t;
    static const bool          is_static = false;
    static const void*         ptr( const array<dim, T>* a ) { return a->ptr(); }
    static size_t              get_nparam( const type_inclosure_t* a ) { return a->size(); }
  };

  template<int dim, typename T>
  struct TraitFormalParam<array<dim, pointer_r<T> > > { 
    typedef arraytype_inclosure_t<dim,T> type_inclosure_t;
    typedef array<dim, R<T> >         signature_t; 
    typedef array<dim, pointer_r<T> > formal_t; 
    typedef ACCESS_MODE_R             mode_t; 
    typedef T                         type_t;
    static const bool                 is_static = false;
    static const void*                get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->_data[i]; }
    static const void*                get_version( const type_inclosure_t* a, unsigned int i ) { return &a->_version[i]; }
    static void                       get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) 
    { r->data = (void*)&a->_data[i]; r->version = (void*)&a->_version[i]; }
    static void                       set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) 
    { if( &a->_data[i] != (type_t*)r->data)
        a->_data[i] = *(type_t*)r->data; 
      a->_version[i] = (type_t*)r->version; 
    }
    static size_t                     get_nparam( const type_inclosure_t* a ) { return a->size(); }
    static size_t                     get_size_param( const type_inclosure_t* a, unsigned int i ) 
    { return TraitFormalParam<type_t>::get_size_param( &a->_data[i], 0); }
    static void                       reducor_fnc(void*, const void*) {}
    static void                       redinit_fnc(void*) {}
   };

  template<int dim, typename T>
  struct TraitFormalParam<array<dim, pointer_w<T> > > { 
    typedef arraytype_inclosure_t<dim,T> type_inclosure_t;
    typedef array<dim, W<T> >         signature_t; 
    typedef array<dim, pointer_w<T> > formal_t; 
    typedef ACCESS_MODE_W             mode_t; 
    typedef T                         type_t;
    static const bool                 is_static = false;
    static const void*                get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->_data[i]; }
    static const void*                get_version( const type_inclosure_t* a, unsigned int i ) { return &a->_version[i]; }
    static void                       get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) 
    { r->data = (void*)&a->_data[i]; r->version = (void*)&a->_version[i]; }
    static void                       set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) 
    { if( &a->_data[i] != (type_t*)r->data)
        a->_data[i] = *(type_t*)r->data; 
      a->_version[i] = (type_t*)r->version; 
    }
    static size_t                     get_nparam( const type_inclosure_t* a ) { return a->size(); }
    static size_t                     get_size_param( const type_inclosure_t* a, unsigned int i ) 
    { return TraitFormalParam<type_t>::get_size_param( &a->_data[i], 0); }
    static void                       reducor_fnc(void*, const void*) {}
    static void                       redinit_fnc(void*) {}
  };

  template<int dim, typename T>
  struct TraitFormalParam<array<dim, pointer_rw<T> > > { 
    typedef arraytype_inclosure_t<dim,T> type_inclosure_t;
    typedef array<dim, RW<T> >        signature_t; 
    typedef array<dim, pointer_rw<T> > formal_t; 
    typedef ACCESS_MODE_RW            mode_t; 
    typedef T                         type_t;
    static const bool                 is_static = false;
    static const void*                get_data   ( const type_inclosure_t* a, unsigned int i ) { return &a->_data[i]; }
    static const void*                get_version( const type_inclosure_t* a, unsigned int i ) { return &a->_version[i]; }
    static void                       get_access ( const type_inclosure_t* a, unsigned int i, kaapi_access_t* r ) 
    { r->data = (void*)&a->_data[i]; r->version = (void*)&a->_version[i]; }
    static void                       set_access ( type_inclosure_t* a, unsigned int i, const kaapi_access_t* r ) 
    { if( &a->_data[i] != (type_t*)r->data)
        a->_data[i] = *(type_t*)r->data; 
      a->_version[i] = (type_t*)r->version; 
    }
    static size_t                     get_nparam ( const type_inclosure_t* a ) { return a->size(); }
    static size_t                     get_size_param( const type_inclosure_t* a, unsigned int i ) 
    { return TraitFormalParam<type_t>::get_size_param( &a->_data[i], 0); }
    static void                       reducor_fnc(void*, const void*) {}
    static void                       redinit_fnc(void*) {}
  };   


  template<int dim, typename T>
  struct TraitFormalParam<array<dim, R<T> > > : public TraitFormalParam<array<dim, pointer_r<T> > > {};
  template<int dim, typename T>
  struct TraitFormalParam<array<dim, W<T> > > : public TraitFormalParam<array<dim, pointer_w<T> > > {};
  template<int dim, typename T>
  struct TraitFormalParam<array<dim, RW<T> > > : public TraitFormalParam<array<dim, pointer_rw<T> > > {};
  
  /* ------ */
//  template<typename UserType>
//  template<typename T>
//  struct TraitFormalParam<pointer_rw<T>& > : public TraitFormalParam<pointer_rw<T> > { 
//  };

#if 0 // no more used: detection of correct passing rule using both effective and formal type.
  template<bool isfunc, class UserType> struct __kaapi_pointer_switcher {};

  template<class UserType> struct __kaapi_pointer_switcher<false, const UserType*> {
    typedef TraitFormalParam<pointer_rp<UserType> >   tfp_t;
  };
  template<class UserType> struct __kaapi_pointer_switcher<false, UserType*> {
    typedef TraitFormalParam<pointer_rpwp<UserType> > tfp_t;
  };
  template<class UserType> struct __kaapi_pointer_switcher<true, const UserType*> {
    typedef TraitFormalParam<Value<UserType*> >        tfp_t;
   };
  template<class UserType> struct __kaapi_pointer_switcher<true, UserType*> {
    typedef TraitFormalParam<Value<UserType*> >        tfp_t;
  };
  
  /* to be able to use pointer to data as arg of spawn. If it is a pointer to function, 
     consider it as a pass-by-value passing rule
  */
  template<typename UserType>
  struct TraitFormalParam<const UserType*> : public 
       __kaapi_pointer_switcher< __kaapi_is_function<const UserType*>::value, const UserType*>::tfp_t
  {
  };

  template<typename UserType>
  struct TraitFormalParam<UserType*> : public 
    __kaapi_pointer_switcher< __kaapi_is_function<UserType*>::value, UserType*>::tfp_t
  {
  };
#endif

  /* used to initialize representation into a closure from effective parameter */
  template<class E, class F, class InClosure>
  struct ConvertEffective2InClosure {
    static void doit(InClosure* inclo, const E& e) { new (inclo) InClosure(e); }
  };


  // --------------------------------------------------------------------  
  /* WARNING WARNING WARNING
     Attribut is responsible for pushing or not closure into the stack thread 
  */
  class DefaultAttribut {
  public:
    void* operator()( kaapi_thread_t* thread, kaapi_task_t* clo) const
    { 
      if (thread->tasklist ==0)
        kaapi_thread_pushtask(thread); 
      else
        kaapi_thread_pushtask_withpartitionid(thread, -1);
      return 0; 
    }
  };
  extern DefaultAttribut SetDefault;
  
  /* The only attribut that can be passed to task creation:
  
  */
  class AttributSchedTask {
    int   _partition;   // logical partition id
  public:
    AttributSchedTask( int s ) : _partition(s) {}
    void* operator()( kaapi_thread_t* thread, kaapi_task_t* clo) const
    { 
      kaapi_thread_pushtask_withpartitionid(thread, _partition);
      return 0;
    }
  };

  inline AttributSchedTask SetPartition( int s )
  { return AttributSchedTask(s); }
  
  // --------------------------------------------------------------------
  // A task forked with SetStaticSched attribut may have first formal
  // parameter a StaticSchedInfo returned by the runtime
  class StaticSchedInfo : public kaapi_staticschedinfo_t {
  public:
    StaticSchedInfo() {}

    /* return the total number of core */
    size_t count() const 
    { size_t retval = 0;
      for (int i=0; i<KAAPI_PROC_TYPE_MAX-1; ++i)
        retval += nkproc[i];
      return retval;
    }
    
    /* return the number of cpu */
    size_t count_cpu() const { return nkproc[KAAPI_PROC_TYPE_CPU-1]; }
    
    /* return the number of gpu */
    size_t count_gpu() const { return nkproc[KAAPI_PROC_TYPE_GPU-1]; }
  };

  // --------------------------------------------------------------------
  /* Static Sched attribut pass to the runtime
     Allows the user to specified number of ressources (at least 
     number threads) which may be scheduled on typed ressources.
     - 4 kinds of constraints that can be specify by the user:
      1. total number of ressources: anonymous ressources that can be scheduled
      on CPU or GPU.
        The user asks for a given number of threads that will be scheduled
        on the available ressources (CPU or GPU) depending of its tasks and
        available ressources at runtime. 
      2. total number of CPUs and (not inclusive) total number of GPUs.
        In that case, the user asks for a fixed number of Ncpus threads and Ngpus threads
        that will be scheduled on the specified architecture.
        Because if CPU is always available, it is possible that no GPU is available.
        It is strongly recommanded that the user tests the number of ressources passed from
        the runtime to the user if it declares its tasks with SchedInfo attribute.
      3. AutoCPU | AutoGPU: detection by the runtime of the number of available ressources.
        Using this specification, the user will received in its schedinfo task's parameter
        the number of available ressources for its task's execution. The number of ressources
        given by the runtime will never exeed the physical number of ressources.
      4. No specification: is equivalent to AutoCPU | AutoGPU.

      In all the cases, if the physical ressources are less than the requested number, then the user's 
      threads will be scheduled non preemptively on the physical ressources.
      
      All requested ressources or the ressources allocated by the runtime are lineary numbered
      from 0 to N-1, where N is the total number of ressources. The first set of ressources
      from 0 to Ncpu-1 corresponds to the CPU ressources. The second set of ressources from
      Ncpu to Ngpu+Ncpu-1 corresponds to the GPU ressources. Both Ncpu and Ngpu are accessible
      through the sched info data structure.
      
      Implementation note.
      -1: means auto detect ressources
      -2: means all ressources of the given type
  */
  class SetStaticSchedAttribut {
    int16_t  _nress;  /* N ressources are requested by the user, -1: fixed by the runtime */
    int16_t  _ncpu;   /* N CPU requested by the user, -1 fixed by the runtime */
    int16_t  _ngpu;   /* N GPU requested by the user, -1 fixed by the runtime */
  public:
    SetStaticSchedAttribut( int nc, int ng )
     : _nress(-1), _ncpu(nc), _ngpu(ng)
    {}
    /* format for N ressources */
    SetStaticSchedAttribut( int n )
     : _nress(n), _ncpu(-1), _ngpu(-1)
    {}
    SetStaticSchedAttribut( )
     : _nress(-1), _ncpu(-1), _ngpu(-1)
    {}
    void* operator()( kaapi_thread_t* thread, kaapi_task_t* clo) const
    { 
      /* push a task that will encapsulated the execution of the top task */
      kaapi_task_t* task = kaapi_thread_toptask(thread);
      kaapi_staticschedtask_arg_t* arg 
        = (kaapi_staticschedtask_arg_t*)kaapi_thread_pushdata( thread, sizeof(kaapi_staticschedtask_arg_t) );
      arg->sub_sp   = task->sp;
      arg->sub_body = (kaapi_task_vararg_body_t)kaapi_task_getuserbody(task);
      arg->schedinfo.nkproc[0]                   = (uint32_t)_nress;
      arg->schedinfo.nkproc[KAAPI_PROC_TYPE_GPU] = (uint32_t)_ngpu;
      arg->schedinfo.nkproc[KAAPI_PROC_TYPE_GPU] = (uint32_t)_ngpu;
      kaapi_task_initdfg(task, kaapi_staticschedtask_body, arg);
      kaapi_thread_pushtask(thread);
      return 0;
    }
  };
  
  /* */
  struct _KaapiCPU_Encode {
    _KaapiCPU_Encode() : ncpu(-1) {}
    _KaapiCPU_Encode( int16_t nc ) : ncpu(nc) {}
    int16_t ncpu;
  };

  /* */
  struct _KaapiGPU_Encode {
    _KaapiGPU_Encode() : ngpu(-1) {}
    _KaapiGPU_Encode( int16_t ng ) : ngpu(ng) {}
    int16_t ngpu;
  };

  /* */
  struct _KaapiCPUGPU_Encode : public _KaapiCPU_Encode, public _KaapiGPU_Encode {
    _KaapiCPUGPU_Encode() {}
    _KaapiCPUGPU_Encode( _KaapiCPU_Encode nc ) 
     : _KaapiCPU_Encode(nc) {}
    _KaapiCPUGPU_Encode( _KaapiGPU_Encode ng ) 
     : _KaapiGPU_Encode(ng) {}
    _KaapiCPUGPU_Encode( _KaapiCPU_Encode nc, _KaapiGPU_Encode ng ) 
     : _KaapiCPU_Encode(nc), _KaapiGPU_Encode(ng) {}
  };

  static const _KaapiCPU_Encode AutoCPU = _KaapiCPU_Encode();
  static const _KaapiGPU_Encode AutoGPU = _KaapiGPU_Encode();
  static const _KaapiCPU_Encode AllCPU  = _KaapiCPU_Encode(-2);
  static const _KaapiGPU_Encode AllGPU  = _KaapiGPU_Encode(-2);
  inline _KaapiCPU_Encode SetnCPU(int nc) { return _KaapiCPU_Encode(nc); }
  inline _KaapiGPU_Encode SetnGPU(int ng) { return _KaapiGPU_Encode(ng); }

  inline _KaapiCPUGPU_Encode operator|( const _KaapiCPU_Encode nc, const _KaapiGPU_Encode ng )
  { return _KaapiCPUGPU_Encode(nc,ng); }
  inline _KaapiCPUGPU_Encode operator|( const _KaapiGPU_Encode ng, const _KaapiCPU_Encode nc )
  { return _KaapiCPUGPU_Encode(nc,ng); }

  /* user level attribut definition */
  inline SetStaticSchedAttribut SetStaticSched(int n)
  { return SetStaticSchedAttribut(n); }
  inline SetStaticSchedAttribut SetStaticSched( _KaapiCPU_Encode xx )
  { return SetStaticSchedAttribut(xx.ncpu, 0); }
  inline SetStaticSchedAttribut SetStaticSched( _KaapiGPU_Encode xx )
  { return SetStaticSchedAttribut(0, xx.ngpu); }
  inline SetStaticSchedAttribut SetStaticSched( _KaapiCPUGPU_Encode xx )
  { return SetStaticSchedAttribut(xx.ncpu, xx.ngpu); }
  inline SetStaticSchedAttribut SetStaticSched()
  { return SetStaticSchedAttribut(); }


  // --------------------------------------------------------------------
  /* Mapping of data into logical set of partitions
  */
  struct BlockCyclic {
  };
  struct Block {
  };

  template<typename DIST, typename T, bool isaccess>
  struct Mapping2DHelper {
  };

  template<typename T>
  struct Mapping2DHelper<Block,T,true> 
  {
    /* assume that T implements the range2D interface */
    static void map( const T& C, size_t n_bloc_i, size_t n_bloc_j )
    {
      std::cout << __PRETTY_FUNCTION__ << " TODO" << std::endl;
      int dim0 __attribute__((unused))= C.dim(0);
      int dim1 __attribute__((unused))= C.dim(1);
    }
  };

  template<typename T>
  struct Mapping2DHelper<BlockCyclic,T,true> 
  {
    /* assume that T implements the range2D interface */
    static void map( const T& C, size_t sz_bloc_i, size_t sz_bloc_j )
    {
      std::cout << __PRETTY_FUNCTION__ << " TODO" << std::endl;
      int dim0 __attribute__((unused))= C.dim(0);
      int dim1 __attribute__((unused))= C.dim(1);
    }
  };

  template<typename DIST>
  struct Mapping2D {
  };

  template<>
  struct Mapping2D<Block> 
  {
    /* assume that T implements the range2D interface */
    template<class T>
    static void map( const T& C, size_t n_bloc_i, size_t n_bloc_j )
    {
      Mapping2DHelper<Block,T,IsAccessMode<typename TraitFormalParam<T>::mode_t>::value>::map( C, n_bloc_i, n_bloc_j );
    }
  };

  template<>
  struct Mapping2D<BlockCyclic> 
  {
    /* assume that T implements the range2D interface */
    template<class T>
    static void map( const T& C, size_t sz_bloc_i, size_t sz_bloc_j )
    {
      Mapping2DHelper<BlockCyclic,T,IsAccessMode<typename TraitFormalParam<T>::mode_t>::value>::map( C, sz_bloc_i, sz_bloc_j );
    }
  };


  // --------------------------------------------------------------------
  /* OwnerComputeRule attribut: specify, during fork that a task should
     have the same site location than the site location of a data.
     Usage: 
     * Spawn<Task>( ka::OCR( <here a ka_pointer_XX> ) (... ): try to get site from the ka_pointer_xx.
     * Spawn<Task>( ka::OCR( <here a C++ pointer> ) (... ): try to get site from mapping of application data.
     * Spawn<Task>( ka::OCR( <any other type> ) (... ): no effect.
  */
  template<typename T, bool isaccess>
  struct OCRAttribut;

  template<typename T>  
  struct OCRAttribut<T,false> { /* not a ka::pointer type, do nothing */
    OCRAttribut<T,false>(const T* a) {}
    void* operator()( kaapi_thread_t* thread, kaapi_task_t* clo) const
    { 
      kaapi_thread_pushtask(thread); 
      return 0;
    }
  };

  template<typename T>  /* case of a ka::pointer type */
  struct OCRAttribut<T,true> {
    const void* _ptr;
    OCRAttribut<T,true>(const T* a)
     : _ptr( (a == 0 ? 0 : a->ptr()) )
    { 
    }
    void* operator()( kaapi_thread_t* thread, kaapi_task_t* clo) const
    { 
      kaapi_thread_pushtask_withocr(thread, _ptr); 
      return 0;
    }
  };

  template<typename T>  
  struct OCRAttribut<T*,false> {
    const void* _ptr;
    OCRAttribut<T*,false>(const T* const* a)
     : _ptr( (const void*)a )
    { }    
    void* operator()( kaapi_thread_t* thread, kaapi_task_t* clo) const
    { 
      kaapi_thread_pushtask_withocr(thread, _ptr); 
      return 0;
    }
  };
  template<typename T>  
  struct OCRAttribut<const T*,false> {
    const void* _ptr;
    OCRAttribut<const T*,false>(const T* const* a)
     : _ptr( (const void*)a )
    { }    
    void* operator()( kaapi_thread_t* thread, kaapi_task_t* clo) const
    { 
      kaapi_thread_pushtask_withocr(thread, _ptr); 
      return 0;
    }
  };
  
  template<typename T>
  inline OCRAttribut<T, IsAccessMode<typename TraitFormalParam<T>::mode_t>::value > OCR( const T& a )
  { return OCRAttribut<T, IsAccessMode<typename TraitFormalParam<T>::mode_t>::value >(&a); }


  // --------------------------------------------------------------------
  template<class F>
  struct TraitIsOut {
    enum { value = 0 };
  };
  template<class F>
  struct TraitIsOut<F&> {
    enum { value = 1 };
  };
  

  // --------------------------------------------------------------------
  /* for better understand error message */
  template<int i>
  struct FOR_ARG {};

  /* for better understand error message */
  template<class TASK>
  struct FOR_TASKNAME {};
  
  /* ME: effectif -> MF: formal */
  template<class ME, class MF, class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE {
//    static void IS_COMPATIBLE();
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_V, ACCESS_MODE_V, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_R, ACCESS_MODE_R, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK> /* this rule is only valid for terminal fork... */
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_W, ACCESS_MODE_W, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_CW, ACCESS_MODE_CW, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_RPWP, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_RW, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_WP, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_W, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_CWP, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_CW, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_R, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, ACCESS_MODE_RP, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RP, ACCESS_MODE_RP, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RP, ACCESS_MODE_R, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_WP, ACCESS_MODE_WP, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_WP, ACCESS_MODE_W, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_CWP, ACCESS_MODE_CW, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class E, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_CWP, ACCESS_MODE_CWP, E, F, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };

  /* specialization if effective mode is C++ pointer + formal is kaapi pointer */
  template<class MF, class T, class F, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_V, MF, T*, F, PARAM, TASK> 
    : WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_RPWP, MF, T*, F, PARAM, TASK> 
  {
  };

  /* specialization if effective mode is C++ pointer + formal parameter is a C++ pointer */
  template<class T, class PARAM, class TASK>
  struct WARNING_UNDEFINED_PASSING_RULE<ACCESS_MODE_V, ACCESS_MODE_V, T*, T*, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };

  /* required for most of stl like parallel algorithm
  */
  template <typename type >
  class counting_iterator : public std::iterator< 
      std::random_access_iterator_tag,     /* category */
      const type                           /* element type */                                            
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
      counting_iterator& operator+=(int i) 
      { 
        _rep+=i;
        return *this;
      }
      counting_iterator& operator+=(size_t i) 
      { 
        _rep+=i;
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
      counting_iterator& operator-=(int i) 
      { 
        _rep -=i;
        return *this;
      }
      counting_iterator& operator-=(size_t i) 
      { 
        _rep -=i;
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

      pointer operator->() 
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
    static void body( kaapi_task_t* task, kaapi_thread_t* stack )
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
      void* operator()()
      { 
        typedef FormatClosure0<TASK> KaapiFormatTask_t;
        kaapi_task_t* clo = kaapi_thread_toptask( _thread );
        kaapi_task_initdfg( clo, KaapiFormatTask_t::default_bodies.cpu_body, 0 );
        /* attribut is reponsible for pushing task into the thread */
        return _attr(_thread, clo);
      }

#include "ka_api_spawn.h"

    protected:
      kaapi_thread_t* _thread;
      const Attr&     _attr;
    };

    template<class TASK>
    Spawner<TASK, DefaultAttribut> Spawn() 
    { return Spawner<TASK, DefaultAttribut>(&_thread, DefaultAttribut()); }

    template<class TASK, class Attr>
    Spawner<TASK, Attr> Spawn(const Attr& a) 
    { return Spawner<TASK, Attr>(&_thread, a); }

  protected:
    kaapi_thread_t _thread;
    friend class SyncGuard;
  };


  // --------------------------------------------------------------------
  class Request;

  /* Stealcontext */
  class StealContext {
  public:
    template<class OBJECT>
    void set_splitter(
          kaapi_task_splitter_t splitter,
          OBJECT*               arg
    )
    { kaapi_steal_setsplitter(&_sc, splitter, arg); }

    /* test preemption on the steal context */
    bool is_preempted() const
    { return kaapi_preemptpoint_isactive(&_sc) != 0; }

    /* return a pointer to args passed by the victim */
    template<class T>
    const T& victim_arg_preemption()
    { 
      kaapi_assert_debug( sizeof(T) <= _sc.sz_data_victim );
      return *static_cast<const T*> (_sc.data_victim); 
    }

    /* return a pointer to args that will be passed to the victim */
    template<class T>
    T& arg_preemption()
    { 
      kaapi_assert_debug( sizeof(T) <= _sc.sz_data_victim );
      return *static_cast<T*> (_sc.data_victim); 
    }

    /* return a pointer to a memory zone that will be received
       by the victim upon the acknowledge will be send.
    */
    template<class T>
    T* return_arg();

    /* signal the victim that it can get */
    void ack_preemption()
    { kaapi_preemptpoint_after_reducer_call(&_sc); }
    
    /* thief iterator: forward iterator */
    struct thief_iterator {
      struct DUMMY_TYPE {};
      
      struct one_thief {
        /* signal_preempt: send preemption signal to the thief.
           When the victim caller send a signal to one of its thief,
           the control flow continues (it is a non blocking operation).
           The returns value depend of the state of the thief. If the thief
           is detected 'finished' before sending the preemption flag, then
           the return value is ECHILD (no child). Else the return values is
           0.
           \retval 0 iff the signal was correctly send to an active thief
           \retval ECHILD iff the thief was finished before the signal occurs
           \retval an error code
        */
        int signal_preempt()
        { return kaapi_preemptasync_thief(_sc, _ktr, 0); }

        /* signal_preempt: send preemption signal to the thief.
           When the victim caller send a signal to one of its thief,
           the control flow continues (it is a non blocking operation).
           The returns value depend of the state of the thief. If the thief
           is detected 'finished' before sending the preemption flag, then
           the return value is ECHILD (no child). Else the return values is
           0.
           The victim may use the variation signal_preempt( arg ) with a
           extra args to pass to the thief. This value is possibly recopied
           and should not change until the victim wait the ack from the thief
           using wait_preempt. 
           \retval 0 iff the signal was correctly send to an active thief
           \retval ECHILD iff the thief was finished before the signal occurs
           \retval an error code
        */
        template<class T>
        int signal_preempt( const T* arg )
        { return kaapi_preemptasync_thief(_sc, _ktr, 0); }

        /* wait_preempt: waits until the thief has received the preemption flag. 
           The caller is suspended until the thief has received the preemption flag
           and has reply to the preemption request. The return value is a pointer
           to the memory region reserved when ones replied to steal requests.
           The value is stored by the thief when it processes the preemption request
           (see ).
           \retval a pointer to data passed by the thief for the victim
        */
        void wait_preempt()
        {
          kaapi_preemptasync_waitthief(_sc,_ktr);
        }

        template<class R>
        R* wait_preempt()
        {
          kaapi_preemptasync_waitthief(_sc,_ktr);
          return (R*)_ktr->data;
        }
      
        /* private part */
        one_thief( kaapi_stealcontext_t* sc, kaapi_taskadaptive_result_t* ktr)
         : _sc(sc), _ktr(ktr)
        {}
        kaapi_stealcontext_t*        _sc;
        kaapi_taskadaptive_result_t* _ktr;
      };
      
      /* */
      one_thief* operator->()
      { return &curr; }

      /* */
      bool operator==(const thief_iterator& i) const
      { return (curr._sc == i.curr._sc) && (curr._ktr == i.curr._ktr); }

      /* */
      bool operator!=(const thief_iterator& i) const
      { return (curr._sc != i.curr._sc) || (curr._ktr != i.curr._ktr); }
  
      /* prefix op*/
      thief_iterator& operator++()
      {
        curr._ktr = kaapi_get_next_thief(curr._ktr);
        return *this;
      }
    protected:
      friend class StealContext;
      thief_iterator( kaapi_stealcontext_t* sc, kaapi_taskadaptive_result_t* ktr)
       : curr(sc, ktr)
      {}
      one_thief curr;
    };
    
    thief_iterator begin_thief()
    {
      return thief_iterator(&_sc, kaapi_get_thief_head(&_sc) );
    }
    thief_iterator end_thief()
    {
      return thief_iterator(&_sc, 0 );
    }
  protected:
    kaapi_stealcontext_t _sc;
    friend class Request;
  };
  
  
  /**
  */
  struct FlagReplyHead {};
  extern FlagReplyHead ReplyHead;
  struct FlagReplyTail {};
  extern FlagReplyTail ReplyTail;

  /* New API: request->Spawn<TASK>(sc)( args ) for adaptive tasks
  */
  class Request {
  private:
    Request() {}

    template<class TASK>
    class Spawner {
    protected:
      Spawner( kaapi_request_t* r, kaapi_stealcontext_t* sc ) 
        : _req(r), _sc(sc), _flag(KAAPI_REQUEST_REPLY_HEAD) 
      {}
      Spawner( kaapi_request_t* r, kaapi_stealcontext_t* sc, int flag_push ) 
        : _req(r), _sc(sc), _flag(flag_push) 
      {}

    public:
      /**
      */      
      void operator()()
      { 
        void* arg __attribute__((unused))
            =kaapi_reply_init_adaptive_task( _sc, _req, KaapiTask0<TASK>::body, 0, 0 );
        kaapi_request_reply(_sc, _req, _flag );
      }

#include "ka_api_reqspawn.h"

    protected:
      kaapi_request_t*      _req;
      kaapi_stealcontext_t* _sc;
      int                   _flag;
      friend class Request;
    };

    template<class TASK, class OUTSTATE>
    class SpawnerKtr {
    protected:
      SpawnerKtr( kaapi_request_t* r, kaapi_stealcontext_t* sc ) 
        : _req(r), _sc(sc), _flag(KAAPI_REQUEST_REPLY_HEAD) 
      {}
      SpawnerKtr( kaapi_request_t* r, kaapi_stealcontext_t* sc, int flag_push ) 
        : _req(r), _sc(sc), _flag(flag_push) 
      {}

    public:
      /**
      */      
      void operator()()
      { 
        void* arg __attribute__((unused))
            =kaapi_reply_init_adaptive_task( _sc, _req, KaapiTask0<TASK>::body, 0, 0 );
        kaapi_request_reply(_sc, _req, _flag );
      }

#include "ka_api_reqspawn.h"

    protected:
      kaapi_request_t*      _req;
      kaapi_stealcontext_t* _sc;
      int                   _flag;
      friend class Request;
    };

  public:
    template<class TASK>
    Spawner<TASK> Spawn(StealContext* sc) { return Spawner<TASK>(&_request, (kaapi_stealcontext_t*)sc); }
    template<class TASK>
    Spawner<TASK> Spawn(StealContext* sc, FlagReplyHead flag) 
    { return Spawner<TASK>(&_request, (kaapi_stealcontext_t*)sc, KAAPI_REQUEST_REPLY_HEAD); }
    template<class TASK>
    Spawner<TASK> Spawn(StealContext* sc, FlagReplyTail flag) 
    { return Spawner<TASK>(&_request, (kaapi_stealcontext_t*)sc, KAAPI_REQUEST_REPLY_TAIL); }

    template<class TASK, class OUTSTATE>
    SpawnerKtr<TASK,OUTSTATE> Spawn(StealContext* sc) 
    { return SpawnerKtr<TASK,OUTSTATE>(&_request, (kaapi_stealcontext_t*)sc); }
    template<class TASK, class OUTSTATE>
    SpawnerKtr<TASK,OUTSTATE> Spawn(StealContext* sc, FlagReplyHead flag) 
    { return SpawnerKtr<TASK,OUTSTATE>(&_request, (kaapi_stealcontext_t*)sc, KAAPI_REQUEST_REPLY_HEAD); }
    template<class TASK, class OUTSTATE>
    SpawnerKtr<TASK,OUTSTATE> Spawn(StealContext* sc, FlagReplyTail flag) 
    { return SpawnerKtr<TASK,OUTSTATE>(&_request, (kaapi_stealcontext_t*)sc, KAAPI_REQUEST_REPLY_TAIL); }

  protected:
    kaapi_request_t _request;
  };

  
  /* push new steal context */
  template<class OBJECT>
  inline int __kaapi_trampoline_lambda( kaapi_stealcontext_t* sc, int nreq, kaapi_request_t* req, void* arg )
  { OBJECT* o = (OBJECT*)arg; 
    (*o)( (StealContext*)sc, nreq, (Request*)req );
    return 0;
  }

  template<class OBJECT>
  inline StealContext* TaskBeginAdaptive(
        int flag,
        const OBJECT& func
  )
  { return (StealContext*)kaapi_task_begin_adaptive(
        kaapi_self_thread(), 
        flag, 
        __kaapi_trampoline_lambda<OBJECT>, 
        (void*)&func); 
  }

  /* wrapper to splitter method */
  template<typename OBJECT, void (OBJECT::*s)(StealContext*, int, Request*)>
  struct Wrapper {
    static int splitter( kaapi_stealcontext_t* sc, int nreq, kaapi_request_t* req, void* arg )
    { OBJECT* o = (OBJECT*)arg; 
      (o->*s)( (StealContext*)sc, nreq, (Request*)req );
      return 0;
    };
  };

  template<typename OBJECT, void (OBJECT::*s)(StealContext*, int, Request*)>
  int WrapperSplitter(kaapi_stealcontext_t* sc, int nreq, kaapi_request_t* req, void* arg )
  { 
    OBJECT* o = (OBJECT*)arg; 
    (o->*s)( (StealContext*)sc, nreq, (Request*)req );
    return 0;
  }

  template<class OBJECT>
  inline StealContext* TaskBeginAdaptive(
        int                   flag,
        kaapi_task_splitter_t splitter,
        OBJECT*               arg
  )
  { return (StealContext*)kaapi_task_begin_adaptive(kaapi_self_thread(), flag, splitter, arg); }

  template<class OBJECT>
  inline StealContext* TaskBeginAdaptive(
        int                   flag
  )
  { return (StealContext*)kaapi_task_begin_adaptive(kaapi_self_thread(), flag, 0, 0); }

  inline void TaskEndAdaptive( StealContext* sc )
  { kaapi_task_end_adaptive((kaapi_stealcontext_t*)sc); }


  // --------------------------------------------------------------------
  template<typename Mapper>
  struct WrapperMapping {
    static kaapi_address_space_id_t mapping_function( void* arg, int nodecount, int tid )
    {
      Mapper* mapper = static_cast<Mapper*> (arg);
      return (kaapi_address_space_id_t)(*mapper)(nodecount, tid);
    }
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

    ThreadGroup(size_t sz) 
     : _size(sz), _created(false)
    {
    }
    void resize(size_t sz)
    { _size = sz; }
    
    size_t size() const
    { return _size; }

    /* begin to partition task */
    void begin_partition( int flag =KAAPI_THGRP_DEFAULT_FLAG)
    {
      if (!_created) 
      { 
        kaapi_threadgroup_create( &_threadgroup, (uint32_t)_size, 
                0, 0 ); 
        _created = true; 
      }
      kaapi_threadgroup_begin_partition( _threadgroup, flag );
      kaapi_set_threadgroup(_threadgroup);
    }

    template<typename Mapping>
    void begin_partition( Mapping& mapobj, int flag =KAAPI_THGRP_DEFAULT_FLAG)
    {
      if (!_created) 
      { 
        kaapi_threadgroup_create( &_threadgroup, (uint32_t)_size, 
                &WrapperMapping<Mapping>::mapping_function, 
                &mapobj ); 
        _created = true; 
      }
      kaapi_threadgroup_begin_partition( _threadgroup, flag );
      kaapi_set_threadgroup(_threadgroup);
    }

    /* begin to partition task */
    void set_iteration_step( int maxstep )
    {
//      kaapi_threadgroup_set_iteration_step( _threadgroup, maxstep );
    }

    void force_archtype(unsigned int part, unsigned int type)
    {
      kaapi_threadgroup_force_archtype(_threadgroup, part, type);
    }

    void force_kasid(unsigned int part, unsigned int arch, unsigned int user)
    {
      kaapi_threadgroup_force_kasid(_threadgroup, part, arch, user);
    }

    /* destroy */
    void destroy( )
    {
      kaapi_threadgroup_destroy( _threadgroup );
    }

    /* internal class required for spawn method */
    class AttributComputeDependencies {
    public:
      AttributComputeDependencies( kaapi_threadgroup_t thgrp, int thid ) 
       : _threadgroup(thgrp), _threadindex(thid) 
      {}
      void* operator()(kaapi_thread_t* thread, kaapi_task_t* task)
      { 
        kaapi_threadgroup_computedependencies( _threadgroup, _threadindex, task );
        return 0;
      }
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
      void* operator()()
      { 
        kaapi_task_t* clo = kaapi_thread_toptask( _thread );
        kaapi_task_initdfg( clo, KaapiTask0<TASK>::body, 0 );
        /* attribut is reponsible for pushing */
        return _attr(_thread, clo);
      }

#include "ka_api_spawn.h"

    protected:
      AttributComputeDependencies _attr;
      kaapi_thread_t*             _thread;
    };  


    /* Interface: threadgroup.Spawn<TASK>(SetPartition(i) [, ATTR])( args ) */
    template<class TASK>
    Spawner<TASK> Spawn(const AttributSchedTask& a) 
    { return Spawner<TASK>(
                  AttributComputeDependencies(_threadgroup, a._partition),
                  kaapi_threadgroup_thread(_threadgroup, a._partition)
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


    /* ForEachDriverLoop: specialized only for random access iterator */
    template<class TASKGENERATOR, typename Iterator, typename tag>
    class ForEachDriverLoop {};

    template<class TASKGENERATOR, typename Iterator>
    class ForEachDriverLoop<TASKGENERATOR,Iterator,std::random_access_iterator_tag> {
    public:
      ForEachDriverLoop(ThreadGroup* thgrp, Iterator beg, Iterator end)
       : _threadgroup(thgrp), _beg(beg), _end(end), step(0), total(0)
      {}
      void prologue()
      {
        /* flag to save */
        _threadgroup->begin_partition( KAAPI_THGRP_SAVE_FLAG );
        _threadgroup->set_iteration_step( _end-_beg );
        tpart = kaapi_get_elapsedtime();
      }

      void epilogue()
      {
        tpart = kaapi_get_elapsedtime()-tpart;
        _threadgroup->end_partition();
        s0 = kaapi_get_elapsedtime();
        while (_beg != _end)
        {
          t0 = kaapi_get_elapsedtime();
          _threadgroup->execute();
          t1 = kaapi_get_elapsedtime();
          if (step >0) total += t1-t0;
          std::cout << step << ":: Time: " << t1 - t0 << std::endl;
          ++step;
          ++_beg;
        }
        s1 = kaapi_get_elapsedtime();
        std::cout << ":: ForEach #loops: " << step << ", total time (except first iteration):" << total
                  << ", average:" << total / (step-1) << ", partition step:" << tpart << std::endl;
      }

    protected:
      ThreadGroup*  _threadgroup;
      Iterator      _beg;
      Iterator      _end;
      int           step;
      double        tpart;
      double        s0, s1, t0,t1,total;
    };

    /** ForEachDriver */
    template<class TASKGENERATOR, typename Iterator>
    class ForEachDriver : 
        public ForEachDriverLoop<TASKGENERATOR, Iterator,typename std::iterator_traits<Iterator>::iterator_category> {
    public:
      ForEachDriver(ThreadGroup* thgrp, Iterator beg, Iterator end)
       : ForEachDriverLoop<TASKGENERATOR, Iterator, typename std::iterator_traits<Iterator>::iterator_category>(
          thgrp, beg, end)
      {}
      /** 0 args **/
      void operator()()
      {
        if (this->_beg == this->_end) return;
        this->prologue();
        TASKGENERATOR();
        this->epilogue();
      }
#include "ka_api_execforeach.h"
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
//      kaapi_threadgroup_begin_execute( _threadgroup );
//      kaapi_threadgroup_end_execute  ( _threadgroup );
    }

    /* save */
    void save()
    {
//      kaapi_threadgroup_save( _threadgroup );
    }

    /* restore */
    void restore()
    {
      kaapi_threadgroup_restore( _threadgroup );
    }
    
    /* memory synchronize */
    void synchronize()
    { 
      kaapi_threadgroup_synchronize(_threadgroup );
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

  template<class TASK, class Attr>
  Thread::Spawner<TASK, Attr> Spawn(const Attr& a) 
  { return Thread::Spawner<TASK, Attr>(kaapi_self_thread(), a); }




  // --------------------------------------------------------------------
  /** Wait execution of all forked tasks of the running task */
  extern void Sync();

  // --------------------------------------------------------------------
  /** Move into the local address space all global memory ops */
  extern void MemorySync();

  template<typename mode_t, typename T> 
  struct MemorySyncFuncClass { 
    static void doit(const T*a) 
    { kaapi_memory_synchronize(); } // HERE MUST CALL SPECIFIC FUNCTION WITH PTR 
  };

  /* specialization for non shared object */
  template<typename T> 
  struct MemorySyncFuncClass<ACCESS_MODE_V,T> { 
    static void doit(const T*a) 
    { }
  };

  template<typename T>
  extern void MemorySync(const T& a)
  { MemorySyncFuncClass<typename TraitFormalParam<T>::mode_t, T>::doit(&a); }

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
  
  template<class TASK, class Attr>
  struct SpawnerMain
  {
    SpawnerMain(kaapi_thread_t* t, const Attr& a)  : _thread(t), _attr(a) {}

    void operator()( int argc, char** argv)
    {
      kaapi_task_t* clo = kaapi_thread_toptask( _thread );
      kaapi_taskmain_arg_t* arg = 
        (kaapi_taskmain_arg_t*)kaapi_thread_pushdata(_thread, sizeof(kaapi_taskmain_arg_t) );
      arg->argc = argc;
      arg->argv = argv;
      arg->mainentry = &MainTaskBodyArgcv<TASK>::body;
      kaapi_task_initdfg( clo, (kaapi_task_body_t)kaapi_taskmain_body, arg );
      _attr( _thread, clo );    
    }

    void operator()()
    {
      kaapi_task_t* clo = kaapi_thread_toptask( _thread );
      kaapi_taskmain_arg_t* arg = 
        (kaapi_taskmain_arg_t*)kaapi_thread_pushdata(_thread, sizeof(kaapi_taskmain_arg_t) );
      arg->argc = 0;
      arg->argv = 0;
      arg->mainentry = &MainTaskBodyNoArgcv<TASK>::body;
      kaapi_task_initdfg( clo, (kaapi_task_body_t)kaapi_taskmain_body, arg );
      _attr( _thread, clo );    
    }

    protected:
      kaapi_thread_t* _thread;
      const Attr&     _attr;
  };

  template<class TASK>
  SpawnerMain<TASK, DefaultAttribut> SpawnMain()
  { 
    return SpawnerMain<TASK, DefaultAttribut>(kaapi_self_thread(), DefaultAttribut());
  }
  template<class TASK, class Attr>
  SpawnerMain<TASK, Attr> SpawnMain(const Attr& a)
  { 
    return SpawnerMain<TASK, Attr>(kaapi_self_thread(), a);
  }
    
  // --------------------------------------------------------------------
  template<class T>
  struct TaskDelete : public Task<1>::Signature<RW<T> > { };
  // --------------------------------------------------------------------
  template<class T>
  struct TaskDestroy : public Task<1>::Signature<RW<T> > { };

} // namespace ka

  // --------------------------------------------------------------------
  template<typename TASK>
  struct TaskFormat {
    typedef typename TASK::Signature View; /* here should be defined using default interpretation of args */
  };

  template<class T> struct TaskBodyCPU<ka::TaskDelete<T> > {
    void operator() ( ka::Thread* thread, ka::pointer_rw<T> res )
    { delete &*res; }
  };

  template<class T> struct TaskBodyCPU<ka::TaskDestroy<T> > {
    void operator() ( ka::Thread* thread, ka::pointer_rw<T> res )
    { res->T::~T(); }
  };
  
namespace ka {
  
  template<bool noneedtaskdelete=false>
  struct SpawnDelete {
    template<class T> static void doit( pointer<T>& ap ) 
    { 
      Spawn<TaskDelete<T> >()(ap); 
#if !defined(KAAPI_NDEBUG)
      ap.ptr(0); 
#endif
    }
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

  template<bool noneedtaskdelete>
  struct SpawnDestroy {
    template<class T> static void doit( auto_variable<T>& av ) 
    { 
      Spawn<TaskDestroy<T> >()(&av); 
#if !defined(KAAPI_NDEBUG)
      av.clear(); 
#endif
    }
  };
  template<>
  struct SpawnDestroy<true> {
    template<class T> static void doit( auto_variable<T>& av ) 
    { 
#if !defined(KAAPI_NDEBUG)
      av.clear();
#endif
    }
  };


  template<class T>
  auto_pointer<T>::~auto_pointer()
  { SpawnDelete<>::doit(*this); }

  template<class T>
  auto_variable<T>::~auto_variable()
  { SpawnDestroy<TraitNoDeleteTask<T>::value>::doit(*this); }


  // --------------------------------------------------------------------
  class SyncGuard {
    kaapi_thread_t*  _thread;
    kaapi_frame_t    _frame;
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
inline ka::OStream& operator<< (ka::OStream& s_out, unsigned char c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, unsigned short c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, unsigned int c )
{ return s_out; }
inline ka::OStream& operator<< (ka::OStream& s_out, unsigned long c )
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
inline ka::IStream& operator>> (ka::IStream& s_in, unsigned char& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, unsigned short& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, unsigned int& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, unsigned long& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, float& c )
{ return s_in; }
inline ka::IStream& operator>> (ka::IStream& s_in, double& c )
{ return s_in; }


#if !defined(_KAAPIPLUSPLUS_NOT_IN_GLOBAL_NAMESPACE)
using namespace ka;
#endif

#endif
