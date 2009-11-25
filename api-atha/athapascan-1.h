/* KAAPI public interface */
// =========================================================================
// (c) INRIA, projet MOAIS, 2009
// Author: T. Gautier, X-Kaapi port
//
//
//
// =========================================================================
#ifndef _ATHAPASCAN_1_H_H
#define _ATHAPASCAN_1_H_H

// This is the new version on top of X-Kaapi
extern "C" { const char* get_kaapi_version(); }

#include "kaapi.h"
#include "atha_error.h"
#include <vector>

namespace atha{}

namespace a1 {

 /* take a constant... should be adjusted */
 enum { STACK_ALLOC_THRESHOLD = KAAPI_MAX_DATA_ALIGNMENT };  

  // --------------------------------------------------------------------
  namespace FormatDef {
#  define KAAPI_DECL_EXT_FORMAT(TYPE,OBJ)\
    extern const kaapi_format_t OBJ;
    
    extern const kaapi_format_t Null;
    KAAPI_DECL_EXT_FORMAT(bool, Bool)
    KAAPI_DECL_EXT_FORMAT(char, Char)
    extern const kaapi_format_t Byte;
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
  }
  typedef kaapi_format_t Format;
  class UpdateFunctionFormat;

  class IStream;
  class OStream;
  class ODotStream;
  
  using atha::Exception;
  using atha::RuntimeError;
  using atha::InvalidArgumentError;
  using atha::RestartException;
  using atha::ServerException;
  using atha::NoFound;
  using atha::BadAlloc;
  using atha::IOError;
  using atha::ComFailure;
  using atha::BadURL;

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
  class Thread;
  
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
  struct SetStack {};
  extern SetStack SetInStack;

  // --------------------------------------------------------------------
  struct SetHeap {};
  extern SetHeap SetInHeap;

  // --------------------------------------------------------------------
  class SetStickyC{};
  extern SetStickyC SetSticky;


  // --------------------------------------------------------------------
  template<class T>
  class Shared {
  public:
    typedef T value_type;
    operator kaapi_access_t&() { return _gd; }
    ~Shared ( ) 
    {
#if 0 /* optimize destructor: do nothing for basic type */
      destroy( stack );
#endif
    }
    
#if 0
    Shared ( value_type* data ) 
    {
      Thread* thread = System::get_current_thread(); 
      if (!data) 
      {
        data = 0;
        if (sizeof(value_type) <= STACK_ALLOC_THRESHOLD) 
        {
            attr.set_instack();
            data = new (thread->allocate(sizeof(value_type))) value_type;
        } else {
            attr.set_inheap();
#if defined(KAAPI_USE_NUMA)
              //WARN LAURENT : hack to sched
              data = new value_type;
#endif
          }
      }
      else
      {
          attr.set_inheap();
      }
      initialize( thread, data, &Util::WrapperFormat<value_type>::theformat, attr);
    }
#endif

#if 0
    Shared ( const SetStack& toto, value_type* data = 0) 
    {
      Thread* thread = System::get_current_thread(); 
      DFG::GlobalData::Attribut attr;
      attr.clear();
      attr.set_instack();
      if(!data) data = new (thread->allocate(sizeof(value_type))) value_type;
      initialize( thread, data, &Util::WrapperFormat<value_type>::theformat, attr);
    }
#endif

#if 0
    Shared ( const SetHeap& toto, value_type* data = 0) 
    {
      Thread* thread = System::get_current_thread(); 
      DFG::GlobalData::Attribut attr;
      attr.clear();
      attr.set_inheap();
      if(!data) data =
#if defined(KAAPI_USE_NUMA)
      //WARN LAURENT
        new value_type;
#else
        0;
#endif
      initialize( thread, data, &Util::WrapperFormat<value_type>::theformat, attr);
    }
#endif

    Shared()
    {
      kaapi_stack_t* stack = kaapi_self_stack();
      _gd = kaapi_stack_pushshareddata(stack,sizeof(T));
      new (_gd.data) T;
    }

    Shared(const value_type& value )
    {
      kaapi_stack_t* stack = kaapi_self_stack();
      _gd = kaapi_stack_pushshareddata(stack,sizeof(T));
      new (_gd.data) T(value);
    }

#if 0
    Shared(const SetStack& toto, const T& value )
    {
      Thread* thread = System::get_current_thread(); 
      _gd  = new (SharedAllocator, thread) T(value)
      _gd._attr = 0;
    }
#endif

#if 0
    Shared(const SetHeap& toto, const T& value )
    {
      _gd.data  = new (SharedAllocator) T(value)
      _gd._attr = 1;
    }
#endif

    Shared(const Shared<value_type>& t) 
     : _gd(t._gd)
    {
      t._gd.data    = 0;
      t._gd.version = 0;
    }

    Shared<T>& operator=(const Shared<value_type>& t) 
    {
      _gd = t._gd;
      t._gd.data    = 0;
      t._gd.version = 0;
      return *this;
    }

  private:
    kaapi_access_t _gd;
  };
  

  // --------------------------------------------------------------------
  template<class T>
  class Shared_rp {
  public:
    typedef T value_type;

    operator kaapi_access_t&() { return _gd; }
    Shared_rp( const kaapi_access_t& a )
     : _gd( a )
    { }
  protected:
    kaapi_access_t _gd;
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_r  {
  public:
    typedef T value_type;

    operator kaapi_access_t&() { return _gd; }
    Shared_r( const kaapi_access_t& a )
     : _gd( a )
    { }

    const value_type& read() const 
    { return *(T*)_gd.data; }
  protected:
    kaapi_access_t _gd;
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_wp  {
  public:
    typedef T value_type;

    operator kaapi_access_t&() { return _gd; }
    Shared_wp( const kaapi_access_t& a )
     : _gd( a )
    { }
  protected:
    kaapi_access_t _gd;
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_w {
  public:
    typedef T value_type;

    operator kaapi_access_t&() { return _gd; }
    Shared_w( const kaapi_access_t& a )
     : _gd( a )
    { }

    void write( const value_type& new_value )
    { 
      T* data = (T*)_gd.data;
      *data = new_value;
    }

    void write(value_type* new_value) 
    { 
      T* data = (T*)_gd.data;
      *data = *new_value;
    }
  protected:
    kaapi_access_t _gd;
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_rpwp {
  public:
    typedef T value_type;

    operator kaapi_access_t&() { return _gd; }
    Shared_rpwp( const kaapi_access_t& a )
     : _gd( a )
    { }
  protected:
    kaapi_access_t _gd;
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_rw {
  public:
    typedef T value_type;

    operator kaapi_access_t&() { return _gd; }
    Shared_rw( const kaapi_access_t& a )
     : _gd( a )
    { }

    value_type& access() const
    { return *(T*)_gd.data; }
  protected:
    kaapi_access_t _gd;
  };


  // --------------------------------------------------------------------
  template<class T>
  struct DefaultAdd {
    void operator()( T& result, const T& value ) const
    {
      result += value;
    }
  };
  
  template<class T, class OpCumul = DefaultAdd<T> >
  class Shared_cwp {
  public:    
    typedef T value_type;

    Shared_cwp(const kaapi_access_t& a )
     : _gd( a )
    { }
  protected:
    kaapi_access_t _gd;
  };


  template<class T, class OpCumul = DefaultAdd<T> >
  class Shared_cw {
  public:
    typedef T value_type;

    Shared_cw( const kaapi_access_t& a )
     : _gd( a )
    { }

    void cumul( const value_type& value )
    {
      static OpCumul op;
      op( *(T*)_gd.data, value );
    }

    void cumul( value_type* value )
    { 
      op( *(T*)_gd.data, *value );
      delete value;
    }
  protected:
    kaapi_access_t _gd;
  };



  // -------------------------------------------------------------------- VECTOR of Shared
//\TODO


  // -------------------------------------------------------------------- VECTOR of Shared


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
    {  return clo; }
  };
  extern SetLocalAttribut SetLocal;

#if 0
  /* DEPRECATED??? to nothing... not yet distributed implementation */
  class AttributSetCost {
    float _cost;
  public:
    AttributSetCost( float c ) : _cost(c) {}
    kaapi_task_t* operator()( kaapi_stack_t*, kaapi_task_t* clo) const
    { return clo; }
  };
  inline AttributSetCost SetCost( float c )
  { return AttributSetCost(c); }
#endif

  /* to nothing... not yet distributed implementation */
  class AttributSetSite {
    int _site;
  public:
    AttributSetSite( int s ) : _site(s) {}
    kaapi_task_t* operator()( kaapi_stack_t*, kaapi_task_t* clo) const
    { return clo; }
  };

  inline AttributSetSite SetSite( int s )
  { return AttributSetSite(s); }
  

  class SetStaticSchedAttribut {
    int _npart;
    int _niter;
  public:
    SetStaticSchedAttribut( int n, int m  ) 
     : _npart(n), _niter(m) {}
    template<class A1_CLO>
    kaapi_task_t* operator()( kaapi_stack_t*, A1_CLO*& clo) const
    { 
      return clo; 
    }
  };
  inline SetStaticSchedAttribut SetStaticSched(int npart, int iter = 1 )
  { return SetStaticSchedAttribut(npart, iter); }

#if 0 //\TODO dans un monde ideal, il faudrait ca
#include "atha_spacecollection.h"
#endif

  template <class T>
  class WrapperFormat {
  public:
    static const kaapi_format_t* format;
  };

  template <class T>
  const kaapi_format_t* WrapperFormat<T>::format = 0;

  template <>
  const kaapi_format_t* WrapperFormat<kaapi_int8_t>::format;
  template <>
  const kaapi_format_t* WrapperFormat<kaapi_int16_t>::format;
  template <>
  const kaapi_format_t* WrapperFormat<kaapi_int32_t>::format;
  template <>
  const kaapi_format_t* WrapperFormat<kaapi_int64_t>::format;
  template <>
  const kaapi_format_t* WrapperFormat<kaapi_uint8_t>::format;
  template <>
  const kaapi_format_t* WrapperFormat<kaapi_uint16_t>::format;
  template <>
  const kaapi_format_t* WrapperFormat<kaapi_uint32_t>::format;
  template <>
  const kaapi_format_t* WrapperFormat<kaapi_uint64_t>::format;
  template <>
  const kaapi_format_t* WrapperFormat<float>::format;
  template <>
  const kaapi_format_t* WrapperFormat<double>::format;

  // --------------------------------------------------------------------
  /* typenames for access mode */
  struct ACCESS_MODE_V {};
  struct ACCESS_MODE_R {};
  struct ACCESS_MODE_W {};
  struct ACCESS_MODE_RW {};
  struct ACCESS_MODE_CW {};
  struct ACCESS_MODE_RP {};
  struct ACCESS_MODE_WP {};
  struct ACCESS_MODE_RPWP {};
  struct ACCESS_MODE_CWP {};

  template<class T>
  struct Trait_ParamClosure {
    typedef T type_inclosure;
    typedef T value_type;
    enum { isshared = false };
    static const kaapi_format_t* format;
    typedef ACCESS_MODE_V mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_V };
    template<class E>
    static void link( type_inclosure& f, const E& e) { f = e; }
  };
  template<class T>
  const kaapi_format_t* Trait_ParamClosure<T>::format = WrapperFormat<T>::format;

  template<class T>
  struct Trait_ParamClosure<const T&> {
    typedef T type_inclosure;
    typedef T value_type;
    enum { isshared = false };
    static const kaapi_format_t* format;
    typedef ACCESS_MODE_V mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_V };
    template<class E>
    static void link( type_inclosure& f, const E& e) { f = e; }
  };
  template<class T>
  const kaapi_format_t* Trait_ParamClosure<const T&>::format = WrapperFormat<T>::format;

  template<class T>
  struct Trait_ParamClosure<Shared<T> > {
    typedef kaapi_access_t type_inclosure;
    enum { isshared = true };
    static const kaapi_format_t* format;
    typedef ACCESS_MODE_RPWP mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_RW| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };
  template<class T>
  const kaapi_format_t* Trait_ParamClosure<Shared<T> >::format = WrapperFormat<T>::format;

  template<class T>
  struct Trait_ParamClosure<Shared_rw<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_rw<T>   value_type;
    enum { isshared = true };
    static const kaapi_format_t* format;
    typedef ACCESS_MODE_RW mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_RW };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };
  template<class T>
  const kaapi_format_t* Trait_ParamClosure<Shared_rw<T> >::format = WrapperFormat<T>::format;

  template<class T>
  struct Trait_ParamClosure<Shared_r<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_r<T>    value_type;
    enum { isshared = true };
    static const kaapi_format_t* format;
    typedef ACCESS_MODE_R mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_R };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };
  template<class T>
  const kaapi_format_t* Trait_ParamClosure<Shared_r<T> >::format = WrapperFormat<T>::format;

  template<class T>
  struct Trait_ParamClosure<Shared_w<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_w<T>    value_type;
    enum { isshared = true };
    static const kaapi_format_t* format;
    typedef ACCESS_MODE_W mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_W };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };
  template<class T>
  const kaapi_format_t* Trait_ParamClosure<Shared_w<T> >::format = WrapperFormat<T>::format;

  template<class T, class F>
  struct Trait_ParamClosure<Shared_cw<T, F> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_cw<T,F> value_type;
    enum { isshared = true };
    static const kaapi_format_t* format;
    typedef ACCESS_MODE_CW mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_CW };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };
  template<class T, class F>
  const kaapi_format_t* Trait_ParamClosure<Shared_cw<T,F> >::format = WrapperFormat<T>::format;

  template<class T>
  struct Trait_ParamClosure<Shared_rpwp<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_rpwp<T> value_type;
    enum { isshared = true };
    static const kaapi_format_t* format;
    typedef ACCESS_MODE_RPWP mode;
    enum { modepostponed = true };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_RW| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };
  template<class T>
  const kaapi_format_t* Trait_ParamClosure<Shared_rpwp<T> >::format = WrapperFormat<T>::format;

  template<class T>
  struct Trait_ParamClosure<Shared_rp<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_rp<T>   value_type;
    enum { isshared = true };
    static const kaapi_format_t* format;
    typedef ACCESS_MODE_RP mode;
    enum { modepostponed = true };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_R| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };
  template<class T>
  const kaapi_format_t* Trait_ParamClosure<Shared_rp<T> >::format = WrapperFormat<T>::format;

  template<class T>
  struct Trait_ParamClosure<Shared_wp<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_wp<T>   value_type;
    enum { isshared = true };
    static const kaapi_format_t* format;
    typedef ACCESS_MODE_WP mode;
    enum { modepostponed = true };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_W| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };
  template<class T>
  const kaapi_format_t* Trait_ParamClosure<Shared_wp<T> >::format = WrapperFormat<T>::format;

  template<class T, class F>
  struct Trait_ParamClosure<Shared_cwp<T, F> > {
    typedef kaapi_access_t  type_inclosure;
    typedef Shared_cwp<T,F> value_type;
    enum { isshared = true };
    static const kaapi_format_t* format;
    typedef ACCESS_MODE_CWP mode;
    enum { modepostponed = true };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_CW| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };
  template<class T, class F>
  const kaapi_format_t* Trait_ParamClosure<Shared_cwp<T,F> >::format = WrapperFormat<T>::format;


  // --------------------------------------------------------------------
  /* for better understand error message */
  template<int i>
  struct ARG {};

  /* for better understand error message */
  template<class TASK>
  struct FOR_TASKNAME {};
  
  template<class ME, class MF, class PARAM, class TASK>
  struct PassingRule {
//    static void IS_COMPATIBLE();
  };
  template<class PARAM, class TASK>
  struct PassingRule<ACCESS_MODE_V, ACCESS_MODE_V, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<ACCESS_MODE_CW, ACCESS_MODE_CW, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<ACCESS_MODE_R, ACCESS_MODE_R, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<ACCESS_MODE_RPWP, ACCESS_MODE_RW, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<ACCESS_MODE_RPWP, ACCESS_MODE_W, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<ACCESS_MODE_RPWP, ACCESS_MODE_CW, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<ACCESS_MODE_RPWP, ACCESS_MODE_R, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<ACCESS_MODE_RP, ACCESS_MODE_R, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<ACCESS_MODE_WP, ACCESS_MODE_W, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK> /* this rule is only valid for terminal fork... */
  struct PassingRule<ACCESS_MODE_W, ACCESS_MODE_W, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<ACCESS_MODE_CWP, ACCESS_MODE_CW, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };


  // --------------------------------------------------------------------
  template<class TASK>
  struct KaapiTask0 {
    static void body( kaapi_task_t* task, kaapi_stack_t* stack )
    { 
      static TASK dummy;
      dummy();
    }
  };

#include "athapascan_closure.h"

  // --------------------------------------------------------------------
  /* New API: thread.Fork<TASK>([ATTR])( args )
     Fork<TASK>([ATTR])(args) with be implemented on top of 
     System::get_current_thread()->Fork<TASK>([ATTR])( args ).
  */
  class Thread {
  public:
    template<class TASK, class Attr>
    class Forker {
    public:
      Forker( kaapi_stack_t* s, const Attr& a ) : _stack(s), _attr(a) {}

      /**
      **/      
      void operator()()
      { 
        kaapi_task_t* clo = kaapi_stack_toptask( _stack);
        kaapi_task_initdfg( _stack, clo, KaapiTask0<TASK>::body, 0 );
        _attr(_stack, clo);
        kaapi_stack_pushtask( _stack);    
      }

#include "athapascan_fork.h"

    protected:
      kaapi_stack_t* _stack;
      const Attr&    _attr;
    };
        
    template<class TASK>
    Forker<TASK, DefaultAttribut> Fork() { return Forker<TASK, DefaultAttribut>(&_stack, DefaultAttribut()); }

    template<class TASK, class Attr>
    Forker<TASK, Attr> Fork(const Attr& a) { return Forker<TASK, Attr>(&_stack, a); }

  protected:
    kaapi_stack_t _stack;
  };

  
  
  // --------------------------------------------------------------------
  /** Top level Fork */
  template<class TASK>
  Thread::Forker<TASK, DefaultAttribut> Fork() { return Thread::Forker<TASK, DefaultAttribut>(kaapi_self_stack(), DefaultAttribut()); }

  template<class TASK, class Attr>
  Thread::Forker<TASK, Attr> Fork(const Attr& a) { return Thread::Forker<TASK, Attr>(kaapi_self_stack(), a); }



  // --------------------------------------------------------------------
  /** Wait execution of all forked tasks of the running task */
  extern void Sync();



  // --------------------------------------------------------------------
  /* Main task */
  template<class TASK>
  struct MainTask {
    int    argc;
    char** argv;
    static void body( kaapi_task_t* task, kaapi_stack_t* stack )
    {
      MainTask<TASK>* args = kaapi_task_getargst( task, MainTask<TASK>);
      TASK()( args->argc, args->argv );
    }
  };
  
  template<class TASK>
  struct ForkerMain
  {
    ForkerMain() 
    { }

    void operator()( int argc, char** argv)
    {
      kaapi_stack_t* stack = kaapi_self_stack();
      kaapi_task_t* clo = kaapi_stack_toptask( stack);
      kaapi_task_initdfg( stack, clo, &MainTask<TASK>::body, kaapi_stack_pushdata(stack, sizeof(MainTask<TASK>)) );
      MainTask<TASK>* arg = kaapi_task_getargst( clo, MainTask<TASK>);
      arg->argc = argc;
      arg->argv = argv;
      kaapi_stack_pushtask( stack);    
    }
  };

  template<class TASK>
  ForkerMain<TASK> ForkMain()
  { 
    return ForkerMain<TASK>();
  }
    

} // namespace a1


// ---------------------------------------------------------------------------------
/** Compatibility with old C++ Kaapi
*/
#include "atha_timer.h"

namespace  atha {
  extern std::ostream& logfile();
}

namespace Util {
  using atha::WallTimer;
  using atha::CpuTimer;
  using atha::SysTimer;
  using atha::HighResTimer;
  using atha::logfile;
  
  /* old names */
  typedef kaapi_uint8_t  ka_uint8_t;
  typedef kaapi_uint16_t ka_uint16_t;
  typedef kaapi_uint32_t ka_uint32_t;
  typedef kaapi_uint64_t ka_uint64_t;

  typedef kaapi_int8_t   ka_int8_t;
  typedef kaapi_int16_t  ka_int16_t;
  typedef kaapi_int32_t  ka_int32_t;
  typedef kaapi_int64_t  ka_int64_t;
  
  using a1::WrapperFormat;
};

// ---------------------------------------------------------------------------------
/* empty stream function: not in this version */
namespace a1 {
  struct IOStream_base {
    enum Mode { 
      IA,   /* immediate access of value */ 
      DA,   /* differed access of value */
      DAC   /* differed access of possibly cyclic pointer value */
    };
  };
  struct OStream: public IOStream_base {
    /** */
    void write( const Format* const f, Mode m, const void* data, size_t count ) {}
  };
  struct IStream: public IOStream_base {
    /** */
    void read( const Format* const f, Mode m, void* const data, size_t count ) {}
  };
  
  inline OStream& operator<< (OStream& m, const bool v )  { return m; }
  inline OStream& operator<< (OStream& m, const char v )  { return m; }
  inline OStream& operator<< (OStream& m, const signed char v )  { return m; }
  inline OStream& operator<< (OStream& m, const unsigned char v )  { return m; }
  inline OStream& operator<< (OStream& m, const short v )  { return m; }
  inline OStream& operator<< (OStream& m, const unsigned short v )  { return m; }
  inline OStream& operator<< (OStream& m, const int v )  { return m; }
  inline OStream& operator<< (OStream& m, const unsigned int v )  { return m; }
  inline OStream& operator<< (OStream& m, const long v )  { return m; }
  inline OStream& operator<< (OStream& m, const unsigned long v )  { return m; }
  inline OStream& operator<< (OStream& m, const long long v )  { return m; }
  inline OStream& operator<< (OStream& m, const unsigned long long v )  { return m; }
  inline OStream& operator<< (OStream& m, const float v )  { return m; }
  inline OStream& operator<< (OStream& m, const double v )  { return m; }
  inline OStream& operator<< (OStream& m, const long double v )  { return m; }
  inline OStream& operator<< (OStream& m, const std::string& v )  { return m; }
//TODO  inline OStream& operator<< (OStream& o, const Pointer& s)  { return m; }
  template<class T>
  inline OStream& operator<< (OStream& m, const std::vector<T>& v )  { return m; }
  template<class Fst, class Snd>
  inline  OStream& operator<< (OStream& m, const std::pair<Fst,Snd>& p )  { return m; }


  /* -----------------------------------
  */
  #ifdef MACOSX_EDITOR
  #pragma mark ----- Input
  #endif
  inline IStream& operator>> (IStream& m, bool& v )  { return m; }
  inline IStream& operator>> (IStream& m, char& v )  { return m; }
  inline IStream& operator>> (IStream& m, signed char& v )  { return m; }
  inline IStream& operator>> (IStream& m, unsigned char& v )  { return m; }
  inline IStream& operator>> (IStream& m, short& v )  { return m; }
  inline IStream& operator>> (IStream& m, unsigned short& v )  { return m; }
  inline IStream& operator>> (IStream& m, int& v )  { return m; }
  inline IStream& operator>> (IStream& m, unsigned int& v )  { return m; }
  inline IStream& operator>> (IStream& m, long& v )  { return m; }
  inline IStream& operator>> (IStream& m, unsigned long& v )  { return m; }
  inline IStream& operator>> (IStream& m, long long& v )  { return m; }
  inline IStream& operator>> (IStream& m, unsigned long long& v )  { return m; }
  inline IStream& operator>> (IStream& m, float& v )  { return m; }
  inline IStream& operator>> (IStream& m, double& v )  { return m; }
  inline IStream& operator>> (IStream& m, long double& v )  { return m; }
  inline IStream& operator>> (IStream& m, std::string& v )  { return m; }
//TODO  inline IStream& operator>> (IStream& i, Pointer& s)  { return m; }
  template<class T> inline IStream& operator>> (IStream& m, std::vector<T>& v )  { return m; }
  template<class Fst, class Snd> inline  IStream& operator>> (IStream& m, std::pair<Fst,Snd>& p )  { return m; }

//  template<class T> inline void WrapperFormat<T>::write( OStream& s, const void* val, size_t count ) const{ }
//  template<class T> inline void WrapperFormat<T>::read ( IStream& s, void* val, size_t count ) const {}
};

// ---------------------------------------------------------------------------------
#if 0 // \TODO
namespace a1 {
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
        // \TODO: only execution in one frame and sub frame but not on top frame
        kaapi_stack_restore_frame( &_thread->_stack, &_frame );
      }
  };
}
#endif

#ifndef ATHAPASCAN_NOT_IN_NAMESPACE
using namespace a1;
#endif

#endif

