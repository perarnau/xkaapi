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
  class Format;
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

    ~Shared ( ) 
    {
#if 0
      Thread* thread = System::get_current_thread(); 
      destroy( thread );
#endif
    }
    
    Shared ( value_type* data = 0 ) 
    {
#if 0
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
#endif
    }

    Shared ( const SetStack& toto, value_type* data = 0) 
    {
#if 0
      Thread* thread = System::get_current_thread(); 
      DFG::GlobalData::Attribut attr;
      attr.clear();
      attr.set_instack();
      if(!data) data = new (thread->allocate(sizeof(value_type))) value_type;
      initialize( thread, data, &Util::WrapperFormat<value_type>::theformat, attr);
#endif
    }

    Shared ( const SetHeap& toto, value_type* data = 0) 
    {
#if 0
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
#endif
    }

    Shared(const value_type& value )
    {
#if 0
      if (sizeof(value_type) <= STACK_ALLOC_THRESHOLD) {
        Thread* thread = System::get_current_thread(); 
        _gd.data  = new (SharedAllocator, thread) T(value)
        _gd._attr = 0;
      } else {
        _gd.data  = new (SharedAllocator) T(value)
        _gd._attr = 1;
      }
#endif
    }

    Shared(const SetStack& toto, const T& value )
    {
#if 0
      Thread* thread = System::get_current_thread(); 
      _gd.data  = new (SharedAllocator, thread) T(value)
      _gd._attr = 0;
#endif
    }

    Shared(const SetHeap& toto, const T& value )
    {
#if 0
      _gd.data  = new (SharedAllocator) T(value)
      _gd._attr = 1;
#endif
    }

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
    int            _attr; /* 0: in stack, 1: in heap */
  };
  

  // --------------------------------------------------------------------
  template<class T>
  class Shared_rp {
  public:
    typedef T value_type;

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
  typedef kaapi_task_t Closure;
  
  class DefaultAttribut {
  public:
    Closure* operator()( Thread*, Closure* clo)
    { return clo; }
  };
  extern DefaultAttribut SetDefault;
  
  /* */
  class UnStealableAttribut {
  public:
    Closure* operator()( Thread*, Closure* clo)
    { clo->flag |= KAAPI_TASK_STICKY; return clo; }
  };
  inline UnStealableAttribut SetUnStealable()
  { return UnStealableAttribut(); }

  /* like default attribut: not yet distributed computation */
  class SetLocalAttribut {
  public:
    Closure* operator()( Thread*, Closure* clo)
    {  return clo; }
  };
  extern SetLocalAttribut SetLocal;

#if 0
  /* DEPRECATED??? to nothing... not yet distributed implementation */
  class AttributSetCost {
    float _cost;
  public:
    AttributSetCost( float c ) : _cost(c) {}
    Closure* operator()( Thread*, Closure* clo)
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
    Closure* operator()( Thread*, Closure* clo)
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
    Closure* operator()( Thread* t, A1_CLO*& clo)
    { 
      return clo; 
    }
  };
  inline SetStaticSchedAttribut SetStaticSched(int npart, int iter = 1 )
  { return SetStaticSchedAttribut(npart, iter); }

#if 0 //\TODO dans un monde ideal, il faudrait ca
#include "atha_spacecollection.h"
#endif

  // --------------------------------------------------------------------
  template<class T>
  struct Trait_ParamClosure {
    typedef T type_inclosure;
    enum { isshared = false };
    enum { mode = KAAPI_ACCESS_MODE_V};
    enum { modepostponed = false };
  };

  template<class T>
  struct Trait_ParamClosure<const T&> {
    typedef T type_inclosure;
    enum { isshared = false };
    enum { mode = KAAPI_ACCESS_MODE_V};
    enum { modepostponed = false };
  };

  template<class T>
  struct Trait_ParamClosure<Shared<T> > {
    typedef kaapi_access_t type_inclosure;
    enum { isshared = true };
    enum { mode = KAAPI_ACCESS_MODE_RW | KAAPI_ACCESS_MODE_P };
    enum { modepostponed = false };
  };

  template<class T>
  struct Trait_ParamClosure<Shared_rw<T> > {
    typedef kaapi_access_t type_inclosure;
    enum { isshared = true };
    enum { mode = KAAPI_ACCESS_MODE_RW };
    enum { modepostponed = false };
  };

  template<class T>
  struct Trait_ParamClosure<Shared_r<T> > {
    typedef kaapi_access_t type_inclosure;
    enum { isshared = true };
    enum { mode = KAAPI_ACCESS_MODE_R };
    enum { modepostponed = false };
  };

  template<class T>
  struct Trait_ParamClosure<Shared_w<T> > {
    typedef kaapi_access_t type_inclosure;
    enum { isshared = true };
    enum { mode = KAAPI_ACCESS_MODE_W };
    enum { modepostponed = false };
  };

  template<class T>
  struct Trait_ParamClosure<Shared_cw<T> > {
    typedef kaapi_access_t type_inclosure;
    enum { isshared = true };
    enum { mode = KAAPI_ACCESS_MODE_CW };
    enum { modepostponed = false };
  };

  template<class T>
  struct Trait_ParamClosure<Shared_rpwp<T> > {
    typedef kaapi_access_t type_inclosure;
    enum { isshared = true };
    enum { mode = KAAPI_ACCESS_MODE_RW | KAAPI_ACCESS_MODE_P };
    enum { modepostponed = true };
  };

  template<class T>
  struct Trait_ParamClosure<Shared_rp<T> > {
    enum { isshared = true };
    enum { mode = KAAPI_ACCESS_MODE_R | KAAPI_ACCESS_MODE_P };
    enum { modepostponed = true };
  };

  template<class T>
  struct Trait_ParamClosure<Shared_wp<T> > {
    typedef kaapi_access_t type_inclosure;
    enum { isshared = true };
    enum { mode = KAAPI_ACCESS_MODE_W | KAAPI_ACCESS_MODE_P };
    enum { modepostponed = true };
  };

  template<class T, class F>
  struct Trait_ParamClosure<Shared_cwp<T, F> > {
    typedef kaapi_access_t type_inclosure;
    enum { isshared = true };
    enum { mode = KAAPI_ACCESS_MODE_CW | KAAPI_ACCESS_MODE_P };
    enum { modepostponed = true };
  };


  // --------------------------------------------------------------------
  /* for better understand error message */
  template<int i>
  struct ARG {};

  /* for better understand error message */
  template<class TASK>
  struct FOR_TASKNAME {};

  template<char ME, char MF, class PARAM, class TASK>
  struct PassingRule {
    static void IS_COMPATIBLE();
  };
  template<class PARAM, class TASK>
  struct PassingRule<KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<KAAPI_ACCESS_MODE_CW, KAAPI_ACCESS_MODE_CW, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<KAAPI_ACCESS_MODE_R, KAAPI_ACCESS_MODE_R, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<KAAPI_ACCESS_MODE_RW|KAAPI_ACCESS_MODE_P, KAAPI_ACCESS_MODE_RW, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<KAAPI_ACCESS_MODE_RW|KAAPI_ACCESS_MODE_P, KAAPI_ACCESS_MODE_W, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<KAAPI_ACCESS_MODE_RW|KAAPI_ACCESS_MODE_P, KAAPI_ACCESS_MODE_CW, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<KAAPI_ACCESS_MODE_RW|KAAPI_ACCESS_MODE_P, KAAPI_ACCESS_MODE_R, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<KAAPI_ACCESS_MODE_R|KAAPI_ACCESS_MODE_P, KAAPI_ACCESS_MODE_R, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<KAAPI_ACCESS_MODE_W|KAAPI_ACCESS_MODE_P, KAAPI_ACCESS_MODE_W, PARAM, TASK> {
    static void IS_COMPATIBLE(){}
  };
  template<class PARAM, class TASK>
  struct PassingRule<KAAPI_ACCESS_MODE_CW|KAAPI_ACCESS_MODE_P, KAAPI_ACCESS_MODE_CW, PARAM, TASK> {
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

  // --------------------------------------------------------------------
  template<class TASK, class F1>
  struct KaapiTask1 {
    F1 f1;
    typedef KaapiTask1<TASK, F1> Self_t;
    static void body( kaapi_task_t* task, kaapi_stack_t* stack )
    { 
      static TASK dummy;
      Self_t* args = kaapi_task_getargst( task, Self_t);
      dummy(args->f1);
    }
  };

  // --------------------------------------------------------------------
  template<class TASK, class F1, class F2>
  struct KaapiTask2 {
    F1 f1;
    F2 f2;
    typedef KaapiTask2<TASK, F1, F2> Self_t;
    static void body( kaapi_task_t* task, kaapi_stack_t* stack )
    { 
      static TASK dummy;
      Self_t* args = kaapi_task_getargst( task, Self_t);
      dummy(args->f1, args->f2);
    }
  };

  // --------------------------------------------------------------------
  template<class TASK, class F1, class F2, class F3>
  struct KaapiTask3 {
    F1 f1;
    F2 f2;
    F3 f3;
    typedef KaapiTask3<TASK, F1, F2, F3> Self_t;
    static void body( kaapi_task_t* task, kaapi_stack_t* stack )
    { 
      static TASK dummy;
      Self_t* args = kaapi_task_getargst( task, Self_t);
      dummy(args->f1, args->f2, args->f3);
    }
  };

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

      template<class E1, class F1>
      Closure* PushArg( void (TASK::*)(F1), const E1& e1 )
      {
        typedef typename Trait_ParamClosure<F1>::type_inclosure F1_CLO;
        typedef KaapiTask1<TASK, F1_CLO> KaapiClosure;

        PassingRule<Trait_ParamClosure<E1>::mode,Trait_ParamClosure<F1>::mode, ARG<1>, FOR_TASKNAME<TASK> >::IS_COMPATIBLE();

        kaapi_task_t* clo = kaapi_stack_toptask( _stack);
        kaapi_task_initdfg( _stack, clo, KaapiClosure::body, kaapi_stack_pushdata(_stack, sizeof(KaapiClosure)) );
        KaapiClosure* arg = kaapi_task_getargst( clo, KaapiClosure);
        arg->f1 = e1;
        return clo;
      }

      template<class E1, class F1, class E2, class F2>
      Closure* PushArg( void (TASK::*)(F1, F2), const E1& e1, const E2& e2 )
      {
        typedef typename Trait_ParamClosure<F1>::type_inclosure F1_CLO;
        typedef typename Trait_ParamClosure<F2>::type_inclosure F2_CLO;
        typedef KaapiTask2<TASK, F1_CLO, F2_CLO> KaapiClosure;

        PassingRule< Trait_ParamClosure<E1>::mode,Trait_ParamClosure<F1>::mode, ARG<1>, FOR_TASKNAME<TASK> >::IS_COMPATIBLE();
        PassingRule< Trait_ParamClosure<E2>::mode,Trait_ParamClosure<F2>::mode, ARG<2>, FOR_TASKNAME<TASK> >::IS_COMPATIBLE();

        kaapi_task_t* clo = kaapi_stack_toptask( _stack);
        kaapi_task_initdfg( _stack, clo, KaapiClosure::body, kaapi_stack_pushdata(_stack, sizeof(KaapiClosure)) );
        KaapiClosure* arg = kaapi_task_getargst( clo, KaapiClosure);
        arg->f1 = e1;
        arg->f2 = e2;
        return clo;
      }

      template<class E1, class F1, class E2, class F2, class E3, class F3>
      Closure* PushArg( void (TASK::*)(F1, F2, F3), const E1& e1, const E2& e2, const E3& e3 )
      {
        typedef typename Trait_ParamClosure<F1>::type_inclosure F1_CLO;
        typedef typename Trait_ParamClosure<F2>::type_inclosure F2_CLO;
        typedef typename Trait_ParamClosure<F3>::type_inclosure F3_CLO;
        typedef KaapiTask3<TASK, F1_CLO, F2_CLO, F3_CLO> KaapiClosure;

        PassingRule< Trait_ParamClosure<E1>::mode,Trait_ParamClosure<F1>::mode, ARG<1>, FOR_TASKNAME<TASK> >::IS_COMPATIBLE();
        PassingRule< Trait_ParamClosure<E2>::mode,Trait_ParamClosure<F2>::mode, ARG<2>, FOR_TASKNAME<TASK> >::IS_COMPATIBLE();
        PassingRule< Trait_ParamClosure<E3>::mode,Trait_ParamClosure<F3>::mode, ARG<3>, FOR_TASKNAME<TASK> >::IS_COMPATIBLE();

        kaapi_task_t* clo = kaapi_stack_toptask( _stack);
        kaapi_task_initdfg( _stack, clo, KaapiClosure::body, kaapi_stack_pushdata(_stack, sizeof(KaapiClosure)) );
        KaapiClosure* arg = kaapi_task_getargst( clo, KaapiClosure);
        arg->f1 = e1;
        arg->f2 = e2;
        arg->f3 = e3;
        return clo;
      }

      /**
      **/
      
      void operator()()
      { 
        kaapi_task_t* clo = kaapi_stack_toptask( _stack);
        kaapi_task_initdfg( _stack, clo, KaapiTask0<TASK>::body, 0 );
        _attr(_stack, clo);
        kaapi_stack_pushtask( _stack);    
      }

      template<class E1> 
      void operator()(const E1& e1)
      { 
        kaapi_task_t* clo = PushArg( &TASK::operator(), e1 );
        _attr(_stack, clo);
        kaapi_stack_pushtask( _stack);    
      }

      template<class E1, class E2> 
      void operator()(const E1& e1, const E2& e2)
      { 
        kaapi_task_t* clo = PushArg( &TASK::operator(), e1, e2 );
        _attr(_stack, clo);
        kaapi_stack_pushtask( _stack);    
      }

      template<class E1, class E2, class E3> 
      void operator()(const E1& e1, const E2& e2, const E3& e3)
      { 
        kaapi_task_t* clo = PushArg( &TASK::operator(), e1, e2, e3 );
        _attr(_stack, clo);
        kaapi_stack_pushtask( _stack);    
      }
    protected:
      kaapi_stack_t* _stack;
      Attr*          _attr;
    };
        
    template<class TASK>
    Forker<TASK, DefaultAttribut> Fork() { return Forker<TASK, DefaultAttribut>(&_stack, DefaultAttribut()); }

    template<class TASK, class Attr>
    Forker<TASK, Attr> Fork(const Attr& a) { return Forker<TASK, Attr>(&_stack, a); }

#if 0   
 template<class TASK>
    struct ForkerMain : protected Forker<MainTask<TASK>,DefaultAttribut> 
    {
      ForkerMain() 
       : Forker<MainTask<TASK>,DefaultAttribut>(0,SetDefault)
      { }

      void operator()( int argc, char** argv)
      {
        TASK()( argc, argv );
      }
    };

    template<class TASK>
    ForkerMain<TASK> ForkMain()
    { 
      return ForkerMain<TASK>();
    }
#endif
    
  protected:
    kaapi_stack_t _stack;
  };

  template<class TASK>
  class MainTask {};
  
  /** Wait execution of all forked tasks of the running task */
  extern void Sync();

} // namespace a1


#if 0 // \TODO: a reprendre correctement
/* specialize the resize of vector of shared */
namespace a1 {
template<class T, class Alloc>
inline void resize_vector(std::vector<a1::Shared<T>,Alloc>& v, typename std::vector<a1::Shared<T>,Alloc>::size_type __new_size)
{
  typename std::vector<a1::Shared<T>,Alloc>::size_type sz = v.size();
  v.resize(__new_size);
  for (typename std::vector<a1::Shared<T>,Alloc>::size_type i=sz; i<__new_size; ++i)
    v[i] = Shared<T>();
}
}
#endif


// ---------------------------------------------------------------------------------
#if 0
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
