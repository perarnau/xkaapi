/* KAAPI public interface */
/*
** athapascan-1.h
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
#ifndef _ATHAPASCAN_1_H_H
#define _ATHAPASCAN_1_H_H

#include "kaapi++.h"

namespace a1 {

  /* take a constant... should be adjusted */
  enum { STACK_ALLOC_THRESHOLD = ka::STACK_ALLOC_THRESHOLD };  

  using namespace ka::FormatDef;

  using ka::Format;

  using ka::FormatUpdateFnc;

  class IStream; /* fwd decl */
  class OStream; /* fwd decl */
  using ka::ODotStream;

  using ka::Exception;
  using ka::RuntimeError;
  using ka::InvalidArgumentError;
  using ka::RestartException;
  using ka::ServerException;
  using ka::NoFound;
  using ka::BadAlloc;
  using ka::IOError;
  using ka::ComFailure;
  using ka::BadURL;


  using ka::System;
  using ka::Community;

  using ka::SetStack;
  using ka::SetHeap;
  using ka::SetStickyC;

  // --------------------------------------------------------------------
  extern SetStack SetInStack;

  // --------------------------------------------------------------------
  extern SetHeap SetInHeap;

  // --------------------------------------------------------------------
  extern SetStickyC SetSticky;

  class Thread;
  
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
      Thread* thread = (Thread*)System::get_current_thread(); 
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
    { 
      kaapi_task_setflags( clo, KAAPI_TASK_STICKY );
      return clo; 
    }
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

  using ka::WrapperFormat;
  using ka::WrapperFormatUpdateFnc;

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
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_format(); }
    typedef ACCESS_MODE_V mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_V };
    template<class E>
    static void link( type_inclosure& f, const E& e) { f = e; }
  };

  template<class T>
  struct Trait_ParamClosure<const T&> {
    typedef T type_inclosure;
    typedef T value_type;
    enum { isshared = false };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_format(); }
    typedef ACCESS_MODE_V mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_V };
    template<class E>
    static void link( type_inclosure& f, const E& e) { f = e; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared<T> > {
    typedef kaapi_access_t type_inclosure;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_format(); }
    typedef ACCESS_MODE_RPWP mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_RW| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared_rw<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_rw<T>   value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_format(); }
    typedef ACCESS_MODE_RW mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_RW };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared_r<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_r<T>    value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_format(); }
    typedef ACCESS_MODE_R mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_R };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared_w<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_w<T>    value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_format(); }
    typedef ACCESS_MODE_W mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_W };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };

  template<class T, class F>
  struct Trait_ParamClosure<Shared_cw<T, F> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_cw<T,F> value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_format(); }
    typedef ACCESS_MODE_CW mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_CW };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared_rpwp<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_rpwp<T> value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_format(); }
    typedef ACCESS_MODE_RPWP mode;
    enum { modepostponed = true };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_RW| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared_rp<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_rp<T>   value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_format(); }
    typedef ACCESS_MODE_RP mode;
    enum { modepostponed = true };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_R| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared_wp<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_wp<T>   value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_format(); }
    typedef ACCESS_MODE_WP mode;
    enum { modepostponed = true };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_W| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };

  template<class T, class F>
  struct Trait_ParamClosure<Shared_cwp<T, F> > {
    typedef kaapi_access_t  type_inclosure;
    typedef Shared_cwp<T,F> value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_format(); }
    typedef ACCESS_MODE_CWP mode;
    enum { modepostponed = true };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_CW| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
  };


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
  inline void Sync() { ka::Sync(); }



  // --------------------------------------------------------------------
  /* Main task */
  template<class TASK>
  struct MainTask : ka::MainTask<TASK> {
  };
  
  template<class TASK>
  struct ForkerMain : ka::SpawnerMain<TASK> {
  };

  template<class TASK>
  ForkerMain<TASK> ForkMain()
  { 
    return ForkerMain<TASK>();
  }
    


  // --------------------------------------------------------------------
  /** MonotonicBound
      FncUpdate should have signature int operator()(T& result, const X& value)
  */
  class KaapiMonotonicBoundRep;
  class KaapiMonotonicBound {
  private:
    void*                   _data;
    const kaapi_format_t*   _fmtdata;
    const kaapi_format_t*   _fmtupdate;
    KaapiMonotonicBoundRep* _reserved;
  protected:
    void initialize( const std::string& name, void* value, const kaapi_format_t* format, const kaapi_format_t* fupdate);
    void terminate();
    void acquire();
    void release();
    const void* read() const;
    void update(const void* value, const kaapi_format_t* fmtvaue);
  };
  
  template<class T, class FncUpdate>
  class MonotonicBound : protected KaapiMonotonicBound {
    /* should not be used */
    MonotonicBound<T,FncUpdate>& operator=(const MonotonicBound<T,FncUpdate>& gv) 
    { return *this; }
    MonotonicBound(const MonotonicBound<T,FncUpdate>& a)
    { }

  public:
    typedef T type_val;

    ~MonotonicBound( ) 
    {
    }
    
    MonotonicBound() 
     : KaapiMonotonicBound()
    {
    }

    void initialize( const std::string& name, T* value = 0) 
    {
      if (value ==0) value = new T;
      KaapiMonotonicBound::initialize( name, value, WrapperFormat<T>::format, WrapperFormatUpdateFnc<FncUpdate>::format );
    }

    void terminate()
    {
      KaapiMonotonicBound::terminate();
    }

    void acquire()
    { KaapiMonotonicBound::acquire(); }    

    void release()
    { KaapiMonotonicBound::release(); }
    
    const T& read() const
    {
      const void* data = KaapiMonotonicBound::read();
      return *(const T*)data;
    }
    
    template<class Y>
    void update( const Y& value )
    {
       KaapiMonotonicBound::update( &value, &WrapperFormat<Y>::theformat );
    }
  };


#if 0
  // --------------------------------------------------------------------
  template<class T>
  class SingleAssignment {
  public:
    inline static const Util::Format* get_data_format()
    { return &Util::WrapperFormat<T>::theformat; }

  public:
    typedef T type_val;

    ~SingleAssignment( ) 
    {
    }
    
    SingleAssignment() 
    {
    }

    SingleAssignment(const SingleAssignment<T>& a)
      : _gv(a._gv)
    {
    }

    void initialize( const std::string& name ) 
    {
      _gv.initialize( name );
    }

    void initialize( const std::string& name, T* value) 
    {
      _gv.initialize( name );
      if (value == 0) value = new T;
      _gv.bind( value, get_data_format() );
    }
    
    void terminate()
    {
      _gv.terminate();
    }

    SingleAssignment<T>& operator=(const SingleAssignment<T>& gv) 
    {
      _gv = gv._gv;
      return *this;
    }

    const T& read() const
    {
      const void* data = _gv.read();
      return *(const T*)data;
    }
    
    template<class Y>
    void write( const Y& value )
    {
       _gv.write( new Y(value), &Util::WrapperFormat<Y>::theformat );
    }

    template<class Y>
    void write( Y* value )
    {
       _gv.write( value, &Util::WrapperFormat<Y>::theformat );
    }
  public:
    NetData::SingleAssignment _gv;
  };
#endif

} // namespace a1


// ---------------------------------------------------------------------------------
/** Compatibility with old C++ Kaapi
*/

namespace  atha {
  using ka::logfile;
}

namespace Util {
  using ka::WallTimer;
  using ka::CpuTimer;
  using ka::SysTimer;
  using ka::HighResTimer;
  using ka::logfile;

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
}

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
}

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


/* ========================================================================= */
/* Initialization / destruction functions
 */
namespace a1 {

  extern void _athakaapi_dummy(void*);
  extern void __attribute__ ((constructor)) atha_init(void);
  extern void __attribute__ ((destructor)) atha_fini(void);

#if !defined(KAAPI_COMPILE_SOURCE)
  /** To force reference to atha_kaapi.cpp in order to link against kaapi_init and kaapi_fini
   */
  static void __attribute__((unused)) __athakaapi_dumy_dummy(void)
  {
    _athakaapi_dummy(NULL);
  }
#endif
}

#ifndef ATHAPASCAN_NOT_IN_NAMESPACE
using namespace a1;
#endif

#endif

