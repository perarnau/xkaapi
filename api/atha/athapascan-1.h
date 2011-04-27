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

#include "kaapi++"

namespace a1 {

  /* take a constant... should be adjusted */
  enum { STACK_ALLOC_THRESHOLD = ka::STACK_ALLOC_THRESHOLD };  

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
#if 0
    KAAPI_DECL_EXT_FORMAT(long long, LLong)
    KAAPI_DECL_EXT_FORMAT(unsigned long long, ULLong)
#endif
    KAAPI_DECL_EXT_FORMAT(float, Float)
    KAAPI_DECL_EXT_FORMAT(double, Double)
    KAAPI_DECL_EXT_FORMAT(long double, LDouble)
  }

  using ka::Format;

  using ka::FormatUpdateFnc;

  class IStream; /* fwd decl */
  class OStream; /* fwd decl */
  using ka::ODotStream;

  using ka::System;
  using ka::Community;

  using ka::SetStack;
  using ka::SetHeap;
  using ka::SetStickyC;

  // --------------------------------------------------------------------
  using ka::SetInStack;

  // --------------------------------------------------------------------
  using ka::SetInHeap;

  // --------------------------------------------------------------------
  using ka::SetSticky;

  class Thread;

  // --------------------------------------------------------------------
  template<class T>
  class Shared : private ka::pointer<T> {
      void destroy( );
  public:
    typedef typename ka::pointer<T>::value_type value_type;

    /* for destructor */
    ~Shared ( ) 
    {
      destroy( );
    }
    
#if 0 /* old API: to do ...*/
    Shared ( value_type* data ) 
    {
    }
    Shared ( const SetStack& toto, value_type* data = 0) 
    {
    }
    Shared ( const SetHeap& toto, value_type* data = 0) 
    {
    }
    Shared(const SetStack& toto, const T& value )
    {
    }
    Shared(const SetHeap& toto, const T& value )
    {
    }
#endif

    Shared() : ka::pointer<T>()
    {
      ptr( ka::Alloca<T>() );
    }

    Shared(const value_type& value ) : ka::pointer<T>()
    {
      ptr( new (kaapi_thread_pushdata( kaapi_self_thread(), sizeof(T))) T(value) );
    }

    Shared(const Shared<value_type>& t) : ka::pointer<T>(t)
    {}

    Shared<T>& operator=(const Shared<value_type>& t) 
    {
      ka::pointer<T>::operator=(t);
      t._ptr = 0;
      return *this;
    }
  };
  

  // --------------------------------------------------------------------
  template<class T>
  class Shared_rp : private ka::pointer_rp<T> {
  public:
    typedef typename ka::pointer_rp<T>::value_type value_type;

    Shared_rp( value_type* a )
     : ka::pointer_rp<T>( a )
    { }
    explicit Shared_rp( kaapi_access_t& a )
     : ka::pointer_rp<T>( a )
    { }
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_r : private ka::pointer_r<T> {
  public:
    typedef typename ka::pointer_r<T>::value_type value_type;

    Shared_r( value_type* a )
     : ka::pointer_r<T>( a )
    { }

    explicit Shared_r( kaapi_access_t& a )
     : ka::pointer_r<T>( a )
    { }

    const value_type& read() const 
    { return *this->ptr(); }
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_wp : public ka::pointer_wp<T> {
  public:
    typedef typename ka::pointer_wp<T>::value_type value_type;

    Shared_wp( value_type* a )
     : ka::pointer_wp<T>( a )
    { }
    explicit Shared_wp( kaapi_access_t& a )
     : ka::pointer_wp<T>( a )
    { }
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_w : private ka::pointer_w<T> {
  public:
    typedef typename ka::pointer_w<T>::value_type value_type;

    Shared_w( value_type* a )
     : ka::pointer_w<T>( a )
    { }
    explicit Shared_w( kaapi_access_t& a )
     : ka::pointer_w<T>( a )
    { }

    void write( const value_type& new_value )
    { 
      this->operator*() = new_value;
    }

#if 0 /* old API */
    void write(value_type* new_value) 
    { }
#endif
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_rpwp : private ka::pointer_rpwp<T> {
  public:
    typedef typename ka::pointer_rpwp<T>::value_type value_type;

    Shared_rpwp( value_type* a )
     : ka::pointer_rpwp<T>( a )
    { }
    explicit Shared_rpwp( kaapi_access_t& a )
     : ka::pointer_rpwp<T>( a )
    { }
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_rw : private ka::pointer_rw<T> {
  public:
    typedef typename ka::pointer_rw<T>::value_type value_type;

    Shared_rw( value_type* a )
     : ka::pointer_rw<T>( a )
    { }

    explicit Shared_rw( kaapi_access_t& a )
     : ka::pointer_rw<T>( a )
    { }

    value_type& access()
    { return *this->ptr(); }
    
    void swap(T*& p) 
    { std::swap(p, this->_ptr); }
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
  class Shared_cwp : private ka::pointer_cwp<T> {
  public:    
    typedef typename ka::pointer_cwp<T>::value_type value_type;

    Shared_cwp( value_type* a )
     : ka::pointer_cwp<T>( a )
    { }

    explicit Shared_cwp( kaapi_access_t& a )
     : ka::pointer_cwp<T>( a )
    { }
  };


  template<class T, class OpCumul = DefaultAdd<T> >
  class Shared_cw : private ka::pointer_cw<T> {
  public:
    typedef typename ka::pointer_cw<T>::value_type value_type;

    Shared_cw( value_type* a )
     : ka::pointer_cw<T>( a )
    { }

    explicit Shared_cw( kaapi_access_t& a )
     : ka::pointer_cw<T>( a )
    { }

    void cumul( const value_type& value )
    {
      op( *this, value );
    }

    void cumul( value_type* value )
    { 
      op( *this, *value );
      delete value;
    }
  };


  // --------------------------------------------------------------------
  template<class T>
  struct TaskDelete {
    void operator() ( a1::Shared_rw<T> res )
    {
      T* ptr = 0;
      res.swap(ptr);
      ptr->T::~T();
    }
  };


  // -------------------------------------------------------------------- VECTOR of Shared
//\TODO


  // -------------------------------------------------------------------- VECTOR of Shared


  // --------------------------------------------------------------------  
  using ka::DefaultAttribut;
  using ka::SetDefault;
#if 0 // DEPRECATED   /* */
  using ka::UnStealableAttribut;
  using ka::SetUnStealable;
  using ka::SetLocalAttribut;
  using ka::SetLocal;
#endif
  using ka::AttributSetPartition;
  using ka::SetPartition;
  
#if 0
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
      //kapi_task_setflags( clo, KAAPI_TASK_STICKY );
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
    kaapi_task_t* operator()( kaapi_thread_t*, kaapi_task_t* clo) const
    { return clo; }
  };
  inline AttributSetCost SetCost( float c )
  { return AttributSetCost(c); }
#endif

  /* to nothing... not yet distributed implementation */
  class AttributSetSite : public ka::AttributSetPartition {
  public:
    AttributSetSite( int s ) : ka::AttributSetPartition(s) {}
    kaapi_task_t* operator()( kaapi_thread_t*, kaapi_task_t* clo) const
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
    kaapi_task_t* operator()( kaapi_thread_t*, A1_CLO*& clo) const
    { 
      return clo; 
    }
  };
  inline SetStaticSchedAttribut SetStaticSched(int npart, int iter = 1 )
  { return SetStaticSchedAttribut(npart, iter); }
#endif

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
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_c_format(); }
    typedef ACCESS_MODE_V mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_V };
    template<class E>
    static void link( type_inclosure& f, const E& e) { f = e; }
    static void* address_data( type_inclosure* t ) { return t; }
    static void* address_version( type_inclosure* t ) { return 0; }
  };

  template<class T>
  struct Trait_ParamClosure<const T&> {
    typedef T type_inclosure;
    typedef T value_type;
    enum { isshared = false };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_c_format(); }
    typedef ACCESS_MODE_V mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_V };
    template<class E>
    static void link( type_inclosure& f, const E& e) { f = e; }
    static void* address_data( type_inclosure* t ) { return t; }
    static void* address_version( type_inclosure* t ) { return 0; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared<T> > {
    typedef kaapi_access_t type_inclosure;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_c_format(); }
    typedef ACCESS_MODE_RPWP mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_RW| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
    static void* address_data( type_inclosure* t ) { return &t->data; }
    static void* address_version( type_inclosure* t ) { return &t->version; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared_rw<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_rw<T>   value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_c_format(); }
    typedef ACCESS_MODE_RW mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_RW };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
    static void* address_data( type_inclosure* t ) { return &t->data; }
    static void* address_version( type_inclosure* t ) { return &t->version; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared_r<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_r<T>    value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_c_format(); }
    typedef ACCESS_MODE_R mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_R };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
    static void* address_data( type_inclosure* t ) { return &t->data; }
    static void* address_version( type_inclosure* t ) { return &t->version; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared_w<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_w<T>    value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_c_format(); }
    typedef ACCESS_MODE_W mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_W };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
    static void* address_data( type_inclosure* t ) { return &t->data; }
    static void* address_version( type_inclosure* t ) { return &t->version; }
  };

  template<class T, class F>
  struct Trait_ParamClosure<Shared_cw<T, F> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_cw<T,F> value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_c_format(); }
    typedef ACCESS_MODE_CW mode;
    enum { modepostponed = false };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_CW };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
    static void* address_data( type_inclosure* t ) { return &t->data; }
    static void* address_version( type_inclosure* t ) { return &t->version; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared_rpwp<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_rpwp<T> value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_c_format(); }
    typedef ACCESS_MODE_RPWP mode;
    enum { modepostponed = true };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_RW| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
    static void* address_data( type_inclosure* t ) { return &t->data; }
    static void* address_version( type_inclosure* t ) { return &t->version; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared_rp<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_rp<T>   value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_c_format(); }
    typedef ACCESS_MODE_RP mode;
    enum { modepostponed = true };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_R| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
    static void* address_data( type_inclosure* t ) { return &t->data; }
    static void* address_version( type_inclosure* t ) { return &t->version; }
  };

  template<class T>
  struct Trait_ParamClosure<Shared_wp<T> > {
    typedef kaapi_access_t type_inclosure;
    typedef Shared_wp<T>   value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_c_format(); }
    typedef ACCESS_MODE_WP mode;
    enum { modepostponed = true };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_W| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
    static void* address_data( type_inclosure* t ) { return &t->data; }
    static void* address_version( type_inclosure* t ) { return &t->version; }
  };

  template<class T, class F>
  struct Trait_ParamClosure<Shared_cwp<T, F> > {
    typedef kaapi_access_t  type_inclosure;
    typedef Shared_cwp<T,F> value_type;
    enum { isshared = true };
    static const kaapi_format_t* get_format() { return WrapperFormat<T>::get_c_format(); }
    typedef ACCESS_MODE_CWP mode;
    enum { modepostponed = true };
    enum { xkaapi_mode = KAAPI_ACCESS_MODE_CW| KAAPI_ACCESS_MODE_P };
    template<class S>
    static void link( type_inclosure& f, const S& e) { f = (kaapi_access_t&)e; }
    static void* address_data( type_inclosure* t ) { return &t->data; }
    static void* address_version( type_inclosure* t ) { return &t->version; }
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
    static TASK dummy;
    static void body( kaapi_task_t* task, kaapi_thread_t* stack )
    { 
      dummy();
    }
  };
  template<class TASK>
  TASK KaapiTask0<TASK>::dummy;

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
      Forker( kaapi_thread_t* f, const Attr& a ) : _thread(f), _attr(a) {}

      /**
      **/      
      void operator()()
      { 
        kaapi_task_t* clo = kaapi_thread_toptask( _thread );
        kaapi_task_initdfg( clo, KaapiTask0<TASK>::body, 0 );
        _attr(_thread, clo);
        kaapi_thread_pushtask( _thread );    
      }

#include "athapascan_fork.h"

    protected:
      kaapi_thread_t* _thread;
      const Attr&    _attr;
    };
        
    template<class TASK>
    Forker<TASK, DefaultAttribut> Fork() { return Forker<TASK, DefaultAttribut>(_thread, DefaultAttribut()); }

    template<class TASK, class Attr>
    Forker<TASK, Attr> Fork(const Attr& a) { return Forker<TASK, Attr>(_thread, a); }

  protected:
    kaapi_thread_t* _thread ;
  };

  
  
  // --------------------------------------------------------------------
  /** Top level Fork */
  template<class TASK>
  Thread::Forker<TASK, DefaultAttribut> Fork() { return Thread::Forker<TASK, DefaultAttribut>(kaapi_self_thread(), DefaultAttribut()); }

  template<class TASK, class Attr>
  Thread::Forker<TASK, Attr> Fork(const Attr& a) { return Thread::Forker<TASK, Attr>(kaapi_self_thread(), a); }



  // --------------------------------------------------------------------
  /** Wait execution of all forked tasks of the running task */
  inline void Sync() { ka::Sync(); }



  // --------------------------------------------------------------------
  template<class TASK>
  struct ForkerMain : ka::SpawnerMain<TASK, ka::DefaultAttribut> {
    ForkerMain() 
      : ka::SpawnerMain<TASK, ka::DefaultAttribut>(kaapi_self_thread(), DefaultAttribut())
    {}
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
    void initialize( const std::string& name, void* value, const Format* format, const Format* fupdate);
    void terminate();
    void acquire();
    void release();
    const void* read() const;
    void update(const void* value, const Format* fmtvaue);
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
      KaapiMonotonicBound::initialize( name, value, WrapperFormat<T>::get_format(), WrapperFormatUpdateFnc<FncUpdate>::get_format() );
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
       KaapiMonotonicBound::update( &value, WrapperFormat<Y>::get_format() );
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
#if defined(__APPLE__) && defined(__ppc__) && defined(__GNUC__)
#else  
  inline OStream& operator<< (OStream& m, const long long v )  { return m; }
  inline OStream& operator<< (OStream& m, const unsigned long long v )  { return m; }
#endif
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
#if defined(__APPLE__) && defined(__ppc__) && defined(__GNUC__)
#else  
  inline IStream& operator>> (IStream& m, long long& v )  { return m; }
  inline IStream& operator>> (IStream& m, unsigned long long& v )  { return m; }
#endif
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
namespace a1 {
  using ka::SyncGuard;

  /* for destructor */
  template<class T>
  void Shared<T>::destroy()
  {
    Fork<TaskDelete<T> >()( *this );
  }

}

#ifndef ATHAPASCAN_NOT_IN_NAMESPACE
using namespace a1;
#endif

#endif

