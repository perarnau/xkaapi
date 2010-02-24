// --------------------------------------------------------------------
/* KAAPI_NUMBER_PARAMS is the number of possible parameters */
/* Fi: format parameters Shared_XX, XX -> XX */
template<>
struct Task<KAAPI_NUMBER_PARAMS> {
  ifelse(KAAPI_NUMBER_PARAMS,0,`',`template<M4_PARAM(`class F$1', `', `, ')>')
  struct Signature { 
    M4_PARAM(`typedef F$1 formal$1_t;
    ', `', `')
    M4_PARAM(`typedef typename Trait_ParamClosure<F$1>::type_inuserfunction type_inuserfunction_F$1;
    ', `', `')
    void operator() ( Thread* thread M4_PARAM(`, type_inuserfunction_F$1', `', `') ) {}
    void get_formal_param ( Thread* thread M4_PARAM(`, F$1 f$1', `', `') ){}
  };
};

/* Kaapi closure representation */
template<class TASK M4_PARAM(`,class F$1', `', ` ')>
struct KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS){ 
 M4_PARAM(`typedef typename Trait_ParamClosure<F$1>::type_inclosure type_inclosure_F$1;
  ', ` ', `')
 M4_PARAM(`typedef typename Trait_ParamClosure<F$1>::type_inuserfunction type_inuserfunction_F$1;
  ', ` ', `')
 M4_PARAM(`type_inclosure_F$1 f$1;
  ', ` ', `')
  typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> Self_t;

  /* */
  static const kaapi_task_bodyid_t bodyid;
};


template<bool hasstackparam, class TASK M4_PARAM(`,class F$1', `', ` ')>
struct KAAPIWRAPPERCPUBODY(KAAPI_NUMBER_PARAMS) {};

/* Kaapi binder to call task with stack args */
template<class TASK M4_PARAM(`,class F$1', `', ` ')>
struct KAAPIWRAPPERCPUBODY(KAAPI_NUMBER_PARAMS)<true, TASK M4_PARAM(`, F$1', `', ` ')> {
 M4_PARAM(`typedef typename Trait_ParamClosure<F$1>::type_inuserfunction type_inuserfunction_F$1;
  ', ` ', `')
  typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> Self_t;

  /* with stack parameters */
  static void body(kaapi_task_t* t, kaapi_stack_t* stack)
  {
    static TaskBodyCPU<TASK> dummy;
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`Self_t* args = kaapi_task_getargst(t, Self_t);')
    dummy( (Thread*)stack ifelse(KAAPI_NUMBER_PARAMS,0,`',`,') M4_PARAM(`(type_inuserfunction_F$1)args->f$1', `', `, '));
  }
};

/* Kaapi binder to call task without stack args */
template<class TASK M4_PARAM(`,class F$1', `', ` ')>
struct KAAPIWRAPPERCPUBODY(KAAPI_NUMBER_PARAMS)<false, TASK M4_PARAM(`, F$1', `', ` ')> {
 M4_PARAM(`typedef typename Trait_ParamClosure<F$1>::type_inuserfunction type_inuserfunction_F$1;
  ', ` ', `')
  typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> Self_t;

  /* */
  static void body(kaapi_task_t* t, kaapi_stack_t* stack)
  {
    static TaskBodyCPU<TASK> dummy;
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`Self_t* args = kaapi_task_getargst(t, Self_t);')
    dummy( M4_PARAM(`(type_inuserfunction_F$1)args->f$1', `', `, '));
  }
};


template<bool hasstackparam, class TASK M4_PARAM(`,class F$1', `', ` ')>
struct KAAPIWRAPPERGPUBODY(KAAPI_NUMBER_PARAMS) {};

/* Kaapi binder to call task with stack args */
template<class TASK M4_PARAM(`,class F$1', `', ` ')>
struct KAAPIWRAPPERGPUBODY(KAAPI_NUMBER_PARAMS)<true, TASK M4_PARAM(`, F$1', `', ` ')> {
 M4_PARAM(`typedef typename Trait_ParamClosure<F$1>::type_inuserfunction type_inuserfunction_F$1;
  ', ` ', `')
  typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> Self_t;

  /* with stack parameters */
  static void body(kaapi_task_t* t, kaapi_stack_t* stack)
  {
    static TaskBodyGPU<TASK> dummy;
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`Self_t* args = kaapi_task_getargst(t, Self_t);')
    dummy( (Thread*)stack ifelse(KAAPI_NUMBER_PARAMS,0,`',`,') M4_PARAM(`(type_inuserfunction_F$1)args->f$1', `', `, '));
  }
};

/* Kaapi binder to call task without stack args */
template<class TASK M4_PARAM(`,class F$1', `', ` ')>
struct KAAPIWRAPPERGPUBODY(KAAPI_NUMBER_PARAMS)<false, TASK M4_PARAM(`, F$1', `', ` ')> {
 M4_PARAM(`typedef typename Trait_ParamClosure<F$1>::type_inuserfunction type_inuserfunction_F$1;
  ', ` ', `')
  typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> Self_t;

  /* */
  static void body(kaapi_task_t* t, kaapi_stack_t* stack)
  {
    static TaskBodyGPU<TASK> dummy;
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`Self_t* args = kaapi_task_getargst(t, Self_t);')
    dummy( M4_PARAM(`(type_inuserfunction_F$1)args->f$1', `', `, '));
  }
};




template<class TASK M4_PARAM(`,class F$1', `', ` ')>
struct KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS) {
  typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> Closure_t;

  static kaapi_format_t    format;
  static kaapi_format_id_t fmid;
  static kaapi_format_t*   getformat()
  { return &format; }
  
  static kaapi_format_t* registerformat()
  {
    if (fmid != 0) return &format;
    
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_access_mode_t   array_mode[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_offset_t        array_offset[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static const kaapi_format_t* array_format[KAAPI_NUMBER_PARAMS];')
    static Closure_t a;
    M4_PARAM(`array_mode[$1-1] = (kaapi_access_mode_t)Trait_ParamClosure<F$1>::xkaapi_mode;
    ',`', `')
    M4_PARAM(`array_offset[$1-1] = (char*)&a.f$1 - (char*)&a;
    ',`', `')
    M4_PARAM(`array_format[$1-1] = Trait_ParamClosure<F$1>::get_format();
    ',`', `')
    static std::string task_name = std::string("__Z")+std::string(typeid(Closure_t).name());
    fmid = kaapi_format_taskregister( 
          &getformat, 
          -1, 
          0, 
          task_name.c_str(),
          sizeof(Closure_t),
          KAAPI_NUMBER_PARAMS,
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_mode'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_offset'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_format')
      );
    return &format;
  }
};


template<class TASK M4_PARAM(`,class F$1', `', ` ')>
struct KAAPI_INITFORMATCLOSURE(KAAPI_NUMBER_PARAMS) {

  M4_PARAM(`typedef typename Trait_ParamClosure<F$1>::type_inuserfunction type_inuserfunction_F$1;
  ', `', `')

  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( M4_PARAM(`type_inuserfunction_F$1', `', `,') ) )
  {
    typedef void (TASK::*type_default_t)(Thread* M4_PARAM(`, type_inuserfunction_F$1', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    return &KAAPIWRAPPERCPUBODY(KAAPI_NUMBER_PARAMS)<false, TASK M4_PARAM(`,F$1', `', ` ')>::body;
  }
  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( Thread* M4_PARAM(`, type_inuserfunction_F$1', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(Thread* M4_PARAM(`, type_inuserfunction_F$1', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    return &KAAPIWRAPPERCPUBODY(KAAPI_NUMBER_PARAMS)<true, TASK M4_PARAM(`,F$1', `', ` ')>::body;
  }

  static kaapi_task_body_t registergpubody( kaapi_format_t* fmt, void (TaskBodyGPU<TASK>::*method)( M4_PARAM(`type_inuserfunction_F$1', `', `,') ) )
  {
    typedef void (TASK::*type_default_t)(Thread* M4_PARAM(`, type_inuserfunction_F$1', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    return &KAAPIWRAPPERGPUBODY(KAAPI_NUMBER_PARAMS)<false, TASK M4_PARAM(`,F$1', `', ` ')>::body;
  }
  static kaapi_task_body_t registergpubody( kaapi_format_t* fmt, void (TaskBodyGPU<TASK>::*method)( Thread* M4_PARAM(`, type_inuserfunction_F$1', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(Thread* M4_PARAM(`, type_inuserfunction_F$1', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    return &KAAPIWRAPPERGPUBODY(KAAPI_NUMBER_PARAMS)<true, TASK M4_PARAM(`,F$1', `', ` ')>::body;
  }


  static kaapi_format_t* registerformat( void (TASK::*method)( Thread* thread M4_PARAM(`, F$1', `', `') ) )
  {
    kaapi_format_t* fmt = KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK  M4_PARAM(`,F$1', `', `')>::registerformat();
    fmt->entrypoint[KAAPI_PROC_TYPE_CPU] = registercpubody( fmt, &TaskBodyCPU<TASK>::operator() );
    fmt->entrypoint[KAAPI_PROC_TYPE_GPU] = registergpubody( fmt, &TaskBodyGPU<TASK>::operator() );
    kaapi_bodies[fmt->bodyid] = fmt->entrypoint[KAAPI_PROC_TYPE_CPU];
    return fmt;
  }
};

template<class TASK M4_PARAM(`,class F$1', `', ` ')>
kaapi_format_t KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', ` ')>::format;

template<class TASK M4_PARAM(`,class F$1', `', ` ')>
kaapi_format_id_t KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', ` ')>::fmid = 0;

template<class TASK M4_PARAM(`,class F$1', `', ` ')>
const kaapi_task_bodyid_t KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', ` ')>::bodyid 
  = KAAPI_INITFORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', ` ')>::registerformat(&TASK::get_formal_param)->bodyid;

