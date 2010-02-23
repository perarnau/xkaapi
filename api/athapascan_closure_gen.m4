// --------------------------------------------------------------------
/* KAAPI_NUMBER_PARAMS is the number of possible parameters */
/* Fi: format parameters Shared_XX, XX -> XX */
template<class TASK M4_PARAM(`,class F$1', `', ` ')>
struct KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS){ 
 M4_PARAM(`typedef typename Trait_ParamClosure<F$1>::type_inclosure type_inclosure_F$1;
  ', ` ', `')
 M4_PARAM(`type_inclosure_F$1 f$1;
  ', ` ', `')
  typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> Self_t;

  /* */
  static void body(kaapi_task_t* t, kaapi_stack_t* stack)
  {
    static TASK dummy;
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`Self_t* args = kaapi_task_getargst(t, Self_t);')
    dummy(M4_PARAM(`args->f$1', `', `, '));
  }

  static kaapi_format_t    format;
  static kaapi_format_id_t fmid;
  static kaapi_format_t* getformat()
  { return &format; }
  static kaapi_format_t* registerformat()
  {
    if (Self_t::fmid != 0) return &Self_t::format;
    
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_access_mode_t   array_mode[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_offset_t        array_offset[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static const kaapi_format_t* array_format[KAAPI_NUMBER_PARAMS];')
    static Self_t a;
    M4_PARAM(`array_mode[$1-1] = (kaapi_access_mode_t)Trait_ParamClosure<F$1>::xkaapi_mode;
    ',`', `')
    M4_PARAM(`array_offset[$1-1] = (char*)&a.f$1 - (char*)&a;
    ',`', `')
    M4_PARAM(`array_format[$1-1] = Trait_ParamClosure<F$1>::format;
    ',`', `')
    
    Self_t::fmid = kaapi_format_taskregister( 
          &Self_t::getformat, 
          &Self_t::body, 
          typeid(Self_t).name(),
          sizeof(Self_t),
          KAAPI_NUMBER_PARAMS,
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_mode'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_offset'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_format')
      );
    /* extend the set of predefined function */
    return &Self_t::format;
  }
  static const kaapi_task_bodyid_t bodyid;
};

template<class TASK M4_PARAM(`,class F$1', `', ` ')>
kaapi_format_t KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', ` ')>::format;

template<class TASK M4_PARAM(`,class F$1', `', ` ')>
kaapi_format_id_t KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', ` ')>::fmid = 0;

template<class TASK M4_PARAM(`,class F$1', `', ` ')>
const kaapi_task_bodyid_t KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', ` ')>::bodyid = registerformat()->bodyid;


