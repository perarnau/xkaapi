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

  static TASK dummy;
  /* */
  static void body(void* taskargs, kaapi_thread_t* thread)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`Self_t* args = (Self_t*)taskargs;')
    dummy(M4_PARAM(`(F$1)args->f$1', `', `, '));
  }

#if 0
  static Format*           format;
  static kaapi_format_id_t fmid;
  static kaapi_format_t*   getformat() { 
    std::cout << "HERE/ATHA" << std::endl;
    if (format==0) format = new Format; return format->get_c_format(); 
  }
#endif
  static kaapi_format_t* registerformat()
  {
#if 0
    if (Self_t::fmid != 0) return getformat();
#endif
    
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_access_mode_t   array_mode[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_offset_t        array_offset_data[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_offset_t        array_offset_version[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static const kaapi_format_t* array_format[KAAPI_NUMBER_PARAMS];')
    static Self_t a;
    M4_PARAM(`array_mode[$1-1] = (kaapi_access_mode_t)Trait_ParamClosure<F$1>::xkaapi_mode;
    ',`', `')
    M4_PARAM(`array_offset_data[$1-1] = (char*)Trait_ParamClosure<F$1>::address_data(&a.f$1) - (char*)&a;
    ',`', `')
    M4_PARAM(`array_offset_version[$1-1] = (char*)Trait_ParamClosure<F$1>::address_version(&a.f$1) - (char*)&a;
    ',`', `')
    M4_PARAM(`array_format[$1-1] = Trait_ParamClosure<F$1>::get_format();
    ',`', `')
    
    static ka::FormatTask task_fmt( 
          typeid(TASK).name(),
          sizeof(Self_t),
          KAAPI_NUMBER_PARAMS,
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_mode'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_offset_data'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_offset_version'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_format'),
          0,
          0,
          0,
          0
      );
    kaapi_format_taskregister_body( task_fmt.get_c_format(), Self_t::body, 0, KAAPI_PROC_TYPE_CPU );
      
    return task_fmt.get_c_format();
  }
  static kaapi_task_body_t registerbody()
  {
    registerformat();
    return KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', ` ')>::body;
  }
  
  static const kaapi_task_body_t default_body;
};

template<class TASK M4_PARAM(`,class F$1', `', ` ')>
TASK KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', ` ')>::dummy;

template<class TASK M4_PARAM(`,class F$1', `', ` ')>
const kaapi_task_body_t KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', ` ')>::default_body = KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', ` ')>::registerbody();


