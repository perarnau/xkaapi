// --------------------------------------------------------------------
/* KAAPI_NUMBER_PARAMS is the number of possible parameters */
template<class TASK M4_PARAM(`,class F$1', `', ` ')>
struct KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS){ 
 M4_PARAM(`typedef F$1 type_F$1;
  ', ` ', `')
 M4_PARAM(`F$1 f$1;
  ', ` ', `')
  typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> Self_t;
  static void body(kaapi_task_t* t, kaapi_stack_t* stack)
  {
    static TASK dummy;
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`Self_t* args = kaapi_task_getargst(t, Self_t);')
//    Self_t* args = kaapi_task_getargst(t, Self_t);
    dummy(M4_PARAM(`args->f$1', `', `, '));
  }
};
