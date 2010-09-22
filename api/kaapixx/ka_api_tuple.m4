ifelse(eval(KAAPI_NUMBER_PARAMS!=KAAPI_MAX_NUMBER_PARAMS),1,
`template<
  M4_PARAM_M(`ifelse(eval(KAAPI_NUMBER_PARAMS>$1-1),1,`ifelse(eval($1>1),1,`, ',`')typename S$1',`')', `', `')
> struct TupleRep<
  M4_PARAM_M(`ifelse(eval(KAAPI_NUMBER_PARAMS>$1-1),1,`S$1',`void')', `', `, ')
>
{
  M4_PARAM(`S$1 r$1;', `', `
  ')
};',`')

ifelse(eval(KAAPI_NUMBER_PARAMS!=KAAPI_MAX_NUMBER_PARAMS),1,
`template<
  M4_PARAM_M(`ifelse(eval(KAAPI_NUMBER_PARAMS>$1-1),1,`ifelse(eval($1>1),1,`, ',`')typename M$1',`')', `', `')
> struct CountTuple<
  M4_PARAM_M(`ifelse(eval(KAAPI_NUMBER_PARAMS>$1-1),1,`M$1',`void')', `', `, ')
>
{
  enum { count = KAAPI_NUMBER_PARAMS};
};',`')