// --------------------------------------------------------------------
// View for task arguments description
template<
  M4_PARAM(`typename M$1', `', `, ')
> struct CountTuple {
  enum { count = KAAPI_NUMBER_PARAMS };
};


template<
  M4_PARAM(`typename S$1', `', `, ')
> struct TupleRep {
  M4_PARAM(`S$1 r$1;', `', `
  ')
};
