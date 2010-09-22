template<
  M4_PARAM(`typename M$1 =void, typename S$1 =void', `', `, ')
> struct Tuple : public TupleRep<M4_PARAM(`S$1', `', `,')> {
  M4_PARAM(`typedef typename TraitUAMParam<M$1>::mode_t mode$1_t;', `', `
  ')
  enum { count =CountTuple<M4_PARAM(`M$1', `', `,')>::count };
};

// --------------------------------------------------------------------
