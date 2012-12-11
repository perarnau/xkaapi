KAF_COMMENT(`kaapi error codes')
KAF_CONST_INT(`KAAPIF_SUCCESS',`0')
KAF_CONST_INT(`KAAPIF_ERR_FAILURE',`-1')
KAF_CONST_INT(`KAAPIF_ERR_EINVAL',`-2')
KAF_CONST_INT(`KAAPIF_ERR_UNIMPL',`-3')

KAF_COMMENT(`kaapi data types')
KAF_CONST_INT(`KAAPIF_TYPE_CHAR',`0')
KAF_CONST_INT(`KAAPIF_TYPE_INT',`1')
KAF_CONST_INT(`KAAPIF_TYPE_REAL',`2')
KAF_CONST_INT(`KAAPIF_TYPE_DOUBLE',`3')
KAF_CONST_INT(`KAAPIF_TYPE_PTR',`4')
KAF_CCONST_INT(`KAAPIF_TYPE_MAX',`5')

KAF_COMMENT(`kaapi modes')
KAF_CONST_INT(`KAAPIF_MODE_R',`0')
KAF_CONST_INT(`KAAPIF_MODE_W',`1')
KAF_CONST_INT(`KAAPIF_MODE_RW',`2')
KAF_CONST_INT(`KAAPIF_MODE_V',`3')
KAF_CCONST_INT(`KAAPIF_MODE_MAX',`4')

KAF_COMMENT(`exported functions')
KAF_FUNC(`KAF_CTYPE_INT',`init',`int32_t*')
KAF_FUNC(`KAF_CTYPE_INT',`finalize',`void')
KAF_FUNC(`KAF_CTYPE_DOUBLE',`get_time',`void')
KAF_FUNC(`KAF_CTYPE_INT',`get_concurrency',`void')
KAF_FUNC(`KAF_CTYPE_INT',`get_thread_num',`void')
KAF_PROC(`set_max_tid',`int32_t*')
KAF_FUNC(`KAF_CTYPE_INT',`get_max_tid',`void')
KAF_PROC(`set_grains',`int32_t*, int32_t*')
KAF_PROC(`set_default_grains',`void')
KAF_FUNC(`KAF_CTYPE_INT',`foreach',`
  int32_t*,  /* first */
  int32_t*,  /* last */
  int32_t*,  /* NARGS */
  void (*)(int32_t*, int32_t*, int32_t*, ...), 
  ...  
')
KAF_FUNC(`KAF_CTYPE_INT',`foreach_with_format',`
  int32_t*,  /* first */
  int32_t*,  /* last */
  int32_t*,  /* NARGS */
  void (*)(int32_t*, int32_t*, int32_t*, ...), 
  ...
')
KAF_FUNC(`KAF_CTYPE_INT',`spawn',`
    int32_t*,    /* NARGS */
    void (*f)(), /* F */
    ...
')
KAF_PROC(`sched_sync',`void')
KAF_FUNC(`KAF_CTYPE_INT',`begin_parallel',`void')
KAF_FUNC(`KAF_CTYPE_INT',`end_parallel',`int32_t*')
KAF_FUNC(`KAF_CTYPE_INT',`begin_parallel_tasklist',`void')
KAF_FUNC(`KAF_CTYPE_INT',`end_parallel_tasklist',`void')
KAF_PROC(`get_version',`uint8_t* s')
KAF_COMMENT(`addr to dim conversion')
KAF_COMMENT(`make and get addresses from i, j, ld')
KAF_COMMENT(`dataflow programming requires data to be identified')
KAF_COMMENT(`by a unique address. from this address, the runtime')
KAF_COMMENT(`can compute data dependencies. in this algorithm,')
KAF_COMMENT(`we choose i * n4 + j as a factice address.')
KAF_FUNC(`KAF_CTYPE_INT64',`make_id2',`int64_t* i, int64_t* j, int64_t* ld')
KAF_FUNC(`KAF_CTYPE_INT64',`get_id2_row',`int64_t* id, int64_t* ld')
KAF_FUNC(`KAF_CTYPE_INT64',`get_id2_col',`int64_t* id, int64_t* ld')
