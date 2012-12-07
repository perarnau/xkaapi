KAF_GROUP(`kaapi error codes',`dnl
KAF_INT(`KAAPIF_SUCCESS',`0')dnl
KAF_INT(`KAAPIF_ERR_FAILURE',`-1')dnl
KAF_INT(`KAAPIF_ERR_EINVAL',`-2')dnl
KAF_INT(`KAAPIF_ERR_UNIMPL',`-3')dnl
')dnl
KAF_GROUP(`kaapi data types',`dnl
KAF_INT(`KAAPIF_TYPE_CHAR',`0')dnl
KAF_INT(`KAAPIF_TYPE_INT',`1')dnl
KAF_INT(`KAAPIF_TYPE_REAL',`2')dnl
KAF_INT(`KAAPIF_TYPE_DOUBLE',`3')dnl
KAF_INT(`KAAPIF_TYPE_PTR',`4')dnl
KAF_INT_H(`KAAPIF_TYPE_MAX',`5')dnl
')dnl
KAF_GROUP(`kaapi modes',`dnl
KAF_INT(`KAAPIF_MODE_R',`0')dnl
KAF_INT(`KAAPIF_MODE_W',`1')dnl
KAF_INT(`KAAPIF_MODE_RW',`2')dnl
KAF_INT(`KAAPIF_MODE_V',`3')dnl
KAF_INT_H(`KAAPIF_MODE_MAX',`4')dnl
')dnl
