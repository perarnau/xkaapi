
#include <stdio.h>

#include "core_cublas.h"

void core_cublas_init(void)
{
  QUARK_Task_Set_Function_Params(CORE_sgemm_quark, NULL, CORE_sgemm_quark_cublas,
                                 QUARK_ARCH_DEFAULT,
                                 KAAPI_TASK_MIN_PRIORITY);  
  QUARK_Task_Set_Function_Params(CORE_dgemm_quark, NULL, CORE_dgemm_quark_cublas,
                                 QUARK_ARCH_DEFAULT,
                                 KAAPI_TASK_MIN_PRIORITY);
  QUARK_Task_Set_Function_Params(CORE_dgemm_f2_quark, NULL, CORE_dgemm_f2_quark_cublas,
                                 QUARK_ARCH_DEFAULT,
                                 KAAPI_TASK_MIN_PRIORITY);
  
  QUARK_Task_Set_Function_Params(CORE_dtrsm_quark, NULL, CORE_dtrsm_quark_cublas,
                                 QUARK_ARCH_DEFAULT,
                                 KAAPI_TASK_MIN_PRIORITY);
  
  QUARK_Task_Set_Function_Params(CORE_dsyrk_quark, NULL, CORE_dsyrk_quark_cublas,
                                 QUARK_ARCH_DEFAULT,
                                 KAAPI_TASK_MIN_PRIORITY);
  
  QUARK_Task_Set_Function_Params(CORE_dpotrf_quark, NULL, NULL,
                                 QUARK_ARCH_CPU_ONLY,
                                 KAAPI_TASK_MAX_PRIORITY);
  QUARK_Task_Set_Function_Params(CORE_dgetrf_incpiv_quark, NULL, NULL,
                                 QUARK_ARCH_CPU_ONLY,
                                 KAAPI_TASK_MAX_PRIORITY);
  
  QUARK_Task_Set_Function_Params(CORE_dssssm_quark, NULL, CORE_dssssm_quark_cublas,
                                 QUARK_ARCH_DEFAULT,
                                 KAAPI_TASK_MIN_PRIORITY);
  QUARK_Task_Set_Function_Params(CORE_dgessm_quark, NULL, CORE_dgessm_quark_cublas,
                                 QUARK_ARCH_DEFAULT,
                                 KAAPI_TASK_MIN_PRIORITY);
}