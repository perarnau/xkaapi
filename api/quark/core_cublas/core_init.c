
#include <stdio.h>

#include "core_cublas.h"

void core_cublas_init(void)
{
  QUARK_Task_Set_GPU_Function(CORE_dgemm_quark, CORE_dgemm_quark_cublas, QUARK_ARCH_DEFAULT);
  QUARK_Task_Set_GPU_Function(CORE_dgemm_f2_quark, CORE_dgemm_f2_quark_cublas, QUARK_ARCH_DEFAULT);
  QUARK_Task_Set_GPU_Function(CORE_dtrsm_quark, CORE_dtrsm_quark_cublas, QUARK_ARCH_DEFAULT);
  QUARK_Task_Set_GPU_Function(CORE_dsyrk_quark, CORE_dsyrk_quark_cublas, QUARK_ARCH_DEFAULT);
  QUARK_Task_Set_GPU_Function(CORE_dssssm_quark, CORE_dssssm_quark_cublas, QUARK_ARCH_DEFAULT);  
}