#include <stdio.h>

int main(int ac, char** av)
{
#define MAX_CASE 32

  unsigned int i;
  unsigned int j;

  printf("#ifndef KAAPIC_DFG_SWITCH_H_INCLUDED\n");
  printf("#define KAAPIC_DFG_SWITCH_H_INCLUDED\n");
  printf("\n");

  printf("#define KAAPIC_MAX_ARGS %u\n",MAX_CASE);
  printf("#define KAAPIC_DFG_SWITCH(__ti) \\\n");
  printf("switch((__ti)->nargs)\\\n");
  printf("{\\\n");

  for (i = 0; i < MAX_CASE; ++i)
  {
    printf("case %u: (__ti)->body(", i);
    for (j = 0; j < i; ++j)
    {
      if (j) printf(", ");
      printf("(__ti)->args[%u].access.data", j);
    }
    printf("); break ;\\\n");
  }

  printf("default: FATAL(); break ;\\\n");
  printf("}\n");

  printf("\n");
  printf("#endif\n");

  return 0;
}
