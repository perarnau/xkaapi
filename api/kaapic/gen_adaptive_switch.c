#include <stdio.h>

int main(int ac, char** av)
{
#define MAX_CASE 32

  unsigned int i;
  unsigned int j;

  printf("#ifndef KAAPIF_ADAPTIVE_SWITCH_H_INCLUDED\n");
  printf("#define KAAPIF_ADAPTIVE_SWITCH_H_INCLUDED\n");
  printf("\n");

  printf("#define KAAPIF_ADAPTIVE_SWITCH(__w, __i, __j, __tid) \\\n");
  printf("switch((__w)->nargs)\\\n");
  printf("{\\\n");

  for (i = 0; i < MAX_CASE; ++i)
  {
    printf("case %u: (__w)->f(__i, __j, __tid", i);

    for (j = 0; j < i; ++j)
      printf(", (__w)->args[%u]", j);

    printf("); break ;\\\n");
  }

  printf("default: FATAL(); break ;\\\n");
  printf("}\n");

  printf("\n");
  printf("#endif\n");

  return 0;
}
