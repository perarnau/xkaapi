#include <stdio.h>

#include "test-toolbox.h"

void
test_check (const char *test_name, bool success_condition)
{
  if (success_condition)
    printf ("test-toolbox: Test %s: Success!\n", test_name);
  else
    printf ("test-toolbox: Test %s: Failure...\n", test_name);
}
