// vector is passed as a parameter. must not be copied

#include <stdio.h>
#include <vector>

int main(int ac, char** av)
{
  std::vector<unsigned int> iv;
  iv.resize(8);

#pragma kaapi parallel
  {
#pragma kaapi loop
    for (unsigned int i = 0; i < iv.size(); ++i) iv[i] = 42;
  }

  for (unsigned int i = 0; i < iv.size(); ++i) printf("%u\n", iv[i]);

  return 0;
}
