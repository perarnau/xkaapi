// vector is passed as a parameter. must not be copied

#include <stdio.h>
#include <vector>

class field
{
  unsigned int k_;

public:
  void addForce(unsigned int g)
  {
    unsigned int sum = 0;

#pragma kaapi parallel
    {
#pragma kaapi loop
    for (unsigned int i = 0; i < 42; ++i)
      sum += g * k_;
    }
  }
};

int main(int ac, char** av)
{
  field f;
  f.addForce(9.8);
  return 0;
}
