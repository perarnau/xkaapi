#include <stdio.h>
#include <vector>

int main(int ac, char** av)
{
  std::vector<unsigned int> iv;
  std::vector<unsigned int> ov;

  iv.resize(8);
  ov.resize(iv.size());
  for (unsigned int i = 0; i < iv.size(); ++i) iv[i] = 42;

  std::vector<unsigned int>::iterator ipos = iv.begin();
  std::vector<unsigned int>::iterator iend = iv.end();
  std::vector<unsigned int>::iterator opos = ov.begin();

#pragma kaapi parallel
  {
#pragma kaapi loop
    for (; ipos != iend; ++opos, ipos = ipos + 1) *opos = *ipos;
  }

  for (unsigned int i = 0; i < ov.size(); ++i) printf("%u\n", ov[i]);

  return 0;
}
