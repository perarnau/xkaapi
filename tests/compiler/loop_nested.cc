#include <stdio.h>
#include <vector>

int main(int ac, char** av)
{
  std::vector<unsigned int> iv;
  iv.resize(256 * 100);
  //iv.resize(8);

  std::vector<unsigned int>::iterator ipos = iv.begin();
  std::vector<unsigned int>::iterator iend = iv.end();

#pragma kaapi parallel
  {
#pragma kaapi loop
    for (; ipos != iend; ipos += 4)
      for (size_t i = 0; i < 3; ++i)
	ipos[i] = 42;
  }

  // for (ipos = iv.begin(); ipos != iv.end(); ++ipos)
  //   printf("%u\n", *ipos);

  return 0;
}
