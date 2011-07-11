// vector is passed as a parameter. must not be copied

#include <stdio.h>
#include <vector>


#if 0 // unused, test if there is concurrency

extern "C" unsigned int kaapi_get_self_kid(void);

static void test_concurrency(void)
{
  if (kaapi_get_self_kid())
    printf("%u\n", kaapi_get_self_kid());
}

#endif // unused


class field
{
private:
  std::vector<unsigned int>& f_;
  unsigned int k_;

public:
  field(unsigned int k, std::vector<unsigned int>& f)
    : k_(k), f_(f) {}

  void addForce(unsigned int g)
  {
#pragma kaapi parallel
    {

#pragma kaapi loop
      for (unsigned int i = 0; i < f_.size(); ++i)
      {
	// test_concurrency();
	f_[i] += g * k_;
      } // kaapi loop
    } // kaapi parallel
  }
};

int main(int ac, char** av)
{
  std::vector<unsigned int> v;
  v.resize(1024 * 100);
  for (unsigned int i = 0; i < v.size(); ++i) v[i] = 0;

#define K 2
#define G 9
  field f(K, v);
  f.addForce(G);

  for (unsigned int i = 0; i < v.size(); ++i)
    if (v[i] != K * G)
    {
      printf("invanlid @%u\n", i);
      break ;
    }

  return 0;
}
