class Toto {
public:
  #pragma kaapi task
  int compute()
  {
  }
};


int main()
{
  int res;
  Toto t;
  Toto* p;

  t.compute();
  
  res = p->compute();
#pragma kaapi sync
  return 0;
}