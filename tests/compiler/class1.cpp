class Toto {
public:
  #pragma kaapi task
  void f_non_const()
  {
  }


  #pragma kaapi task
  void f() const
  {
  }
};


int main()
{
  Toto t;
  t.f();
  Toto* p;
  p->f();
#pragma kaapi sync
  return 0;
}