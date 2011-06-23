#include <iostream>

class Toto {
public:
  virtual void print() const
  {
    std::cout << "Toto" << std::endl;
  }

  #pragma kaapi task
  void f() const
  {
    this->print();
  }
};

class Titi : public Toto {
public:
  void print() const
  {
    std::cout << "Titi" << std::endl;
  }
};


int main()
{
  Toto t;
  Titi p;
#pragma kaapi parallel 
  {
    t.f();
    p.f();
  }
  return 0;
}