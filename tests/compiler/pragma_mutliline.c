#pragma kaapi task \
input(bar) \
output(baz)
void fu(void* bar, void* baz) {}

int main(int argc, char** argv)
{
#pragma kaapi start
  fu(0, 0);
#pragma kaapi finish
  return 0;
}
