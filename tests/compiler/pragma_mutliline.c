#pragma kaapi task \
write(bar) \
read(baz)
static void fu(double* bar, double* baz) {}

int main(int argc, char** argv)
{
#pragma kaapi start
  { fu(0, 0); }
#pragma kaapi barrier
#pragma kaapi finish
  return 0;
}
