#pragma kaapi task value(n)
long fibonacci(const long n)
{
  if (n < 2) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

int main(int ac, char** av)
{
#pragma kaapi start
  {
    printf("%ld\n", fibonacci(30));
  }
#pragma kaapi finish
  return 0;
}
