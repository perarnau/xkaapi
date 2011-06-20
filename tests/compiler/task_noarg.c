#pragma kaapi task
static void fu(void) {}

int main(int argc, char** argv)
{
#pragma kaapi start
  fu();
#pragma kaapi finish
  return 0;
}
