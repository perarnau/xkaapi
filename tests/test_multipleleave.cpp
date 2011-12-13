#include <kaapi++>

int test( int argc, char** argv )
{
  std::cout << "Rien ne va plus." << std::endl;
  return 0;
}

int Execute( int argc, char **argv, int (*func)( int, char** ), int returnExecpt = 2, int returnUnknownExcept = 4 )
{
  try
  {
    ka::Community com = ka::System::join_community( argc, argv );

    int run = 0;
    if (func) run = func( argc, argv );

    com.leave();
    ka::System::terminate();
    return run;
  }
  catch(const std::exception& e)
  {
    ka::logfile()<<"Catch : "<<e.what()<<std::endl;
    return returnExecpt;
  } catch(...) {
    ka::logfile()<<"Catch unknown exception"<<std::endl;
    return returnUnknownExcept;
  }
}

int main(int argc, char** argv)
{
  Execute( argc, argv, &test );
  std::cout << "Tout va bien !" << std::endl;

  Execute( argc, argv, &test );
  std::cout << "Tout va bien !" << std::endl;
}

