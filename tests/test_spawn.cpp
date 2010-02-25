/****************************************************************************
 * 
 *  Test spawn of task
 *
 ***************************************************************************/
#include "test_task.h"

/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
     /* rpwp -> all other modes */
     ka::pointer<int> p1;
     ka::Spawn<TaskR<int> >()(p1);
     ka::Spawn<TaskW<int> >()(p1);
     ka::Spawn<TaskRW<int> >()(p1);
     ka::Spawn<TaskRp<int> >()(p1);
     ka::Spawn<TaskWp<int> >()(p1);
     ka::Spawn<TaskRpWp<int> >()(p1);

     /* rpwp -> all other modes */
     ka::pointer_rpwp<int> p2;
     ka::Spawn<TaskR<int> >()(p2);
     ka::Spawn<TaskW<int> >()(p2);
     ka::Spawn<TaskRW<int> >()(p2);
     ka::Spawn<TaskRp<int> >()(p2);
     ka::Spawn<TaskWp<int> >()(p2);
     ka::Spawn<TaskRpWp<int> >()(p2);

     /* rp -> r / rp */
     ka::pointer_rp<int> p3;
     ka::Spawn<TaskR<int> >()(p3);
     ka::Spawn<TaskRp<int> >()(p3);

     /* wp -> w / wp */
     ka::pointer_wp<int> p4;
     ka::Spawn<TaskW<int> >()(p4);
     ka::Spawn<TaskWp<int> >()(p4);

     /* r -> r */
     ka::pointer_r<int> p5;
     ka::Spawn<TaskR<int> >()(p5);

     /* w -> w, only if terminal */ 
     ka::pointer_w<int> p6;
     ka::Spawn<TaskW<int> >()(p6);

     /* failed Rp,R -> W*/
     ka::Spawn<TaskW<int> >()(p2);
     ka::Spawn<TaskW<int> >()(p5);

     /* failed Rp,R -> Wp*/
     ka::Spawn<TaskWp<int> >()(p2);
     ka::Spawn<TaskWp<int> >()(p5);

     /* failed Wp,W -> R*/
     ka::Spawn<TaskR<int> >()(p4);
     ka::Spawn<TaskR<int> >()(p6);

     /* failed Wp,W -> R*/
     ka::Spawn<TaskRp<int> >()(p4);
     ka::Spawn<TaskRp<int> >()(p6);
  }
};


/* main entry point : Kaapi initialization
*/
int main(int argc, char** argv)
{
  try {
    /* Join the initial group of computation : it is defining
       when launching the program by a1run.
    */
    ka::Community com = ka::System::join_community( argc, argv );
    
    /* Start computation by forking the main task */
    ka::SpawnMain<doit>()(argc, argv); 
    
    /* Leave the community: at return to this call no more athapascan
       tasks or shared could be created.
    */
    com.leave();

    /* */
    ka::System::terminate();
  }
  catch (const ka::InvalidArgumentError& E) {
    ka::logfile() << "Catch invalid arg" << std::endl;
  }
  catch (const ka::BadAlloc& E) {
    ka::logfile() << "Catch bad alloc" << std::endl;
  }
  catch (const ka::Exception& E) {
    ka::logfile() << "Catch : "; E.print(std::cout); std::cout << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}

