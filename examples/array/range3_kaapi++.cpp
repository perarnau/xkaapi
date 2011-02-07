/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#include <iostream>
#include <stdlib.h>
#include <string>
#include "kaapi++" // this is the new C++ interface for Kaapi

/* Task Print
 * this task prints the sum of the entries of an array 
 * each entries is view as a pointer object:
    array<1,R<int> > means that each entry may be read by the task
 */
struct TaskPrintMatrix : public ka::Task<2>::Signature<std::string,  ka::R<ka::range2d<double> > > {};

template<>
struct TaskBodyCPU<TaskPrintMatrix> {
  void operator() ( std::string s, ka::range2d_r<double> array  )
  {
    size_t d0 = array.dim(0);
    size_t d1 = array.dim(1);
    std::cout << "In '" << s << "' TaskPrintMatrix/CPU, matrix = " << d0 << "x" << d1 << std::endl;
    for (size_t i=0; i < d0; ++i)
    {
      for (size_t j=0; j < d1; ++j)
      {
        std::cout << array(i,j) << " ";
      }
      std::cout << std::endl;
    }
  }
};


/* Main task of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    int n= 10;
    if (argc >1) n = atoi(argv[1]);

    double* data = new double[2*n*2*n];
    for (int i=0; i<2*n; ++i)
      for (int j=0; j<2*n; ++j)
        data[i*n+j] = i*n+j;
    
    /* form a view of data as an 2-dimensional array */
    ka::range2d<double> array(data, n, n, 2*n); 
    int res = 0;
    
    /* Here the dependencies is accross each entries of the array arr
    */
    ka::Spawn<TaskPrintMatrix>()( "A =", array(ka::rangeindex::full, ka::rangeindex::full) );
    ka::Spawn<TaskPrintMatrix>()( "A C1=", array(ka::rangeindex::full, ka::rangeindex(0, n/2)) );
    ka::Spawn<TaskPrintMatrix>()( "A C2=", array(ka::rangeindex::full, ka::rangeindex(n/2, n)) );
    ka::Spawn<TaskPrintMatrix>()( "A 11=", array(ka::rangeindex(0, n/2), ka::rangeindex(0, n/2)) );
    ka::Spawn<TaskPrintMatrix>()( "A 22=", array(ka::rangeindex(n/2, n),ka::rangeindex(n/2, n)) );
    ka::Sync();
    std::cout << "Res = " << res << std::endl;
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
  catch (const std::exception& E) {
    ka::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}
