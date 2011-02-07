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
#include "kaapi++" // this is the new C++ interface for Kaapi

// --------------------------------------------------------------------
/* Task Print
 * this task prints the sum of the entries of an array 
 * each entries is view as a pointer object:
    array<1,R<int> > means that each entry may be read by the task
 */
struct TaskPrint : public ka::Task<2>::Signature<int, ka::R<ka::range1d<double> > > {};

template<>
struct TaskBodyCPU<TaskPrint> {
  void operator() ( int beg, ka::range1d_r<double> array  )
  {
    size_t d0 = array.size();
    std::cout << "In TaskPrint/CPU, array = " << d0 << std::endl;
    std::cout << "at " << beg << "=[ ";
    for (size_t i=0; i < d0; ++i)
      std::cout << array[i] << " ";
    std::cout << " ]" << std::endl;
  }
};



// --------------------------------------------------------------------
/* Task Init
 */
struct TaskInit : public ka::Task<1>::Signature<ka::W<ka::range1d<double> > > {};

template<>
struct TaskBodyCPU<TaskInit> {
  void operator() ( ka::range1d_w<double> array  )
  {
    size_t d0 = array.size();
    std::cout << "In TaskInit/CPU, array = " << d0 << std::endl;
    for (size_t i=0; i < d0; ++i)
      array[i] = 1.0;
  }
};



// --------------------------------------------------------------------
struct ExtractInterface: public ka::Task<3>::Signature< 
        ka::R<ka::range1d<double> >,
        int,
        ka::W<double>
> {};

template<>
struct TaskBodyCPU<ExtractInterface> {
  void operator() ( ka::range1d_r<double> domain,
                    int                   dir, 
                    ka::pointer_w<double> sdi 
  )
  {
    if (dir == -1) *sdi = domain[0];
    else *sdi = domain[domain.size()-1];
  }
};



// --------------------------------------------------------------------
struct UpdateInternal: public ka::Task<2>::Signature< 
        ka::W<ka::range1d<double> >,
        ka::R<ka::range1d<double> > 
> {};

template<> struct TaskBodyCPU<UpdateInternal> {
  void operator() ( ka::range1d_w<double> new_domain, 
                    ka::range1d_r<double> old_domain )
  {
    size_t sz = new_domain.size();
    for (size_t i=1; i<sz-1; ++i)
      new_domain[i] = old_domain[i-1] + old_domain[i] + old_domain[i+1];
  }
};



// --------------------------------------------------------------------
struct UpdateExternal: public ka::Task<3>::Signature< 
        ka::RW<ka::range1d<double> >,
        int,
        ka::R<double> 
> {};

template<> struct TaskBodyCPU<UpdateExternal> {
  void operator() ( ka::range1d_rw<double> new_domain, 
                    int dir,
                    ka::pointer_r<double>  value )
  {
    if (dir == -1) new_domain[0] += *value;
    else new_domain[new_domain.size()-1] += *value;
  }
};



// --------------------------------------------------------------------
/** Pseudo Jacobi 1D kernel
*/
struct Kernel {
  void operator() ( int n, int blocksize,
                    ka::array<1,double> old_domain,
                    ka::array<1,double> new_domain,
                    ka::array<1,double> sdi_left,
                    ka::array<1,double> sdi_right
                  )
  {
   
    for (int niter = 0; niter<2; ++niter)
    {
      // extract fontier
      for (int i =0; i<n; i+=blocksize)
      {
        int beg  = i; 
        int end  = (i+blocksize) <n ? i+blocksize : n;
        int site = beg/blocksize;
        if (beg-1 >0) 
        {
          ka::Spawn<ExtractInterface>(ka::SetPartition(site)) ( 
                       old_domain[ka::rangeindex(beg, end)],
                       -1,
                       &sdi_left[site]
          );
        }
        if (end+1 <n) {
          ka::Spawn<ExtractInterface>(ka::SetPartition(site)) ( 
                       old_domain[ka::rangeindex(beg, end)],
                       1,
                       &sdi_right[site]
          );
        }
      }
      
      // internal update
      for (int i =0; i<n; i+=blocksize)
      {
        int beg  = i; 
        int end  = (i+blocksize) <n ? i+blocksize : n;
        int site = beg/blocksize;
        ka::Spawn<UpdateInternal>(ka::SetPartition(site)) ( 
                       new_domain[ ka::rangeindex(beg, end) ],
                       old_domain[ ka::rangeindex(beg, end) ]
        );
      }

      // external update
      for (int i =0; i<n; i+=blocksize)
      {
        int beg  = i; 
        int end  = (i+blocksize) <n ? i+blocksize : n;
        int site = beg/blocksize;
        if (beg-1 >0) 
        {
          ka::Spawn<UpdateExternal>(ka::SetPartition(site)) ( 
                       new_domain[ka::rangeindex(beg, end)],
                       -1,
                       &sdi_right[site-1]
          );
        }
        if (end+1 <n) 
        {
          ka::Spawn<UpdateExternal>(ka::SetPartition(site)) ( 
                       new_domain[ka::rangeindex(beg, end)],
                       1,
                       &sdi_left[site+1]
          );
        }
      }
      
      for (int i =0; i<n; i+=blocksize)
      {
        int beg  = i; 
        int end  = (i+blocksize) <n ? i+blocksize : n;
        int site = beg/blocksize;
        ka::Spawn<TaskPrint>(ka::SetPartition(site)) ( beg, old_domain[ka::rangeindex(beg, end)] );
      }
      
      old_domain.swap( new_domain );
    }

  }

};



// --------------------------------------------------------------------
/** Main Task, only executed on one process.
*/
struct doit {
  void operator()(int argc, char** argv )
  {   
    ka::logfile() << "Starting Pseudo Jacobi" << std::endl;  

    // number of partitions = X*Y*Z
    int n = 20;
    if (argc > 1) n = atoi(argv[1]);
    int blocksize = 10;
    if (argc > 2) blocksize = atoi(argv[2]);
    int niter = 4;
    
    ka::ThreadGroup threadgroup( (n+blocksize-1)/blocksize );

    // Computation domains
    ka::array<1,double> old_domain( new double[n], n );
    ka::array<1,double> new_domain( new double[n], n );


    ka::array<1,double> sdi_left ( new double[threadgroup.size()], threadgroup.size() );
    ka::array<1,double> sdi_right( new double[threadgroup.size()], threadgroup.size() );

    for (int i=0; i<n; ++i)
    {
      old_domain[i] = 1.0;
      new_domain[i] = 0.0;
    }

    threadgroup.ForEach<Kernel>()
      ( ka::counting_iterator<int>(0), ka::counting_iterator<int>(niter) ) /* iteration space */
      ( n, blocksize, old_domain, new_domain, sdi_left, sdi_right );       /* args for the kernel */
  }
};


// --------------------------------------------------------------------
/** Main of the program
*/
int main( int argc, char** argv ) 
{
  try {
    ka::Community com = ka::System::join_community( argc, argv );

    ka::SpawnMain<doit>()(argc, argv); 
    com.leave();
    ka::System::terminate();
  }
  catch (const std::exception& ex) {
    std::cerr << "[main] catch exception: " << ex.what() << std::endl;
  }
  catch (...) {
    std::cerr << "[main] catch unknown exception" << std::endl;
  }
  return 0;
}
