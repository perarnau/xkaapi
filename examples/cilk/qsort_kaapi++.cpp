/*
 * qsort_cilk2kaapi.cpp
 *
 * An implementation of quicksort based on Cilk parallelization but using Kaapi C++ construction
 * TG
 *
 * Copyright (c) 2007-2008 Cilk Arts, Inc.  55 Cambridge Street,
 * Burlington, MA 01803.  Patents pending.  All rights reserved. You may
 * freely use the sample code to guide development of your own works,
 * provided that you reproduce this notice in any works you make that
 * use the sample code.  This sample code is provided "AS IS" without
 * warranty of any kind, either express or implied, including but not
 * limited to any implied warranty of non-infringement, merchantability
 * or fitness for a particular purpose.  In no event shall Cilk Arts,
 * Inc. be liable for any direct, indirect, special, or consequential
 * damages, or any other damages whatsoever, for any use of or reliance
 * on this sample code, including, without limitation, any lost
 * opportunity, lost profits, business interruption, loss of programs or
 * data, even if expressly advised of or otherwise aware of the
 * possibility of such damages, whether in an action of contract,
 * negligence, tort, or otherwise.
 *
 */
#include <kaapi++>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <functional>

// Sort the range between bidirectional iterators begin and end.
// end is one past the final element in the range.
// Use the Quick Sort algorithm, using recursive divide and conquer.
// This function is NOT the same as the Standard C Library qsort() function.
// This implementation is pure C++ code before Cilk++ conversion.
struct Task_QSort : public ka::Task<1>::Signature<ka::RW< ka::range1d<int> > > {};

template<>
struct TaskBodyCPU<Task_QSort>
{
  void operator()(ka::range1d_rw<int> range)
  {
    int* begin = range.begin();
    int* end   = range.end();
    if (begin != end) 
    {
        --end;  // Exclude last element (pivot) from partition
        
        // required explicit cast to int*
        int* middle 
            = std::partition( (int*)begin, (int*)end, 
                              std::bind2nd(std::less<int>(), *end));
        std::swap(*end, *middle);    // move pivot to middle
        ka::Spawn<Task_QSort> ()( ka::array<1,int>(begin, middle-begin) );
        (*this)( ka::range1d<int>(middle+1, end-middle) );
//        (*this)(++middle, ++end); // Exclude pivot and restore end
    }
  }
};

// A simple test harness 
int qmain(int n)
{
    int* a = new int[n];

    for (int i = 0; i < n; ++i)
        a[i] = i;

    std::random_shuffle((int*)a, (int*)(a + n));
    std::cout << "Sorting " << n << " integers" << std::endl;

    double t0 = kaapi_get_elapsedtime();
    ka::Spawn<Task_QSort>()( ka::array<1,int>(a, n) );
    ka::Sync();
    double t1 = kaapi_get_elapsedtime();

    std::cout << (t1-t0) << " seconds" << std::endl;

    // Confirm that a is sorted and that each element contains the index.
    for (int i = 0; i < n - 1; ++i) {
      if (a[i] >= a[i + 1] || a[i] != i) {
          std::cout << "Sort failed at location i=" << i << " a[i] = "
                    << a[i] << " a[i+1] = " << a[i + 1] << std::endl;
          delete[] a;
          return 1;
      }
    }
    std::cout << "Sort succeeded." << std::endl;
    delete[] a;

    return 0;
}

int main(int argc, char* argv[])
{
  ka::Community com = ka::System::join_community(argc,argv);
  int n = 10 * 1000 * 1000;
  if (argc > 1) {
       n = std::atoi(argv[1]);
       if (n <= 0) {
            std::cerr << "Invalid argument" << std::endl;
            std::cerr << "Usage: qsort N" << std::endl;
            std::cerr << "       N = number of elements to sort" << std::endl;
            return 1;
       }
  }

  int err = qmain(n);
  com.leave();
  ka::System::terminate();
  return err;
}
