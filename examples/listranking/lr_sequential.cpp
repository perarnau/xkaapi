/*
** xkaapi
** 
** Copyright 2010 INRIA.
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
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "lr_list.h"
#include "kaapi++"  /* only for get_elapsedtime() */


int main(int argc, char * argv[])
{
	long int i, j, first_el;

	list LV;

  list::size_t num_elements = (list::size_t)atoi(argv[1]);
	if (num_elements < 1)
	{
		printf("The number of the elements on the list should be 1 at least\n");
    exit(1);
	}
  
	LV.resize(num_elements);
#if DEBUG
	printf("Creating the list...\n");
#endif

  /* randomize the list */
  LR.randomize();
  if (num_elements < 100)
  {
    LV.print( std::cout << "List is:" );
    std::cout << std::endl;
  }

  double start = kaapi_get_elapsedtime();
  /* find the head of the list */
  list::index_t h_head = LS.head();
  list::index_t lR;
  list::index_t le = LV.lr_head( h_head, LV[h_head].nS, lR);
  LV[le].R = lR+1;
  double stop = get_elapsedtime();

#if DEBUG
  std::cout << "Head of the list is at index: " << h_head << std::endl;
  std::cout << "Last rank of the list is    : " << LV[lr].R << std::endl;
#endif

  if (LV.testOK() ) 
  {
    std::cout << "List is correctly ranked" << std::endl;
  }
  
  std::cout << "LR in time:" << stop - start << " s" << std::endl;
  if (num_elements < 100)
  {
    LV.print( std::cout << "After list ranking:\n" );
    std::cout << std::endl;
  }
  
	return 0;
}

