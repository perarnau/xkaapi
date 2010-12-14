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


list::index_t JajaSequential( list& LV, list::size_t num_splitters )
{
  /* find the head of the list */
  list::index_t h_head = LV.head();

  /* Step1: generate num_splitters sublists */
  sublist* sLV = new (alloca(sizeof(sublist)*num_splitters)) sublist[num_splitters];
  if (sLV ==0) exit(1);
  sLV[0].head    = h_head;
  sLV[0].save_nS = LV[h_head].nS;
  LV[h_head].nS  = -1;
  LV[h_head].R   = 0;
  LV.split( sLV, 1, num_splitters );

  if (LV.size() <100)
  {
    LV.print( std::cout << "\nAfter splitting the list:\n" );
    std::cout << std::endl;
  }
  
  /* Step 2: compute list ranking of each sublist */
  for (list::size_t j = 0; j<num_splitters; ++j)
  {
    /* return the last element of the sublist (begin of the next sublist) */
    list::index_t le = LV.lr_head( sLV[j].head, sLV[j].save_nS, sLV[j].R);
    list::index_t nsl = -LV[le].nS;
    if (nsl == (list::index_t)LV.size())
      sLV[j].next = -1;
    else 
      sLV[j].next = nsl -1; 
  }
  
  /* Step 3: sequentual prefix computation of the first rank of each sub list */
  sLV[0].pR = 0;
  list::index_t is  = 0;
  list::index_t ins = sLV[is].next;
  
  while (ins != -1)
  {
    sLV[ins].pR = 1+sLV[is].R + sLV[is].pR;
    is = ins;
    ins = sLV[is].next;
  }
    
  /* step 4: parallel update rank of each sublist
     This is the original Jaja & Helman algorithm, without using auxilary array to
     speedup global rank update.
  */
  list::index_t sh_head = sLV[0].head;
  LV[sh_head].nS = sLV[0].save_nS;
  for (list::size_t j = 0; j<num_splitters; ++j)
  {
    sh_head = sLV[j].head;
    list::index_t pR = sLV[j].pR;
    list::index_t last_sublist;
    list::index_t nsh_head;
    nsh_head = sLV[j].save_nS;
    LV[sh_head].nS = nsh_head;
    if (sLV[j].next == -1) 
      last_sublist = -LV.size(); 
    else 
      last_sublist = sLV[ sLV[j].next ].head;
    while (1)
    {
      LV[sh_head].R += pR;
      if (nsh_head == last_sublist) break;
      sh_head = nsh_head;
      nsh_head = LV[sh_head].nS;
    }
    if (nsh_head == -(list::index_t)LV.size()) LV[sh_head].R = (list::index_t)LV.size()-1;
  }

  return h_head;
}


/*
*/
int main(int argc, char * argv[])
{
	list LV;

  list::index_t num_elements  = 10;
  list::index_t num_splitters = 2;
  
  if (argc >1) num_elements  = (long int)atoi(argv[1]);
  if (argc >2) num_splitters = (long int)atoi(argv[2]);

	if (num_elements < 2)
	{
		printf("The number of the elements on the list should be 1 at least\n");
    exit(1);
	}
	if (num_splitters > num_elements/2)
	{
		printf("Too important number of splitters. It should be less than n/2\n");
    exit(1);
	}
  
	LV.resize(num_elements);

  /* randomize the list */
  LV.randomize();
  if (num_elements < 100)
  {
    LV.print( std::cout << "List is:\n" );
    std::cout << std::endl;
  }

  /* call the Jaja & Helman algorithm part of the computation */
  double start = kaapi_get_elapsedtime();
  list::index_t h_head = JajaSequential(LV, num_splitters);
  double stop = kaapi_get_elapsedtime();

  std::cout << "head of the list is at index: " << h_head << std::endl;
  if (LV.testOK( ) ) 
  {
    std::cout << "List is correctly ranked" << std::endl;
  }
  
  std::cout << "LR in time:" << stop - start << " s" << std::endl;
  if (num_elements <100)
  {
    LV.print( std::cout << "\nAfter splitting the list:\n" );
    std::cout << std::endl;
  }
  
	return 0;
}

