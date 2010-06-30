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
#include "test_main.h"
#include "test_task.h"

/* Main of the program
*/
void doit::operator()(int argc, char** argv )
{
   ka::pointer<int> p1;
   ka::pointer_rpwp<int> p2;
   ka::pointer_rp<int> p3;
   ka::pointer_wp<int> p4;
   ka::pointer_r<int> p5;
   ka::pointer_w<int> p6;
   ka::pointer_rw<int> p7;

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

   /* failed Rp,R,Wp,W,RW -> RW*/
   ka::Spawn<TaskRW<int> >()(p3);
   ka::Spawn<TaskRW<int> >()(p5);
   ka::Spawn<TaskRW<int> >()(p4);
   ka::Spawn<TaskRW<int> >()(p6);
   ka::Spawn<TaskRW<int> >()(p7);

   /* failed Rp,R,Wp,W,RW -> RpWp*/
   ka::Spawn<TaskRpWp<int> >()(p3);
   ka::Spawn<TaskRpWp<int> >()(p5);
   ka::Spawn<TaskRpWp<int> >()(p4);
   ka::Spawn<TaskRpWp<int> >()(p6);
   ka::Spawn<TaskRpWp<int> >()(p7);
}
