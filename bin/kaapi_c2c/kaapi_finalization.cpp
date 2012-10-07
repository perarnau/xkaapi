/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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


#include "rose_headers.h"
#include "globals.h"


// kaapi finalization pass

void DoKaapiFinalization(SgProject* project)
{
  // add a kaapi_sched_sync all before each synced_call
  synced_stmt_iterator_type pos = all_synced_stmts.begin();
  synced_stmt_iterator_type end = all_synced_stmts.end();
  for (; pos != end; ++pos)
  {
    SgExprStatement* const sync_stmt =
      SageBuilder::buildFunctionCallStmt
      (
       "kaapi_sched_sync", 
       SageBuilder::buildIntType(), 
       SageBuilder::buildExprListExp(),
       pos->second.scope_
      );

    // insert before stmt
    SageInterface::insertStatement(pos->first, sync_stmt, true);
  }
}
