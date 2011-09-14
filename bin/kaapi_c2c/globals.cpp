/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009 INRIA.
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


#include "globals.h"


// all the global variables are defined here

synced_stmt_map_type all_synced_stmts;
std::list<KaapiTaskAttribute*> all_tasks;
std::map<std::string,KaapiTaskAttribute*> all_manglename2tasks;

ListTaskFunctionDeclaration all_task_func_decl;
std::set<SgFunctionDeclaration*> all_signature_func_decl;

// used to mark already instanciated template
std::set<std::string> all_template_instanciate;
std::set<SgFunctionDefinition*> all_template_instanciate_definition;

std::map<std::string,KaapiReduceOperator_t*> kaapi_user_definedoperator;
SgType* kaapi_access_ROSE_type;
SgType* kaapi_task_ROSE_type;
SgType* kaapi_thread_ROSE_type;
SgType* kaapi_frame_ROSE_type;
SgType* kaapi_workqueue_ROSE_type;
SgType* kaapi_workqueue_index_ROSE_type;
SgType* kaapi_stealcontext_ROSE_type;
SgType* kaapi_request_ROSE_type;
SgType* kaapi_taskadaptive_result_ROSE_type;
SgType* kaapi_task_body_ROSE_type;
SgType* kaapi_splitter_context_ROSE_type;
