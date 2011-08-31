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


#ifndef GLOBALS_H_INCLUDED
# define GLOBALS_H_INCLUDED


#include <string>
#include <set>
#include <list>
#include <map>
#include <algorithm>
#include "rose_headers.h"
#include "kaapi_task.h"


typedef struct synced_stmt
{
  // synced calls needs scope + varlist
  SgScopeStatement* scope_;
  std::list<SgInitializedName*> var_;

  synced_stmt() : scope_(NULL) {}

} synced_stmt_t;

typedef std::pair<SgStatement*, synced_stmt> synced_stmt_pair_type;
typedef std::map<SgStatement*, synced_stmt > synced_stmt_map_type;
typedef synced_stmt_map_type::iterator synced_stmt_iterator_type;
extern synced_stmt_map_type all_synced_stmts;

extern std::list<KaapiTaskAttribute*> all_tasks;
extern std::map<std::string,KaapiTaskAttribute*> all_manglename2tasks;

typedef std::list<std::pair<SgFunctionDeclaration*, std::string> > ListTaskFunctionDeclaration;
extern ListTaskFunctionDeclaration all_task_func_decl;
extern std::set<SgFunctionDeclaration*> all_signature_func_decl;

// used to mark already instanciated template
extern std::set<std::string> all_template_instanciate; 
extern std::set<SgFunctionDefinition*> all_template_instanciate_definition;

extern std::map<std::string,KaapiReduceOperator_t*> kaapi_user_definedoperator;

extern SgType* kaapi_access_ROSE_type;
extern SgType* kaapi_task_ROSE_type;
extern SgType* kaapi_thread_ROSE_type;
extern SgType* kaapi_frame_ROSE_type;
extern SgType* kaapi_workqueue_ROSE_type;
extern SgType* kaapi_workqueue_index_ROSE_type;
extern SgType* kaapi_stealcontext_ROSE_type;
extern SgType* kaapi_request_ROSE_type;
extern SgType* kaapi_taskadaptive_result_ROSE_type;
extern SgType* kaapi_task_body_ROSE_type;
extern SgType* kaapi_splitter_context_ROSE_type;


#endif // ! GLOBALS_H_INCLUDED
