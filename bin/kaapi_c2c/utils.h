/*
** xkaapi
** 
**
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


#ifndef UTILS_H_INCLUDED
# define UTILS_H_INCLUDED


#include <stdlib.h>
#include <string>
#include <vector>
#include <set>
#include "rose_headers.h"


#define SOURCE_POSITION \
Sg_File_Info::generateDefaultFileInfoForTransformationNode()


SgClassDeclaration* buildClassDeclarationAndDefinition
(
  const std::string& name, 
  SgScopeStatement* scope
);

SgClassDeclaration* buildStructDeclaration
( 
 SgScopeStatement* scope,
 const std::vector<SgType*>& memberTypes, 
 const std::vector<std::string>& memberNames,
 const std::string& structName
);

SgVariableDeclaration* buildStructVariable
(
 SgScopeStatement* scope,
 const std::vector<SgType*>& memberTypes, 
 const std::vector<std::string>& memberNames,
 const std::string& structName = "",
 const std::string& varName = "", 
 SgInitializer *initializer = NULL
);

SgVariableDeclaration* buildStructVariable
(
 SgScopeStatement* scope,
 SgClassDeclaration* classDeclaration,
 const std::string& varName = "",
 SgInitializer *initializer = NULL
);

SgVariableDeclaration* buildStructPointerVariable
( 
 SgScopeStatement* scope,
 SgClassDeclaration* classDeclaration,
 const std::string& varName,
 SgInitializer *initializer = NULL
);

SgVariableDeclaration* buildInsertDeclarationKaapiThread
(
 SgScopeStatement* scope,
 SgScopeStatement* append = NULL
);

SgVariableDeclaration* buildThreadVariableDecl
(
 SgScopeStatement* scope
);

void buildVariableAssignment
(
 SgExprStatement* cur_stmt,
 SgVariableDeclaration* var_decl,
 SgScopeStatement* scope
);

SgClassDeclaration* buildOutlineArgStruct
(
 const std::set<SgVariableSymbol*>& listvar,
 SgGlobal* scope,
 SgClassDefinition* this_class_def,
 const char* name_prefix = NULL,
 bool do_pointers = true
);


#endif // ! UTILS_H_INCLUDED
