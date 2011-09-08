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


#include <assert.h>
#include <string>
#include <vector>
#include "rose_headers.h"
#include "globals.h"
#include "utils.h"


SgClassDeclaration*
buildClassDeclarationAndDefinition (
  const std::string& name, 
  SgScopeStatement* scope
)
{
  // This function builds a class declaration and definition 
  // (both the defining and nondefining declarations as required).

  // Build a file info object marked as a transformation
  Sg_File_Info* fileInfo = Sg_File_Info::generateDefaultFileInfoForTransformationNode();
  assert(fileInfo != NULL);

  // This is the class definition (the fileInfo is the position of the opening brace)
  SgClassDefinition* classDefinition   = new SgClassDefinition(fileInfo);
  assert(classDefinition != NULL);

  // Set the end of construct explictly (where not a transformation this is the location of the closing brace)
  classDefinition->set_endOfConstruct(fileInfo);

  // This is the defining declaration for the class (with a reference to the class definition)
  SgClassDeclaration* classDeclaration 
    = new SgClassDeclaration(fileInfo,name.c_str(),SgClassDeclaration::e_struct,NULL,classDefinition);
  assert(classDeclaration != NULL);

  // Set the defining declaration in the defining declaration!
  classDeclaration->set_definingDeclaration(classDeclaration);

  // Set the non defining declaration in the defining declaration (both are required)
  SgClassDeclaration* nondefiningClassDeclaration 
    = new SgClassDeclaration(fileInfo,name.c_str(),SgClassDeclaration::e_struct,NULL,NULL);
  assert(classDeclaration != NULL);
  nondefiningClassDeclaration->set_type(SgClassType::createType(nondefiningClassDeclaration));

  // Set the internal reference to the non-defining declaration
  classDeclaration->set_firstNondefiningDeclaration(nondefiningClassDeclaration);
  classDeclaration->set_type(nondefiningClassDeclaration->get_type());

  // Set the defining and no-defining declarations in the non-defining class declaration!
  nondefiningClassDeclaration->set_firstNondefiningDeclaration(nondefiningClassDeclaration);
  nondefiningClassDeclaration->set_definingDeclaration(classDeclaration);

  // Set the nondefining declaration as a forward declaration!
  nondefiningClassDeclaration->setForward();

  // Don't forget the set the declaration in the definition (IR node constructors are side-effect free!)!
  classDefinition->set_declaration(classDeclaration);

  // set the scope explicitly (name qualification tricks can imply it is not always the parent IR node!)
  classDeclaration->set_scope(scope);
  nondefiningClassDeclaration->set_scope(scope);

  // some error checking
  assert(classDeclaration->get_definingDeclaration() != NULL);
  assert(classDeclaration->get_firstNondefiningDeclaration() != NULL);
  assert(classDeclaration->get_definition() != NULL);

  // DQ (9/8/2007): Need to add function symbol to global scope!
  //     printf ("Fixing up the symbol table in scope = %p = %s for class = %p = %s \n",scope,scope->class_name().c_str(),classDeclaration,classDeclaration->get_name().str());
  SgClassSymbol* classSymbol = new SgClassSymbol(classDeclaration);
  scope->insert_symbol(classDeclaration->get_name(),classSymbol);
  ROSE_ASSERT(scope->lookup_class_symbol(classDeclaration->get_name()) != NULL);

  return classDeclaration;
}



SgClassDeclaration* buildStructDeclaration ( 
    SgScopeStatement* scope,
    const std::vector<SgType*>& memberTypes, 
    const std::vector<std::string>& memberNames,
    const std::string& structName
)
{
  ROSE_ASSERT(memberTypes.size() == memberNames.size());
  SgClassDeclaration* classDeclaration = buildClassDeclarationAndDefinition(structName,scope);
  std::vector<SgType*>::const_iterator typeIterator            = memberTypes.begin();
  std::vector<std::string>::const_iterator  memberNameIterator = memberNames.begin();
  while (typeIterator != memberTypes.end())
  {
    // printf ("Adding data member type = %s variable name = %s \n",(*typeIterator)->unparseToString().c_str(),memberNameIterator->c_str());
    SgVariableDeclaration* memberDeclaration = new SgVariableDeclaration(SOURCE_POSITION,
      *memberNameIterator,
      *typeIterator,NULL
    );
    memberDeclaration->set_endOfConstruct(SOURCE_POSITION);
    
    classDeclaration->get_definition()->append_member(memberDeclaration);
    
    memberDeclaration->set_parent(classDeclaration->get_definition());
    // Liao (2/13/2008) scope and symbols for member variables
    SgInitializedName* initializedName = *(memberDeclaration->get_variables().begin());
    initializedName->set_scope(classDeclaration->get_definition());
    
    // set nondefning declaration pointer
    memberDeclaration->set_definingDeclaration(memberDeclaration);
//    memberDeclaration->set_firstNondefiningDeclaration(memberDeclaration);
    
    SgVariableSymbol* variableSymbol = new SgVariableSymbol(initializedName);
    classDeclaration->get_definition()->insert_symbol(*memberNameIterator,variableSymbol);
    
    typeIterator++;
    memberNameIterator++;
  }
  
  return classDeclaration;
}



/**
 */
SgVariableDeclaration* buildStructVariable ( 
    SgScopeStatement* scope,
    const std::vector<SgType*>& memberTypes, 
    const std::vector<std::string>& memberNames,
    const std::string& structName, 
    const std::string& varName, 
    SgInitializer *initializer 
)
{
  ROSE_ASSERT(memberTypes.size() == memberNames.size());
  SgClassDeclaration* classDeclaration = buildClassDeclarationAndDefinition(structName,scope);
  std::vector<SgType*>::const_iterator typeIterator       = memberTypes.begin();
  std::vector<std::string>::const_iterator  memberNameIterator = memberNames.begin();
  while (typeIterator != memberTypes.end())
  {
    // printf ("Adding data member type = %s variable name = %s \n",(*typeIterator)->unparseToString().c_str(),memberNameIterator->c_str());
    SgVariableDeclaration* memberDeclaration = new SgVariableDeclaration(SOURCE_POSITION,*memberNameIterator,*typeIterator,NULL);
    memberDeclaration->set_endOfConstruct(SOURCE_POSITION);
    
    classDeclaration->get_definition()->append_member(memberDeclaration);
    
    memberDeclaration->set_parent(classDeclaration->get_definition());
    // Liao (2/13/2008) scope and symbols for member variables
    SgInitializedName* initializedName = *(memberDeclaration->get_variables().begin());
    initializedName->set_scope(classDeclaration->get_definition());
    
    // set nondefning declaration pointer
    memberDeclaration->set_firstNondefiningDeclaration(memberDeclaration);
    
    SgVariableSymbol* variableSymbol = new SgVariableSymbol(initializedName);
    classDeclaration->get_definition()->insert_symbol(*memberNameIterator,variableSymbol);
    
    typeIterator++;
    memberNameIterator++;
  }
  
  SgClassType* classType = new SgClassType(classDeclaration->get_firstNondefiningDeclaration());
  SgVariableDeclaration* variableDeclaration = new SgVariableDeclaration(SOURCE_POSITION,varName,classType,initializer);
  variableDeclaration->set_endOfConstruct(SOURCE_POSITION);
  
  //Liao (2/13/2008) scope and symbols for struct variable
  SgInitializedName* initializedName = *(variableDeclaration->get_variables().begin());
  initializedName->set_scope(scope);
  
  SgVariableSymbol* variableSymbol = new SgVariableSymbol(initializedName);
  scope->insert_symbol(varName,variableSymbol);
  
  //set nondefining declaration 
  variableDeclaration->set_firstNondefiningDeclaration(variableDeclaration);
  
  // This is required, since it is not set in the SgVariableDeclaration constructor
  if (initializer !=0)
    initializer->set_parent(variableDeclaration);
  
  variableDeclaration->set_variableDeclarationContainsBaseTypeDefiningDeclaration(true);
  variableDeclaration->set_baseTypeDefiningDeclaration(classDeclaration->get_definingDeclaration());
  
  classDeclaration->set_parent(variableDeclaration);
  
  return variableDeclaration;
}



/*
*/
SgVariableDeclaration* buildStructVariable ( 
      SgScopeStatement* scope,
      SgClassDeclaration* classDeclaration,
      const std::string& varName, 
      SgInitializer *initializer 
)
{
  SgClassType* classType = new SgClassType(classDeclaration->get_firstNondefiningDeclaration());
  SgVariableDeclaration* variableDeclaration = new SgVariableDeclaration(SOURCE_POSITION,varName,classType,initializer);
  variableDeclaration->set_endOfConstruct(SOURCE_POSITION);
  
  //Liao (2/13/2008) scope and symbols for struct variable
  SgInitializedName* initializedName = *(variableDeclaration->get_variables().begin());
  initializedName->set_scope(scope);
  
  SgVariableSymbol* variableSymbol = new SgVariableSymbol(initializedName);
  scope->insert_symbol(varName,variableSymbol);
  
  //set nondefining declaration 
  variableDeclaration->set_firstNondefiningDeclaration(variableDeclaration);
  
  // This is required, since it is not set in the SgVariableDeclaration constructor
  if (initializer !=0)
    initializer->set_parent(variableDeclaration);
  
  variableDeclaration->set_variableDeclarationContainsBaseTypeDefiningDeclaration(true);
  variableDeclaration->set_baseTypeDefiningDeclaration(classDeclaration->get_definingDeclaration());
  
  classDeclaration->set_parent(variableDeclaration);
  
  return variableDeclaration;
}


/*
*/
SgVariableDeclaration* buildStructPointerVariable ( 
      SgScopeStatement* scope,
      SgClassDeclaration* classDeclaration,
      const std::string& varName, 
      SgInitializer *initializer 
)
{
  SgClassType* classType = new SgClassType(classDeclaration->get_firstNondefiningDeclaration());
  SgVariableDeclaration* variableDeclaration = new SgVariableDeclaration(
    SOURCE_POSITION,
    varName,
    SageBuilder::buildPointerType( classType ),
    initializer
  );
  variableDeclaration->set_endOfConstruct(SOURCE_POSITION);
  
  //Liao (2/13/2008) scope and symbols for struct variable
  SgInitializedName* initializedName = *(variableDeclaration->get_variables().begin());
  initializedName->set_scope(scope);
  
  SgVariableSymbol* variableSymbol = new SgVariableSymbol(initializedName);
  scope->insert_symbol(varName,variableSymbol);
  
  //set nondefining declaration 
  variableDeclaration->set_firstNondefiningDeclaration(variableDeclaration);
  
  // This is required, since it is not set in the SgVariableDeclaration constructor
  if (initializer !=0)
    initializer->set_parent(variableDeclaration);
  
  variableDeclaration->set_variableDeclarationContainsBaseTypeDefiningDeclaration(true);
  variableDeclaration->set_baseTypeDefiningDeclaration(classDeclaration->get_definingDeclaration());
  
  classDeclaration->set_parent(variableDeclaration);
  
  return variableDeclaration;
}


/***/
SgVariableDeclaration* buildInsertDeclarationKaapiThread( SgScopeStatement* scope, SgScopeStatement* append )
{
  if (append ==0) append = scope;
#if 0
  /* Change scope to generate definition outside a loop */
  SgScopeStatement* loop = SageInterface::findEnclosingLoop( scope );
  while (loop != 0)
  {
    scope = loop;
    loop = SageInterface::findEnclosingLoop( scope );
  }
#endif
  SgVariableDeclaration* newkaapi_thread = SageBuilder::buildVariableDeclaration ( 
    "__kaapi_thread", 
    SageBuilder::buildPointerType( kaapi_thread_ROSE_type ), 
    SageBuilder::buildAssignInitializer(
      SageBuilder::buildFunctionCallExp(
        "kaapi_self_thread",
        SageBuilder::buildPointerType(kaapi_thread_ROSE_type),
        SageBuilder::buildExprListExp(),
        scope
      ),
      0
    ),
    append
  );
  SageInterface::prependStatement(newkaapi_thread, scope);
  return newkaapi_thread;
}

SgVariableDeclaration* buildThreadVariableDecl( SgScopeStatement* scope )
{
  SgVariableDeclaration* const decl =
    SageBuilder::buildVariableDeclaration
    ( 
     "__kaapi_thread",
     SageBuilder::buildPointerType( kaapi_thread_ROSE_type )
    );
  SageInterface::prependStatement(decl, scope);
  return decl;
}

void buildVariableAssignment (
   SgExprStatement* cur_stmt, // current statement
   SgVariableDeclaration* var_decl, // variable decl
   SgScopeStatement* scope // needed
)
{
  SgFunctionCallExp* const call_expr =
    SageBuilder::buildFunctionCallExp
    (
     "kaapi_self_thread",
     SageBuilder::buildPointerType(kaapi_thread_ROSE_type),
     SageBuilder::buildExprListExp(),
     scope
    );

  SgVarRefExp* const var_expr =
    SageBuilder::buildVarRefExp(var_decl);

  SgExprStatement* const assign_stmt =
    SageBuilder::buildAssignStatement
    (var_expr, call_expr);

  SageInterface::insertStatement(cur_stmt, assign_stmt);
}

/** Generate the structure environnement to outline statement
*/
SgClassDeclaration* buildOutlineArgStruct(
  const std::set<SgVariableSymbol*>& listvar,
  SgGlobal*                          scope,
  SgClassDefinition*		     this_class_def,
  const char*			     name_prefix,
  bool				     do_pointers
)
{
  static int cnt_sa = 0;
  if (name_prefix == NULL) name_prefix = "__kaapi_task_arg_";
  std::ostringstream context_name;
  context_name << name_prefix << cnt_sa++;
  SgClassDeclaration* contexttype =  buildClassDeclarationAndDefinition( 
      context_name.str(),
      scope
  );

  /* Build the structure for the argument of most of the other function:
     The data structure contains
     - all free parameters required to call the entrypoint for loop execution
     The structure store pointer to the actual parameters.
     - a p_this pointer, if it is a method
  */
  std::set<SgVariableSymbol*>::iterator ivar_beg = listvar.begin();
  std::set<SgVariableSymbol*>::iterator ivar_end = listvar.end();
  while (ivar_beg != ivar_end)
  {
    SgVariableDeclaration* memberDeclaration;
    if (do_pointers == true)
    {
      /* add p_<name> */
      memberDeclaration = SageBuilder::buildVariableDeclaration
	(
	 "p_" + (*ivar_beg)->get_name(),
	 SageBuilder::buildPointerType((*ivar_beg)->get_type()),
	 0, 
	 contexttype->get_definition()
	);
    }
    else
    {
      /* add p_<name> */
      memberDeclaration = SageBuilder::buildVariableDeclaration
	(
	 (*ivar_beg)->get_name(),
	 (*ivar_beg)->get_type(),
	 0, 
	 contexttype->get_definition()
	);
    }
    memberDeclaration->set_endOfConstruct(SOURCE_POSITION);
    contexttype->get_definition()->append_member(memberDeclaration);

    ++ivar_beg;
  }

  // add p_this
  if (this_class_def != NULL)
  {
    SgVariableDeclaration* memberDeclaration;
    if (do_pointers == true)
    {
      memberDeclaration = SageBuilder::buildVariableDeclaration
	(
	 "p_this",
	 SageBuilder::buildPointerType
	 (this_class_def->get_declaration()->get_type()),
	 0,
	 contexttype->get_definition()
	);
    }
    else
    {
      memberDeclaration = SageBuilder::buildVariableDeclaration
	(
	 "this",
	 this_class_def->get_declaration()->get_type(),
	 0,
	 contexttype->get_definition()
	);
    }
    memberDeclaration->set_endOfConstruct(SOURCE_POSITION);
    contexttype->get_definition()->append_member(memberDeclaration);
  }

  contexttype->set_endOfConstruct(SOURCE_POSITION);

  return contexttype;
}
