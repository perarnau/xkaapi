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


#include <iostream>
#include "rose_headers.h"
#include "kaapi_taskcall.h"
#include "kaapi_abort.h"
#include "utils.h"
#include "globals.h"


/** Return True if it is a valid expression with a function call to a task
    statement should be the enclosing statement of fc.
*/
static bool isValidTaskWithRetValCallExpression(SgFunctionCallExp* fc, SgStatement* statement)
{
  /* ****** Verification of the good use of this case
      - retval is ignored during the call:  fu(<args>)
      - retval is used in expression:  res = fu(<args>) or fu(<args>) + fi(<orgs>)
      
      We only accept = 
        - fu(<args>)
        - <expression> = fu(<args>) 
      
      Assuming sg == oc->statement of the correct SgNode' type following each case.
      All the following variantT values correspond to subclass of SgStatement
  */
  switch (statement->variantT()) 
  {
    case V_SgExprStatement:  /* in sg->get_expression() where may reside a function call to a task */
                           /* in this form, if sg->get_expression() == SgAssignOp, the lhs = sg->get_lhs_operand () */
    {
      /* only accepted form:
          <expression> = <func call>; 
      */
      SgExpression* expr = isSgExprStatement(statement)->get_expression();

      if (isSgFunctionCallExp(expr)) return true;

      if (isSgAssignOp(expr) == 0) break;

      SgAssignOp* expr_assign = isSgAssignOp( expr );
      if (isSgFunctionCallExp(expr_assign->get_rhs_operand()) != fc)
        break;
      
      return true;
    }

/* A vérifier 
*/
    case V_SgAssignStatement:   /* var = expr, var == sg->get_label() and expr == sg->get_value() */
      return true;

/* NOT Accepted */
    case V_SgVariableDefinition:/* in sg->get_vardefn()->get_initializer() (an SgExpression*) */
    case V_SgReturnStmt:	/* return expr , expr == sg->get_expression() */
    case V_SgForInitStatement:  /* may be in the SgStatement of the initialization list (get_init_stmt)
                                   but in that case, EnclosingStatement should have returned one of the concrete
                                   statement
                                 */
    default:
      break;
  } /* switch */

  Sg_File_Info* fileInfo = fc->get_file_info();
  std::cerr << "****[kaapi_c2c] Function call to task is not in a canonical expression or statement."
            << "     In filename '" << fileInfo->get_filename() 
            << "' LINE: " << fileInfo->get_line()
            << std::endl;
  KaapiAbort("**** error");
  
  return false;
}


void KaapiTaskCallTraversal::visit(SgNode* node)
{
  /* propagate the parallelregion attribut to all nodes which has parent within */
  if ((node->get_parent() !=0) && (node->get_parent()->getAttribute("kaapiisparallelregion") !=0))
  {
    node->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
  }

    
  /* */
  SgFunctionCallExp* fc = isSgFunctionCallExp(node);
  if (fc)
  {
    SgStatement* exprstatement = SageInterface::getEnclosingStatement( fc );
    if (exprstatement == NULL) return ;

#if CONFIG_ENABLE_DEBUG
    Sg_File_Info* fileInfo = node->get_file_info();
    std::cerr << "****[kaapi_c2c] Message: found function call expression.\n"
	      << "     In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
	      << std::endl;
#endif // CONFIG_ENABLE_DEBUG

#if 1
    if (exprstatement->getAttribute("kaapinotask") !=0) 
      return;
#endif
    if (exprstatement->getAttribute("kaapiwrappercall") !=0) 
      return; /* call made by the wrapper do not replace it by task creation */
        
    if (fc !=0)
    {
      SgFunctionDeclaration* fdecl = fc->getAssociatedFunctionDeclaration();
#if CONFIG_ENABLE_DEBUG
      std::cerr << "****[kaapi_c2c] Message: found function call expression. fdecl: @" << fdecl
		<< std::endl;
#endif // CONFIG_ENABLE_DEBUG
      if (fdecl !=0)
      {
	SgSymbol* symb = fdecl->search_for_symbol_from_symbol_table();
#if CONFIG_ENABLE_DEBUG
	std::cerr << "****[kaapi_c2c] Message: found function call expression. fdecl: @" << fdecl << " symb: @" << symb
		  << std::endl;
#endif // CONFIG_ENABLE_DEBUG
	KaapiTaskAttribute* kta = (KaapiTaskAttribute*)symb->getAttribute("kaapitask");
	if (kta && kta->is_signature == true) kta = 0;
	if (kta !=0)
	{
#if CONFIG_ENABLE_DEBUG
	  std::cerr << "****[kaapi_c2c] Message: found function call expression. fdecl: @" << fdecl << " symb: @" << symb
		    << std::endl;
#endif // CONFIG_ENABLE_DEBUG
	  /* store both the container (basic block) and the funccall expr */

	  // some scope dont support prepending statements, ie.
	  // for (...) fu(); when no accolades are added. we thus
	  // move up until we found a sgBasicBlock
	  SgScopeStatement* scope = SageInterface::getScope( fc );
	  if (isSgBasicBlock(scope) == NULL)
	  {
	    SgNode* node = isSgNode(scope);
	    while (node && isSgBasicBlock(node) == NULL)
	      node = node->get_parent();
	    if (node == NULL) KaapiAbort("scope not found");
	    scope = (SgScopeStatement*)isSgBasicBlock(node);
	  }

	  SgScopeStatement* loop = SageInterface::findEnclosingLoop( exprstatement );

	  if (!kta->has_retval || isValidTaskWithRetValCallExpression(fc, exprstatement ))
	  {
	    _listcall.push_back( OneCall(scope, fc, exprstatement, kta, loop) );
	  }

#if CONFIG_ENABLE_DEBUG
	  if (loop !=0)
	    std::cout << "Find enclosing loop of a task declaration:" << loop->class_name() << std::endl
		      << "At line: " << loop->get_file_info()->get_line()
		      << std::endl;
#endif // CONFIG_ENABLE_DEBUG

#if 0 // TG no important here: see below in the main: when TemplateInstance are processed
	  { SgTemplateInstantiationFunctionDecl* sg_tmpldecl = isSgTemplateInstantiationFunctionDecl( fdecl );
	    if (sg_tmpldecl !=0)
	      {
#if CONFIG_ENABLE_DEBUG
		std::cerr << "This is a call to a template function instanciation\n" << std::endl;
#endif // CONFIG_ENABLE_DEBUG
		KaapiPragmaString* kps 
		  = (KaapiPragmaString*)sg_tmpldecl->get_templateDeclaration()->getAttribute("kaapi_templatetask");
		if (kps != 0)
		  {
#if CONFIG_ENABLE_DEBUG
		    std::cerr << "This is a call to a TASK template function instanciation, definition=" 
			      << sg_tmpldecl->get_definition() << "\n" << std::endl;
#endif // CONFIG_ENABLE_DEBUG
		    if (sg_tmpldecl->get_definition() !=0)
		      all_template_instanciate_definition.insert(sg_tmpldecl->get_definition());
		  }
	      }
	  }
#endif
	} // fdecl !=0 & kta != 0
	else {
#if CONFIG_ENABLE_DEBUG
	  Sg_File_Info* fileInfo = node->get_file_info();
	  std::cerr << "****[kaapi_c2c] Message: function call expression to " << fdecl->get_name().str() 
		    << " with empty task declaration. "
		    << " Function declaration: @" << fdecl
		    << " Function Symbol: @" << fdecl->search_for_symbol_from_symbol_table()
		    << ". Ignored.\n"
		    << "     In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
		    << std::endl;
#endif // CONFIG_ENABLE_DEBUG
	}
      }
      else {
#if CONFIG_ENABLE_DEBUG
	Sg_File_Info* fileInfo = node->get_file_info();
	std::cerr << "****[kaapi_c2c] Warning: function call expression with empty declaration is ignored.\n"
		  << "     In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
		  << std::endl;
#endif // CONFIG_ENABLE_DEBUG
      }
    }
  }
}


// replace function calls by kaapi task spawns

/** Change functioncall to task creation 
    OneCall represents the AST node where the function call to a task was detected.
*/
void buildFunCall2TaskSpawn( OneCall* oc )
{
  static int cnt = 0;
  std::ostringstream arg_name;
  arg_name << "__kaapi_arg_" << cnt++;
  SgClassType* classtype =new SgClassType(oc->kta->paramclass->get_firstNondefiningDeclaration());

  /* look for __kaapi_thread in the scope or in parent scope */
  SgVariableSymbol* newkaapi_threadvar = 
    SageInterface::lookupVariableSymbolInParentScopes(
        "__kaapi_thread", 
        oc->scope 
  );
  if (newkaapi_threadvar ==0)
  {
    buildInsertDeclarationKaapiThread( oc->scope );
  }

  /* */
  SgVariableDeclaration* variableDeclaration =
    SageBuilder::buildVariableDeclaration (
      arg_name.str(), 
      SageBuilder::buildPointerType(classtype), 
      SageBuilder::buildAssignInitializer(
        SageBuilder::buildCastExp(
          SageBuilder::buildFunctionCallExp(
            "kaapi_thread_pushdata",
            SageBuilder::buildPointerType(SageBuilder::buildVoidType()),
            SageBuilder::buildExprListExp(
              SageBuilder::buildVarRefExp ("__kaapi_thread", oc->scope ),
              SageBuilder::buildSizeOfOp( classtype )
            ),
            oc->scope
          ),
          SageBuilder::buildPointerType(classtype)
        )
      ),
      oc->scope
    );

  SageInterface::insertStatement(oc->statement, variableDeclaration, false);
  SageInterface::removeStatement(oc->statement);
  SgStatement* last_statement = variableDeclaration;
  SgStatement* assign_statement;

  SgExpressionPtrList& listexpr = oc->fc->get_args()->get_expressions();
  SgExpressionPtrList::iterator iebeg;
  int i = 0;
  
  /* Generate initialization to the this field in case of method call
  */
  if (oc->kta->has_this) /* means ->  get_function is 'a.f' or 'a->f' to the class object 'a' */
  {
    SgExpression* f = oc->fc->get_function();
    SgExpression* object = 0;
    if (isSgArrowExp(f))
    {
      object= isSgArrowExp(f)->get_lhs_operand();
    }
    else if (isSgDotExp(f))
    {
      object= SageBuilder::buildAddressOfOp( isSgDotExp(f)->get_lhs_operand() );
    } 
    else 
    {
      KaapiAbort( "*** Internal compiler error: mal formed AST" );
    }
    
    std::ostringstream fieldname;
    fieldname << arg_name.str() << "->thisfield.data";
    assign_statement = SageBuilder::buildExprStatement(
      SageBuilder::buildAssignOp(
        /* dummy-> */
        SageBuilder::buildOpaqueVarRefExp (fieldname.str(),oc->scope),
        /* expr */
        SageBuilder::buildCastExp(
          object,
          SageBuilder::buildPointerType(
            SageBuilder::buildVoidType()
          )
        )
      )
    );
    SageInterface::insertStatement(last_statement, assign_statement, false);
    last_statement = assign_statement;
  }

  /* Generate initialization to each of the formal parameter except those that
     where defined as end of interval, which are initialized after.
  */
  for (iebeg = listexpr.begin(); iebeg != listexpr.end(); ++iebeg, ++i)
  {
    /* generate end of interval initialization after all previous field init */
    if (oc->kta->israngedecl[i] == 2) 
      continue;

    std::ostringstream fieldname;
    
    if (oc->kta->formal_param[i].mode == KAAPI_V_MODE)
    {
      fieldname << arg_name.str() << "->f" << i;
      assign_statement = SageBuilder::buildExprStatement(
        SageBuilder::buildAssignOp(
          /* dummy-> */
          SageBuilder::buildOpaqueVarRefExp (fieldname.str(),oc->scope),
          /* expr */
            *iebeg
        )
      );
    }
    else 
    {
      fieldname << arg_name.str() << "->f" << i << ".data";
      assign_statement = SageBuilder::buildExprStatement(
        SageBuilder::buildAssignOp(
          /* dummy-> */
          SageBuilder::buildOpaqueVarRefExp (fieldname.str(),oc->scope),
          /* expr */
          SageBuilder::buildCastExp(
            *iebeg,
            SageBuilder::buildPointerType(
              SageBuilder::buildVoidType()
            )
          )
        )
      );
    }
    SageInterface::insertStatement(last_statement, assign_statement, false);
    last_statement = assign_statement;
  }

#if 0 // handle return value
  if (oc->kta->has_retval)
  {
    printf("handling return value\n");
    // TODO: retrieve the lhs part...
    SgExpression* lhs_node = 0;
    switch (oc->statement->variantT()) 
    {
      case V_SgAssignStatement:
        lhs_node = isSgAssignStatement(oc->statement)->get_label();
        break;
      case V_SgExprStatement:
      {
        SgExpression* expr = isSgExprStatement(oc->statement)->get_expression();
        SgAssignOp* expr_assign = isSgAssignOp( expr );
        lhs_node = expr_assign->get_lhs_operand();
      } break; 
      default:
        KaapiAbort("**** Internal error"); /* isValid... was not call ? */
    }

    // assign arg->f[last].data = &lhs;
    SgStatement* assign_stmt;
    std::ostringstream fieldname;
    fieldname << arg_name.str() << "->r.data";
    assign_statement = SageBuilder::buildExprStatement(
      SageBuilder::buildAssignOp (
        SageBuilder::buildOpaqueVarRefExp(fieldname.str(), oc->scope),
        SageBuilder::buildCastExp(
          lhs_node,
          SageBuilder::buildPointerType(SageBuilder::buildVoidType())
        )
      )
    );
    SageInterface::insertStatement(last_statement, assign_stmt, false);
    last_statement = assign_stmt;
  }
#else
  if (oc->kta->has_retval)
  {
    SgExpression* const expr =
      isSgExprStatement(oc->statement)->get_expression();
    SgAssignOp* const assign_op = isSgAssignOp(expr);
    SgExpression* lhs_ref;
    if (assign_op)
    {
      SgExpression* const lhs_expr = assign_op->get_lhs_operand();
      lhs_ref = SageBuilder::buildAddressOfOp(lhs_expr);
    } // assign_op
    else
    {
      // build NULL expression
      std::ostringstream zero("0");
      SgExpression* const zero_expr =
        SageBuilder::buildOpaqueVarRefExp(zero.str(), oc->scope);
      lhs_ref = SageBuilder::buildCastExp
        (zero_expr, SageBuilder::buildPointerType(SageBuilder::buildVoidType()));
    }

    SgStatement* assign_stmt;
    std::ostringstream fieldname;
    fieldname << arg_name.str() << "->r.data";
    assign_stmt = SageBuilder::buildExprStatement
    (
     SageBuilder::buildAssignOp
     (
      SageBuilder::buildOpaqueVarRefExp(fieldname.str(), oc->scope),
      lhs_ref
     )
    );
    SageInterface::insertStatement(last_statement, assign_stmt, false);
    last_statement = assign_stmt;
  }
#endif // handle return value

  /* generate initialization of end of range */
  i = 0;
  for (iebeg = listexpr.begin(); iebeg != listexpr.end(); ++iebeg, ++i)
  {
    if (oc->kta->israngedecl[i] <= 1) continue;

    SgStatement* assign_statement;
    std::ostringstream fieldname;
    fieldname << arg_name.str() << "->f" << i;

    std::ostringstream fieldname_firstbound;
    fieldname_firstbound << "f" << oc->kta->formal_param[i].attr->index_firstbound << ".data";
    
    assign_statement = SageBuilder::buildExprStatement(
      SageBuilder::buildAssignOp(
        /* dummy-> */
        SageBuilder::buildOpaqueVarRefExp (fieldname.str(),oc->scope),
        /* expr = *iebeg - (cast)arg_name->f_index_firstbound */
        SageBuilder::buildSubtractOp( 
          *iebeg,
          SageBuilder::buildCastExp(
            SageBuilder::buildArrowExp(
              SageBuilder::buildVarRefExp(arg_name.str(),oc->scope),
              SageBuilder::buildOpaqueVarRefExp( fieldname_firstbound.str(), oc->scope )
            ),
            oc->kta->formal_param[i].type
          )
        )
      )
    );
    SageInterface::insertStatement(last_statement, assign_statement, false);
    last_statement = assign_statement;
  }

  /* generate copy of global parameter */
  i = 0;
  std::vector<KaapiTaskFormalParam>::iterator ibeg;
  for (ibeg = oc->kta->extra_param.begin(); ibeg != oc->kta->extra_param.end(); ++ibeg, ++i)
  {
    SgStatement* assign_statement;
    std::ostringstream fieldname;
    fieldname << arg_name.str() << "->e" << i;

    assign_statement = SageBuilder::buildExprStatement(
      SageBuilder::buildFunctionCallExp(
        "memcpy",
        SageBuilder::buildPointerType(SageBuilder::buildVoidType()),
        SageBuilder::buildExprListExp(
          SageBuilder::buildAddressOfOp(
            SageBuilder::buildOpaqueVarRefExp (fieldname.str(),oc->scope)
          ),
          SageBuilder::buildAddressOfOp(
            SageBuilder::buildVarRefExp(ibeg->initname,oc->scope)
          ),
          SageBuilder::buildSizeOfOp(
            SageBuilder::buildVarRefExp(ibeg->initname,oc->scope)
          )
        ),
        oc->scope
      )
    );
    SageInterface::insertStatement(last_statement, assign_statement, false);
    last_statement = assign_statement;
  }

  
  static int task_cnt = 0;
  std::ostringstream task_name;
  task_name << "__kaapi_task_" << task_cnt++;
  
  
  SgVariableDeclaration* taskDeclaration =
    SageBuilder::buildVariableDeclaration (
      task_name.str(), 
      SageBuilder::buildPointerType(kaapi_task_ROSE_type), 
      SageBuilder::buildAssignInitializer(
        SageBuilder::buildFunctionCallExp(
          "kaapi_thread_toptask",
          SageBuilder::buildPointerType(kaapi_task_ROSE_type),
          SageBuilder::buildExprListExp(
            SageBuilder::buildVarRefExp ("__kaapi_thread", oc->scope )
          ),
          oc->scope
        )
      ),
      oc->scope
    );
  
  SageInterface::insertStatement(last_statement, taskDeclaration, false);
  last_statement = taskDeclaration;

  SgStatement* init_task_statement = SageBuilder::buildExprStatement(
    SageBuilder::buildFunctionCallExp(
      "kaapi_task_init",
      SageBuilder::buildVoidType(),
      SageBuilder::buildExprListExp(
        SageBuilder::buildVarRefExp (task_name.str(), oc->scope ),
        SageBuilder::buildFunctionRefExp (oc->kta->wrapper_decl),// oc->scope ),
        SageBuilder::buildVarRefExp (arg_name.str(), oc->scope )
      ),
      oc->scope
    )
  );
  SageInterface::insertStatement(last_statement, init_task_statement, false);
  last_statement = init_task_statement;

  SgStatement* push_task_statement = SageBuilder::buildExprStatement(
    SageBuilder::buildFunctionCallExp(
      "kaapi_thread_pushtask",
      SageBuilder::buildVoidType(),
      SageBuilder::buildExprListExp(
        SageBuilder::buildVarRefExp ("__kaapi_thread", oc->scope )
      ),
      oc->scope
    )
  );
  SageInterface::insertStatement(last_statement, push_task_statement, false);
  last_statement = push_task_statement;
}


/** Add save/restore frame for SgScopeStatement forloop kind.
    Such forloop are:
          case V_SgDoWhileStmt: 
          case V_SgWhileStmt:
          case V_SgForStatement:
*/
void buildInsertSaveRestoreFrame( SgScopeStatement* forloop )
{
  if (forloop ==0) return;
  if (forloop->getAttribute("kaapifor_ok") !=0) return;
  forloop->setAttribute("kaapifor_ok", (AstAttribute*)-1);
  
  switch (forloop->variantT()) /* see ROSE documentation if missing node or node */
  {
    case V_SgDoWhileStmt: 
    case V_SgWhileStmt:
    case V_SgForStatement:
      break;
    default: 
      return;
  }


  SgScopeStatement* scope = SageInterface::getScope( forloop->get_parent() );
  
  SgVariableSymbol* newkaapi_threadvar = 
    SageInterface::lookupVariableSymbolInParentScopes(
        "__kaapi_thread", 
        scope 
  );
  if (newkaapi_threadvar ==0)
    buildInsertDeclarationKaapiThread( scope );

#if 0        
  static int cnt = 0;
  std::ostringstream arg_name;
  arg_name << "__kaapi_frame_" << cnt++;
  SgVariableDeclaration* variableDeclaration =
    SageBuilder::buildVariableDeclaration (
      arg_name.str(), 
      kaapi_frame_ROSE_type,
      0,
      scope
    );
  
  /* insert before */
  SageInterface::insertStatement(forloop, variableDeclaration, true);
#endif

  SgStatement* saveframe_statement = SageBuilder::buildExprStatement(
    SageBuilder::buildAssignOp(
      SageBuilder::buildVarRefExp ("__kaapi_thread", scope ),
      SageBuilder::buildFunctionCallExp(
        "kaapi_thread_push_frame",
        SageBuilder::buildIntType(),
        SageBuilder::buildExprListExp(
        ),
        scope
      )
    )
  );
#if 0
  /* insert after declaration */
  SageInterface::insertStatement(variableDeclaration, saveframe_statement, false);
#else
  /* insert before statement */
  SageInterface::insertStatement(forloop, saveframe_statement, true);
#endif

  {
    StaticCFG::CFG cfg( forloop );
    cfg.buildCFG();
    //->get_loop_body() ); //SageInterface::getScope(forloop) ); 
    //SageInterface::getEnclosingFunctionDefinition(forloop)); // /*forloop*/);
    SgGraphNode* nodeend = cfg.cfgForEnd( forloop );

    /* find the break statements: all instructions that go in the exit node */
    std::vector< SgDirectedGraphEdge * > in = cfg.getInEdges(nodeend);
    for (size_t i=0; i<in.size(); ++i)
    {
      /* from node in the edge is the output to the loop to the loop statement */
      SgStatement* node = isSgStatement(in[i]->get_from()->get_SgNode());
        std::cout << "1. Find exit loop statement, from:" << in[i]->get_from()->get_SgNode()->class_name() 
                  << " at line: " << in[i]->get_from()->get_SgNode()->get_file_info()->get_line()
                  << " to:" << in[i]->get_to()->get_SgNode()->class_name() 
                  << " at line: " << in[i]->get_to()->get_SgNode()->get_file_info()->get_line()
                  << std::endl;
                  
      
      /* add a kaapi_restore_frame */
      switch (node->variantT()) 
      {
        case V_SgGotoStatement:
        case V_SgReturnStmt:
        {
          SgStatement* syncframe_statement = SageBuilder::buildExprStatement(
            SageBuilder::buildFunctionCallExp(
              "kaapi_sched_sync",
              SageBuilder::buildIntType(),
              SageBuilder::buildExprListExp(),
              scope
            )
          );
          /* insert before */
          SageInterface::insertStatement(node, syncframe_statement, true);

          SgStatement* saveframe_statement = SageBuilder::buildExprStatement(
            SageBuilder::buildAssignOp(
              SageBuilder::buildVarRefExp ("__kaapi_thread", scope ),
              SageBuilder::buildFunctionCallExp(
                "kaapi_thread_pop_frame",
                SageBuilder::buildIntType(),
                SageBuilder::buildExprListExp(
                ),
                scope
              )
            )
          );
          /* insert after */
          SageInterface::insertStatement(syncframe_statement, saveframe_statement, false);
        } break;
        
        case V_SgBreakStmt: /* normal control flow break: will execute the epilogue after the loop */
        default:  
          break;
      }
    }
    
    SgGraphNode* nodebeg = cfg.getExit();
    /* find instruction that go out the current function */
    std::vector< SgDirectedGraphEdge * > out = cfg.getInEdges(nodebeg); //cfg.getExit()); 
    //cfgForEnd(isSgForStatement(forloop)->get_loop_body()));
    for (size_t i=0; i<out.size(); ++i)
    {
      /* from node in the edge is the output to the loop to the loop statement */
      std::cout << "2. Find exit loop statement, from:" << out[i]->get_from()->get_SgNode()->class_name() 
		<< " at line: " << out[i]->get_from()->get_SgNode()->get_file_info()->get_line()
		<< " to:" << out[i]->get_to()->get_SgNode()->class_name() 
		<< " at line: " << out[i]->get_to()->get_SgNode()->get_file_info()->get_line()
		<< std::endl;
    }
      
    /* insert kaapi_sched_sync + kaapi_restore_frame after the loop */
    SgStatement* syncframe_statement = SageBuilder::buildExprStatement(
      SageBuilder::buildFunctionCallExp(
        "kaapi_sched_sync",
        SageBuilder::buildIntType(),
        SageBuilder::buildExprListExp(),
        scope
      )
    );
    /* insert after */
    SageInterface::insertStatement(isSgStatement(nodeend->get_SgNode()), syncframe_statement, false);

    SgStatement* restoreframe_statement = SageBuilder::buildExprStatement(
      SageBuilder::buildAssignOp(
        SageBuilder::buildVarRefExp ("__kaapi_thread", scope ),
        SageBuilder::buildFunctionCallExp(
          "kaapi_thread_pop_frame",
          SageBuilder::buildIntType(),
          SageBuilder::buildExprListExp(
          ),
          scope
        )
      )
    );
    /* insert after */
    SageInterface::insertStatement(syncframe_statement, restoreframe_statement, false);
  }
}

void DoKaapiTaskCall( KaapiTaskCallTraversal* ktct, SgGlobal* gscope )
{
  std::list<OneCall>::iterator ifc;
  for (ifc = ktct->_listcall.begin(); ifc != ktct->_listcall.end(); ++ifc)
  {
    OneCall& oc = *ifc;

#if 0 /* Cannot do that automatically: save/pop is right for iterative application (while ?) */
    /* add save restore frame for task creation inside loop 
       before buildFunCall2TaskSpawn, because it may declared __kaapi_thread
       before task spawn in the loop !
    */
    if (oc.forloop != 0)
      buildInsertSaveRestoreFrame( oc.forloop );
#endif

    /* replace function call by task spawn */
    buildFunCall2TaskSpawn( &oc );
  }
}
