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


#include <set>
#include "utils.h"
#include "globals.h"
#include "rose_headers.h"
#include "kaapi_abort.h"
#include "kaapi_c2c_task.h"


// forward decls

/* return a function from the loop statement
*/
static void buildLoopEntrypoint
( 
 SgClassDefinition* this_class_def,
 SgClassDeclaration* contexttype,
 SgGlobal* scope,
 SgFunctionDeclaration*& partial_decl,
 SgFunctionDeclaration*& tramp_decl,
 KaapiTaskAttribute*     kta
);

/* */
static void buildFreeVariable(SgScopeStatement* scope, std::set<SgVariableSymbol*>& list);


/** Preorder traversal: */
class LHSVisitorTraversal : public AstSimpleProcessing {
private:
  static inline bool isMemberExpression(SgNode* n)
  {
    // consider n a member if
    // . it is a rhs
    // . the binary operator is either dot or arrow

    SgNode* const parent = n->get_parent();
    if (parent == NULL || parent == n) return false;
    if (isSgBinaryOp(parent) == NULL) return false;

    if (isSgBinaryOp(parent)->get_lhs_operand() == n)
      return false;

    if (isSgArrowExp(parent) != NULL) return true;
    else if (isSgDotExp(parent) != NULL) return true;

    return false;
  }

public:
  LHSVisitorTraversal( )
  { }
  virtual void visit(SgNode* n)
  {
    SgBinaryOp* binop = isSgBinaryOp(n);
    if (binop !=0)
    {
      std::cout << "Binop " << binop->class_name() << std::endl;
      switch (binop->variantT())
      {
        case V_SgAssignOp:
        case V_SgPlusAssignOp:
        case V_SgMinusAssignOp:
        case V_SgAndAssignOp:
        case V_SgIorAssignOp:
        case V_SgMultAssignOp:
        case V_SgDivAssignOp:
        case V_SgModAssignOp:
        case V_SgXorAssignOp:
        case V_SgLshiftAssignOp:
        case V_SgRshiftAssignOp:
        case V_SgPointerAssignOp: /* kesako ? */
          binop->get_lhs_operand()->setAttribute("kaapiOUT", (AstAttribute*)-1);
          std::cout << "    set LHS " << binop->get_lhs_operand()->class_name() << std::endl;
          break;

        case V_SgPntrArrRefExp:
        default:
          if (binop->/*get_parent()->*/getAttribute("kaapiOUT") != 0)
          {
            binop->get_lhs_operand()->setAttribute("kaapiOUT", (AstAttribute*)-1);
            std::cout << "    inherited LHS " << binop->get_lhs_operand()->class_name() << std::endl;
          }
          break;
      }
    }
    SgVariableDeclaration* vardecl = isSgVariableDeclaration(n);
    if (vardecl !=0)
    {
      const SgInitializedNamePtrList& list = vardecl->get_variables();
      SgInitializedNamePtrList::const_iterator ibeg = list.begin();
      SgInitializedNamePtrList::const_iterator iend = list.end();
      for (; ibeg != iend; ++ibeg)
      {
        SgVariableSymbol* sym = isSgVariableSymbol((*ibeg)->get_symbol_from_symbol_table());
        if (sym !=0)
          declvar.insert(sym);
      }
    }

    SgVarRefExp* varref = isSgVarRefExp(n);
    if (varref !=0)
    {
      if (n->getAttribute("kaapiOUT"))
      {
//        outputvar.insert( std::make_pair(varref->get_symbol(), true) );

	  if (isMemberExpression(varref) == false)
	  {
	    // in fu->bar = baz; like statements, dont consider
	    // bar as a variable reference to be passed as a
	    // member of the task context.
	    outputvar[varref->get_symbol()] = true;
	  }

          std::cout << varref->get_symbol()->get_name().str() << " " << varref->get_symbol() << " is output" << std::endl;
      }
      else
      {
//        if (outputvar.find(varref->get_symbol()) == outputvar.end())
//          outputvar.insert( std::make_pair(varref->get_symbol(), false) );
          std::cout << varref->get_symbol()->get_name().str() << " " << varref->get_symbol() << " is input" << std::endl;

	  if (isMemberExpression(n) == false)
	  {
	    // baz = fu->bar; see above comment.
	    outputvar[varref->get_symbol()] |= false;
	  }

      }
    }
  }

  std::map<SgVariableSymbol*,bool> outputvar;
  std::set<SgVariableSymbol*> declvar;
};

/* Return the list of free variables, ie variable defined outside the scope.
*/
static void buildFreeVariable(SgScopeStatement* scope, std::set<SgVariableSymbol*>& list)
{
  /* set kaapiOUT to all node that are part of lhs of operand */
  LHSVisitorTraversal lhsvisitor;
  lhsvisitor.traverse(scope,preorder);

  std::cout << "List of variables in scope:" << scope->class_name() 
            << " at line:" << scope->get_file_info()->get_line()
            << std::endl;
  std::map<SgVariableSymbol*,bool>::iterator i;
  for (i = lhsvisitor.outputvar.begin(); i != lhsvisitor.outputvar.end(); ++i)
  {
    SgName name = i->first->get_name();
    if (lhsvisitor.declvar.find(i->first) == lhsvisitor.declvar.end())
    {
      SgVariableSymbol* sym = scope->lookup_variable_symbol( name );
      if (sym == 0)
      {
        std::cout << "  " << (i->second ? "[OUT var]" : "[IN var] ") 
                  << name << " is defined outside the scope" 
                  << std::endl;
        list.insert( i->first );
      }
    }
  }
#if 0
  std::set<SgVariableSymbol*>::iterator is;
  for (is = lhsvisitor.declvar.begin(); is != lhsvisitor.declvar.end(); ++is)
  {
    SgName name = (*is)->get_name();
    std::cout << "  Declared variable inside the scope or an inner scope: "
              << name
              << std::endl;
  }
#endif

#if 0
  Rose_STL_Container<SgNode*> listvaref = NodeQuery::querySubTree (scope,V_SgVarRefExp);
  Rose_STL_Container<SgNode*>::iterator i;
  for (  i = listvaref.begin(); i != listvaref.end(); i++)
  {
    SgName name = isSgVarRefExp(*i)->get_symbol()->get_name();
    SgVariableSymbol* sym = scope->lookup_variable_symbol( name );
    if (sym == 0)
      std::cout << "  -" << name << " is defined outside the scope" 
                << (isSgVarRefExp(*i)->getAttribute("kaapiOUT") !=0 ? " LHS" : "" )
                << std::endl;
//    else
//      std::cout << "  +" << name << " is defined inside the scope" << std::endl;
  }
#endif
  std::cout << std::endl;
  
  
}


// loop canonicalizer

class forLoopCanonicalizer
{
public:
  class AffineVariable
  {
  public:
    // refer to step clause enum
    SgInitializedName* name_;
    unsigned int op_;
    SgExpression* incr_;

    AffineVariable
    (SgInitializedName* name, unsigned int op, SgExpression* incr)
      : name_(name), op_(op), incr_(incr) {}
  };

  typedef std::list<AffineVariable> AffineVariableList;

private:

  // original loop statement
  SgForStatement* for_stmt_;

  // doStepLabelTransform
  SgLabelStatement* label_stmt_;

  // findIteratorName
  SgInitializedName* iter_name_;

  // findAffineVariables
  std::list<AffineVariable>& affine_vars_;

  // normalizeCxxOperators

  // test clause
  enum
  {
    LESS_STRICT = 0,
    LESS_EQ,
    GREATER_STRICT,
    GREATER_EQ,
    NOT_EQ
  };
  unsigned int test_op_;
  SgExpression* test_lhs_;
  SgExpression* test_rhs_;

  // step clause
  enum
  {
    PLUS_PLUS = 0,
    MINUS_MINUS,
    PLUS_ASSIGN,
    MINUS_ASSIGN,
    ASSIGN_ADD,
    ASSIGN_SUB
  };
  SgExpression* stride_;
  bool is_forward_;

  static bool isVarModified(SgNode*, SgInitializedName*);
  static bool getModifiedVariable
  (SgNode*, SgInitializedName*&, unsigned int&, SgExpression*&);
  static bool isIncreasingExpression(unsigned int);
  static bool isInclusiveOperator(unsigned int);
  static bool getNormalizedCxxOperator
  (SgExpression*, unsigned int&, SgExpression*&, SgExpression*&);
  static bool getCxxTestOperatorLhs
  (SgExpression*, SgInitializedName*&);
  static bool getNormalizedStep
  (SgExpression*, unsigned int&, SgExpression*&);
  static bool getNormalizedStep
  (SgExpression*, unsigned int&, SgExpression*&, SgExpression*&);

  forLoopCanonicalizer(SgForStatement* for_stmt, std::list<AffineVariable>& affine_vars)
    : for_stmt_(for_stmt), affine_vars_(affine_vars) {}

  // applied transforms and passes
  bool findIteratorName();
  bool findAffineVariables();
  bool doStepLabelTransform();
  bool doMultipleStepTransform();
  bool normalizeCxxOperators();
  bool normalizeTest();
  bool doStrictIntegerTransform();

public:
  static bool canonicalize(SgForStatement*);
  static bool canonicalize(SgForStatement*, std::list<AffineVariable>&);

  static bool isDecreasingOp(unsigned int);
};


bool forLoopCanonicalizer::isDecreasingOp(unsigned int op)
{
  switch (op)
  {
  case PLUS_PLUS:
  case PLUS_ASSIGN:
  case ASSIGN_ADD:
    return false;
    break ;
  default: break;
  }
  return true;
}

bool forLoopCanonicalizer::doStepLabelTransform()
{
  // insert a step statement at the end of the body
  // replace all the continue by goto statements

  SgScopeStatement* const scope_stmt =
    SageInterface::getEnclosingProcedure(for_stmt_);

  SgStatement* const old_body = SageInterface::getLoopBody(for_stmt_);
  SgBasicBlock* const new_body = SageBuilder::buildBasicBlock();

  SgName label_name = "kaapi_continue_label__";
  label_name << ++SageInterface::gensym_counter;

  label_stmt_ = SageBuilder::buildLabelStatement
    (label_name, SageBuilder::buildBasicBlock(), scope_stmt);

  SageInterface::changeContinuesToGotos(old_body, label_stmt_);
  SageInterface::appendStatement(old_body, new_body);
  SageInterface::appendStatement(label_stmt_, new_body);

  SageInterface::setLoopBody(for_stmt_, new_body);

  return true;
}

bool forLoopCanonicalizer::isInclusiveOperator(unsigned int op)
{
  if (op == NOT_EQ) return false;
  else if (op == LESS_STRICT) return false;
  else if (op == GREATER_STRICT) return false;
  return true;
}

bool forLoopCanonicalizer::doStrictIntegerTransform()
{
  // assume findIteratorName
  // assume normalizeCxxOperators
  // assume doMultipleStepTransform
  // assume test expression lhs is the iterator
  // assume test expression rhs is the high bound

  // turn a pointer type iterator into a strict integer
  // for (double* p = array; p != end; ++p)
  //   body();
  // becomes: 
  // __j = end - p;
  // for (__i = 0; __i < __j; ++__i)
  // {
  //   body();
  // next: ++p;
  // }

#define CONFIG_LOCAL_DEBUG

  SgExpression* const incr_expr = for_stmt_->get_increment();
  // assume(incr_expr);

  SgExpression* hi_expr = NULL;
  SgExpression* lo_expr = NULL;

  if ((test_op_ == LESS_STRICT) || (test_op_ == LESS_EQ))
  {
    if (is_forward_ == false)
    {
      // assume invalid for now
#ifdef CONFIG_LOCAL_DEBUG
      printf("(i < j) test expression, but decreasing step\n");
#endif
      return false;
    }

    hi_expr = test_rhs_;
    lo_expr = test_lhs_;
  }
  else if ((test_op_ == GREATER_STRICT) || (test_op_ == GREATER_EQ))
  {
    if (is_forward_ == true)
    {
      // assume invalid for now
#ifdef CONFIG_LOCAL_DEBUG
      printf("(i > j) test expression, but increasing step\n");
#endif
      return false;
    }

    hi_expr = test_lhs_;
    lo_expr = test_rhs_;
  }
  else if (test_op_ == NOT_EQ)
  {
    if (is_forward_ == true)
    {
      hi_expr = test_rhs_;
      lo_expr = test_lhs_;
    }
    else
    {
      hi_expr = test_lhs_;
      lo_expr = test_rhs_;
    }
  }
  else
  {
#ifdef CONFIG_LOCAL_DEBUG
    printf("invalid test expression\n");
#endif
    return false;
  }

  //
  // move the loop init statement just before the loop
  SgStatementPtrList& init_ptrlist = for_stmt_->get_init_stmt();
  if (init_ptrlist.size() >= 2) return false;
  if (init_ptrlist.size() == 1)
  {
    SgStatement* const init_stmt = init_ptrlist.front();
    SageInterface::insertStatement(for_stmt_, init_stmt);
    init_ptrlist.pop_back();
  }

  //
  // generate the variable holding the difference. note that the
  // resulting count may be positive even if the loop cond is not
  // initially true. This case is handled by the canonicalizer
  // in generateInitialTest.
  // long count = ((hi - lo) % incr ? 1 : 0) + (hi - lo) / incr;

  // scope
  SgScopeStatement* const count_scope =
    SageInterface::getScope(for_stmt_->get_parent());
  if (count_scope == NULL)
  {
#ifdef CONFIG_LOCAL_DEBUG
    printf("invalid top scope\n");
#endif
    return false;
  }

  // declaration
  SgName count_name = "__kaapi_count_";
  count_name << ++SageInterface::gensym_counter;

  // substraction. add 1 to inclsuive (non strict) operators.
  SgExpression* diff_op = SageBuilder::buildSubtractOp(hi_expr, lo_expr);
  if (isInclusiveOperator(test_op_))
  {
    diff_op = SageBuilder::buildAddOp
      (diff_op, SageBuilder::buildLongIntVal(1));
  }

  // modulus and ternary expressions
  diff_op->set_need_paren(true);
  SgExpression* const noteq_expr = SageBuilder::buildNotEqualOp
  (
   SageBuilder::buildModOp(diff_op, stride_),
   SageBuilder::buildLongIntVal(0)
  );

  SgConditionalExp* const tern_expr = SageBuilder::buildConditionalExp
  (
   noteq_expr,
   SageBuilder::buildLongIntVal(1),
   SageBuilder::buildLongIntVal(0)
  );

  SgExpression* const add_op = SageBuilder::buildAddOp
    (SageBuilder::buildDivideOp(diff_op, stride_), tern_expr);

  // insert count declaration
  SgType* const count_type = SageBuilder::buildLongType();
  SgVariableDeclaration* const count_decl =
    SageBuilder::buildVariableDeclaration
    (count_name, count_type, 0, count_scope);
  SgVarRefExp* const count_vref_expr =
    SageBuilder::buildOpaqueVarRefExp(count_name, count_scope);
  SageInterface::prependStatement(count_decl, count_scope);

  // insert for init statement
  SgExprStatement* const assign_stmt = SageBuilder::buildAssignStatement
    (count_vref_expr, SageBuilder::buildLongIntVal(0));
  for_stmt_->append_init_stmt(assign_stmt);

  // insert count < limit expression. delete previous test
  // expression before clobbering and set as the new one.
  SgStatement* const test_stmt = for_stmt_->get_test();
  if (test_stmt != NULL)
  {
    for_stmt_->set_test(NULL);
    delete test_stmt;
  }

  SgExpression* const less_expr =
    SageBuilder::buildLessThanOp(count_vref_expr, add_op);
  for_stmt_->set_test_expr(less_expr);

  // move increment to end of body
  SgStatement* const incr_stmt =
    SageBuilder::buildExprStatement(incr_expr);
  SgScopeStatement* const scope_stmt =
    isSgScopeStatement(SageInterface::getLoopBody(for_stmt_));
  SageInterface::appendStatement(incr_stmt, scope_stmt);

  // insert ++count as increment expression
  for_stmt_->set_increment(SageBuilder::buildPlusPlusOp(count_vref_expr));

  return true;

#undef CONFIG_LOCAL_DEBUG
}

bool forLoopCanonicalizer::getNormalizedCxxOperator
(SgExpression* expr, unsigned int& op, SgExpression*& lhs, SgExpression*& rhs)
{
  // TODO: check if this is a method

#define CONFIG_LOCAL_DEBUG

  if (SageInterface::is_Cxx_language() == false) return false;

  SgFunctionCallExp* const call_expr = isSgFunctionCallExp(expr);
  if (call_expr == NULL) return false;

  SgName name = call_expr->getAssociatedFunctionDeclaration()->get_name();

  // name to op
  if (name == "operator<") op = LESS_STRICT;
  else if (name == "operator<=") op = LESS_EQ;
  else if (name == "operator>") op = GREATER_STRICT;
  else if (name == "operator>=") op = GREATER_EQ;
  else if (name == "operator!=") op = NOT_EQ;
  else
  {
#ifdef CONFIG_LOCAL_DEBUG
    printf("invalid cxx operator: %s\n", name.str());
#endif
    return false;
  }

  SgExpressionPtrList const expr_list =
    call_expr->get_args()->get_expressions();

  SgDotExp* const dot_expr = isSgDotExp(call_expr->get_function());
  if (dot_expr != NULL)
  {
    // object instance the lhs
    lhs = dot_expr->get_lhs_operand();
    if (expr_list.size() != 1) return false;
    rhs = expr_list[0];
  }
  else // not a method
  {
    if (expr_list.size() != 2) return false;
    lhs = expr_list[0];
    rhs = expr_list[1];
  }

  return true;

#undef CONFIG_LOCAL_DEBUG
}

bool forLoopCanonicalizer::getCxxTestOperatorLhs
(SgExpression* expr, SgInitializedName*& iter_name)
{
  // return true if the caller must stop processing
  // the name may not be set upon return, indicating
  // either a function call or an invalid operator

  unsigned int op;
  SgExpression* lhs;
  SgExpression* rhs;

  const bool is_operator = getNormalizedCxxOperator
    (expr, op, lhs, rhs);

  // caller continue processing
  if (is_operator == false) return false;

  // return true from now to indicate a Cxx operator

  iter_name = NULL;

  SgVarRefExp* const vref_expr = isSgVarRefExp(lhs);
  if (vref_expr == NULL) return true;

  iter_name = vref_expr->get_symbol()->get_declaration();

  return true;
}

bool forLoopCanonicalizer::getNormalizedStep
(SgExpression* expr, unsigned int& op, SgExpression*& lhs, SgExpression*& rhs)
{
  // assume expression is a canonical operation

  // only support the following forms:
  // ++fu and fu++
  // fu += bar
  // fu = fu + bar

#define CONFIG_LOCAL_DEBUG

  if (SageInterface::is_Cxx_language() == true)
  {
    SgFunctionCallExp* const call_expr = isSgFunctionCallExp(expr);
    if (call_expr != NULL)
    {
      // arg offset, for this pointer
      unsigned int has_this = 0;

      SgDotExp* const dot_expr = isSgDotExp(call_expr->get_function());
      if (dot_expr != NULL)
      {
	// object instance the lhs
	lhs = dot_expr->get_lhs_operand();
	has_this = 1;
      }

      // get from argument list
      SgExpressionPtrList const expr_list =
	call_expr->get_args()->get_expressions();
      if (has_this == 0)
      {
	// must be at least one arg
	if (expr_list.size() == 0)
	{
	  return false;
	}

	lhs = expr_list[0];
      }

      if (lhs == NULL) return false;

      SgName name =
	call_expr->getAssociatedFunctionDeclaration()->get_name();

      if (name == "operator--")
      {
	rhs = NULL;
	op = MINUS_MINUS;
	return true;
      }
      else if (name == "operator++")
      {
	rhs = NULL;
	op = PLUS_PLUS;
	return true;
      }
      else if (name == "operator+=")
      {
	if (expr_list.size() < (2 - has_this))
	{
	  return false;
	}

	rhs = expr_list[1 - has_this];
	op = PLUS_ASSIGN;
	return true;
      }
      else if (name == "operator-=")
      {
	if (expr_list.size() < (2 - has_this))
	{
	  return false;
	}

	op = MINUS_ASSIGN;
	rhs = expr_list[1 - has_this];
	return true;
      }

      // error reached
#ifdef CONFIG_LOCAL_DEBUG
      printf("invalid cxx operator: %s\n", name.str());
#endif
      return false;

    } // cxx call case
  } // cxx case

  // non cxx operator case

  if (isSgAssignOp(expr))
  {
    SgExpression* const saved_expr = expr;

    expr = isSgAssignOp(expr)->get_rhs_operand();

    if (isSgAddOp(expr) != NULL)
    {
      lhs = isSgAddOp(expr)->get_lhs_operand();
      rhs = isSgAddOp(expr)->get_rhs_operand();
      op = ASSIGN_ADD;
      return true;
    }
    else if (isSgSubtractOp(expr) != NULL)
    {
      lhs = isSgSubtractOp(expr)->get_lhs_operand();
      rhs = isSgSubtractOp(expr)->get_rhs_operand();
      op = ASSIGN_SUB;
      return true;
    }
    else if (isSgAssignInitializer(expr) != NULL)
    {
      // the only accepted forms are
      // fu = fu.op(bar)
      // fu = op(fu, bar)

      lhs = isSgAssignOp(saved_expr)->get_lhs_operand();

      // return the rhs
      expr = isSgAssignInitializer(expr)->get_operand();
      if (expr == NULL) return false;

      SgFunctionCallExp* const call_expr = isSgFunctionCallExp(expr);
      if (call_expr == NULL) return false;
      SgName name =
	call_expr->getAssociatedFunctionDeclaration()->get_name();
      if (name == "operator-") op = ASSIGN_SUB;
      else if (name == "operator+") op = ASSIGN_ADD;
      else return false;

      SgExpressionPtrList const expr_list =
	call_expr->get_args()->get_expressions();

      SgDotExp* const dot_expr = isSgDotExp(call_expr->get_function());
      if (dot_expr != NULL) // fu = fu.op(bar);
      {
	if (SageInterface::is_Cxx_language() == false) return false;
	if (expr_list.size() != 1) return false;
	rhs = expr_list[0];
	return true;
      }
      else // fu = op(fu, bar);
      {
	if (expr_list.size() != 2) return false;
	rhs = expr_list[1];
	return true;
      }

      return false;
    } // isSgInitializer
  }
  else if (isSgBinaryOp(expr))
  {
    SgBinaryOp* const binop = isSgBinaryOp(expr);
    lhs = binop->get_lhs_operand();
    rhs = binop->get_rhs_operand();
    if (binop->variantT() == V_SgPlusAssignOp)
    {
      op = PLUS_ASSIGN;
      return true;
    }
    else if (binop->variantT() == V_SgMinusAssignOp)
    {
      op = MINUS_ASSIGN;
      return true;
    }
  }
  else if (isSgUnaryOp(expr))
  {
    SgUnaryOp* const unop = isSgUnaryOp(expr);
    lhs = unop->get_operand();
    rhs = NULL;
    if (unop->variantT() == V_SgPlusPlusOp)
    {
      op = PLUS_PLUS;
      return true;
    }
    else if (unop->variantT() == V_SgMinusMinusOp)
    {
      op = MINUS_MINUS;
      return true;
    }
  }

  // error reached
#ifdef CONFIG_LOCAL_DEBUG
  printf("invalid assignment: %s\n", expr->class_name().c_str());
#endif
  return false;

#undef CONFIG_LOCAL_DEBUG
}

bool forLoopCanonicalizer::getNormalizedStep
(SgExpression* expr, unsigned int& op, SgExpression*& rhs)
{
  SgExpression* unused_lhs;
  return getNormalizedStep(expr, op, unused_lhs, rhs);
}

bool forLoopCanonicalizer::isIncreasingExpression(unsigned int op)
{
  // return true if the expression is an increasing one

  if (op == PLUS_PLUS) return true;
  else if (op == PLUS_ASSIGN) return true;
  else if (op == ASSIGN_ADD) return true;
  return false;
}

bool forLoopCanonicalizer::normalizeCxxOperators()
{
  // assume doMultipleStepTransform

  // fill and internal representation of operators
  // since C++ turns BinaryOp and UnaryOp into
  // function calls. return false if such representation
  // cannot be built

  // upon return, the following members are filled
  // test_op_, test_lhs_, test_rhs_
  // is_forward_, stride_

#define CONFIG_LOCAL_DEBUG

  //
  // test expression

  SgExpression* const test_expr = for_stmt_->get_test_expr();
  if (test_expr == NULL)
  {
#ifdef CONFIG_LOCAL_DEBUG
    printf("test_expr == NULL\n");
#endif
    return false;
  }

  if (isSgBinaryOp(test_expr))
  {
    SgBinaryOp* const binary_op = isSgBinaryOp(test_expr);
    test_lhs_ = binary_op->get_lhs_operand();
    test_rhs_ = binary_op->get_rhs_operand();
    if (isSgLessThanOp(binary_op)) test_op_ = LESS_STRICT;
    else if (isSgLessOrEqualOp(binary_op)) test_op_ = LESS_EQ;
    else if (isSgGreaterThanOp(binary_op)) test_op_ = GREATER_STRICT;
    else if (isSgGreaterOrEqualOp(binary_op)) test_op_ = GREATER_EQ;
    else if (isSgNotEqualOp(binary_op)) test_op_ = NOT_EQ;
    else
    {
#ifdef CONFIG_LOCAL_DEBUG
      printf("invalid binop %s\n", binary_op->class_name().c_str());
#endif
    }
  }
  else // look for cxx operator
  {
    const bool is_valid = getNormalizedCxxOperator
      (test_expr, test_op_, test_lhs_, test_rhs_);
    if (is_valid == false) return false;
  }

  if ((test_lhs_ == NULL) || (test_rhs_ == NULL))
  {
#ifdef CONFIG_LOCAL_DEBUG
    printf("invalid (test_lhs_ || test_rhs_)\n");
#endif
    return false;
  }

  //
  // increment expression

  SgExpression* const incr_expr = for_stmt_->get_increment();
  if (incr_expr == NULL)
  {
#ifdef CONFIG_LOCAL_DEBUG
    printf("incr_expr == NULL\n");
#endif
    return false;
  }

  unsigned int step_op;
  SgExpression* rhs;
  if (getNormalizedStep(incr_expr, step_op, rhs) == false)
  {
#ifdef CONFIG_LOCAL_DEBUG
    printf("cannot normalize step\n");
#endif
    return false;
  }

  // is this a forward iteration
  is_forward_ = isIncreasingExpression(step_op);

  // get the stride expression
  if ((step_op != PLUS_PLUS) && (step_op != MINUS_MINUS))
  {
    // fu += stride;
    // fu = fu + stride;
    stride_ = rhs;
  }
  else
  {
    // ++fu;
    stride_ = SageBuilder::buildLongIntVal(1);
  }

  return true;

#undef CONFIG_LOCAL_DEBUG
}

bool forLoopCanonicalizer::isVarModified
(SgNode* node, SgInitializedName* looked_name)
{
  // return true if the var identified by
  // name is modified by the subtree under node

  unsigned int op;
  SgExpression* lhs;
  SgExpression* rhs;

  // assume isSgExpression(node)

  if (getNormalizedStep(isSgExpression(node), op, lhs, rhs) == false)
    return false;

  SgVarRefExp* const vref_expr = isSgVarRefExp(lhs);
  if (vref_expr == NULL) return false;
  if (vref_expr->get_symbol() == NULL) return false;

  SgInitializedName* const her_name =
    vref_expr->get_symbol()->get_declaration();
  if (her_name == NULL) return false;
  return her_name->get_name() == looked_name->get_name();
}

bool forLoopCanonicalizer::getModifiedVariable
(SgNode* node, SgInitializedName*& name, unsigned int& op, SgExpression*& expr)
{
  // expr the modifying expression
  // TODO: redundant with isVarModified

  SgExpression* lhs;

  // assume isSgExpression(node)

  if (getNormalizedStep(isSgExpression(node), op, lhs, expr) == false)
    return false;

  SgVarRefExp* const vref_expr = isSgVarRefExp(lhs);
  if (vref_expr == NULL) return false;
  if (vref_expr->get_symbol() == NULL) return false;

  name = vref_expr->get_symbol()->get_declaration();
  if (name == NULL) return false;
  return true;
}

bool forLoopCanonicalizer::doMultipleStepTransform()
{
#define CONFIG_LOCAL_DEBUG 1

  // assume findIteratorName

  // if the increment part is made of multiple statements,
  // detect the one we should keep in the canonical form
  // and move the other after the end of body label

  SgExpression* const incr_expr = for_stmt_->get_increment();
  if (incr_expr == NULL) return true;

  SgCommaOpExp* comma_expr = isSgCommaOpExp(incr_expr);
  if (comma_expr == NULL) return true;

  // foreach comma tree node, check if the lhs operand
  // contains a modifying operation on iter_name. there
  // must be only one such occurence. move every other
  // expression to the end of the body.
  SgNode* node;
  SgNode* iter_node = NULL;
  bool is_done = false;
  while (1)
  {
    node = comma_expr->get_rhs_operand();

  skip_rhs_operand:

    // process the node here. find iterator in rhs lhs operand
    const bool is_modified = isVarModified(node, iter_name_);
    if (is_modified == true)
    {
      // modifying twice the iterator is considered an error
      if (iter_node != NULL)
      {
#ifdef CONFIG_LOCAL_DEBUG
	printf("iterator modified twice\n");
#endif
	return false;
      }

      iter_node = node;
    }
    else
    {
      SgStatement* const new_stmt =
	SageBuilder::buildExprStatement(isSgExpression(node));
      SgScopeStatement* const scope_stmt =
	isSgScopeStatement(SageInterface::getLoopBody(for_stmt_));
      SageInterface::appendStatement(new_stmt, scope_stmt);
    }

    if (is_done == true) break ;

    // last node reached, lhs is a leaf
    if (isSgCommaOpExp(comma_expr->get_lhs_operand()) == NULL)
    {
      is_done = true;
      node = comma_expr->get_lhs_operand();
      goto skip_rhs_operand;
    }

    // next tree node
    comma_expr = isSgCommaOpExp(comma_expr->get_lhs_operand());
  }

  if (iter_node == NULL)
  {
#ifdef CONFIG_LOCAL_DEBUG
    printf("%s iterator variable not found in increment\n",
	   iter_name_->get_qualified_name().str());
#endif // CONFIG_LOCAL_DEBUG
    return false;
  }

  // move all but matched one to the end of body
  // TODO: delete old increment?
  iter_node->set_parent(incr_expr->get_parent());
  for_stmt_->set_increment(isSgExpression(iter_node));
  
  return true;

#undef CONFIG_LOCAL_DEBUG
}

bool forLoopCanonicalizer::normalizeTest()
{
  // assume normalizeCxxOperators()
  // transform test condition to canonical one

  return true;
}

bool forLoopCanonicalizer::findIteratorName(void)
{
  // find and store the iterator in iter_name_
  // assumed to be the loop test expression lhs

#define CONFIG_LOCAL_DEBUG

  // find ithe iterator name using for loop initializer

  SgExpression* const test_expr = for_stmt_->get_test_expr();
  if (test_expr == NULL) return NULL;

  // run earlier than cxx normalization pass
  if (getCxxTestOperatorLhs(test_expr, iter_name_) == true)
  {
    // may be an invalid operator
    return iter_name_ != NULL;
  }

  // non cxx case. assume a binary operator.
  SgBinaryOp* const binary_op = isSgBinaryOp(test_expr);
  if (binary_op == NULL)
  {
#ifdef CONFIG_LOCAL_DEBUG
    printf("not a BinaryOp: %s\n", test_expr->class_name().c_str());
#endif
    return false;
  }

  SgVarRefExp* const vref_expr = isSgVarRefExp(binary_op->get_lhs_operand());
  if (vref_expr == NULL)
  {
#ifdef CONFIG_LOCAL_DEBUG
    printf("not an VarRefExp: %s\n", vref_expr->class_name().c_str());
#endif
    return false;
  }

  iter_name_ = vref_expr->get_symbol()->get_declaration();

  return true;

#undef CONFIG_LOCAL_DEBUG
}

bool forLoopCanonicalizer::findAffineVariables(void)
{
  // keep track of affine variables. An affine
  // variable is a named expression present in
  // the increment clause of the loop.

  // no prior pass assumed, but this code
  // is redundant with the subtree iteration
  // of doMultipleStepTransform

#define CONFIG_LOCAL_DEBUG

  // variable name, op, modifying expression
  SgInitializedName* name;
  unsigned int op;
  SgExpression* expr;

  // process increment expression first since the
  // affine constraint is put on the increment clause

  SgExpression* const incr_expr = for_stmt_->get_increment();
  if (incr_expr == NULL) return true;

  SgCommaOpExp* comma_expr = isSgCommaOpExp(incr_expr);
  if (comma_expr == NULL)
  {
    if (getModifiedVariable(isSgNode(incr_expr), name, op, expr))
      affine_vars_.push_back(AffineVariable(name, op, expr));
    return true;
  }

  // foreach comma tree node, check if the lhs operand
  // contains a modifying operation on iter_name. there
  // must be only one such occurence. move every other
  // expression to the end of the body.
  SgNode* node;
  bool is_done = false;
  while (1)
  {
    node = comma_expr->get_rhs_operand();

  skip_rhs_operand:

    // process the node here. track modified variables.
    if (getModifiedVariable(node, name, op, expr))
      affine_vars_.push_back(AffineVariable(name, op, expr));

    if (is_done == true) break ;

    // last node reached, lhs is a leaf
    if (isSgCommaOpExp(comma_expr->get_lhs_operand()) == NULL)
    {
      is_done = true;
      node = comma_expr->get_lhs_operand();
      goto skip_rhs_operand;
    }

    // next tree node
    comma_expr = isSgCommaOpExp(comma_expr->get_lhs_operand());
  }

  return true;

#undef CONFIG_LOCAL_DEBUG
}

bool forLoopCanonicalizer::canonicalize
(
 SgForStatement* for_stmt,
 std::list<AffineVariable>& affine_vars
)
{
  forLoopCanonicalizer canon(for_stmt, affine_vars);

  // order matters. refer to method comments.
  if (canon.findAffineVariables() == false) return false;
  if (canon.findIteratorName() == false) return false;
  if (canon.doStepLabelTransform() == false) return false;
  if (canon.doMultipleStepTransform() == false) return false;
  if (canon.normalizeCxxOperators() == false) return false;
  if (canon.normalizeTest() == false) return false;
  if (canon.doStrictIntegerTransform() == false) return false;

  return true;
}

bool forLoopCanonicalizer::canonicalize(SgForStatement* for_stmt)
{
  std::list<AffineVariable> unused_vars;
  return canonicalize(for_stmt, unused_vars);
}


/* helper class to find two symbol with same name */
class compare_symbol_name {
public:
  compare_symbol_name( const std::string& name )
   : _name(name)
  {}
  bool operator()( const SgVariableSymbol* s )
  { return s->get_name() == _name; }
  const std::string& _name;
};

// forward decl

static void buildLoopEntrypointBody
( 
  SgFunctionDeclaration*       func,
  SgFunctionDeclaration*       splitter,
  std::set<SgVariableSymbol*>& listvar,
  SgClassDeclaration*          contexttype,
  SgInitializedName*           ivar,
  SgExpression*                global_begin_iter,
  SgExpression*                global_step,
  SgStatement*                 loopbody,
  SgGlobal*                    scope,
  bool			       hasIncrementalIterationSpace,
  bool			       isInclusiveUpperBound,
  forLoopCanonicalizer::AffineVariableList&,
  KaapiTaskAttribute*
);


/** Transform a loop with independent iteration in an adaptive form
    The loop must have a cannonical form:
       for ( i = begin; i<end; ++i)
         body
         
    The transformation will inline the function depending of input parameters (begin,end)
    and all other variables passed by references.:
    for_loop_adaptive(begin,end)
    {
      work = kaapi_workqueue(begin, end);

      begin_adaptive()
      
      while (range = wq.pop(popsize))
      {
        for (i = range.begin; i<range.end; ++i)
          body
      }
      end_adaptive()
    }
    
    The splitter generated call create tasks that call for_loop_adaptive(stolen_range.begin, stolen_range.end),
    all variables in the body are accessed either through copies or by references.
    The clause will indicate the sharing rule (shared or private) in a similar fashion as for openMP.
    Note on popsize:
      - currently the popsize is computed...from scratch ? = N/(K*kaapi_get_concurrency()) as for
      auto-partitionner with intel TBB.
  
   On success, return a SgScopeStatement that correspond to rewrite of the loop statement.  
*/
SgStatement* buildConvertLoop2Adaptative( 
  SgScopeStatement*       loop,
  SgFunctionDeclaration*& entrypoint,
  SgFunctionDeclaration*& splitter,
  SgClassDeclaration*&    contexttype,
  KaapiTaskAttribute*     kta
)
{
  entrypoint  = 0;
  splitter    = 0;
  contexttype = 0;
  if (loop ==0) return 0;
  
  switch (loop->variantT()) /* see ROSE documentation if missing node or node */
  {
    case V_SgForStatement:
      break;
    default: /* should never enter here, because verification was made during pragma directive */
      return 0;
  }

  SgForStatement* forloop = isSgForStatement( loop );
  SgScopeStatement* scope = SageInterface::getScope( forloop->get_parent() );

  SgExpression* const saved_test_expr = forloop->get_test_expr();

  std::list<forLoopCanonicalizer::AffineVariable> affine_vars;
  if (forLoopCanonicalizer::canonicalize(forloop, affine_vars) == false)
  {
    std::cerr << "****[kaapi_c2c] canonicalize loop failed" << std::endl;
    return 0;
  }

#if 0 /* normalization will rename iteration variable and change test to have <= or >= 
         not required
      */
  if (!SageInterface::forLoopNormalization (forloop))
  {
    /* cannot */
    Sg_File_Info* fileInfo = forloop->get_file_info();
    std::cerr << "****[kaapi_c2c] wraning. Loop cannot be normalized."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
#endif

  /* Get access to the canonical representation of the loop
  */
  SgInitializedName* ivar;
  SgExpression* begin_iter;
  SgExpression* end_iter;
  SgExpression* step;
  SgStatement*  loopbody;
  bool hasIncrementalIterationSpace;
  bool isInclusiveUpperBound;

  bool retval = SageInterface::isCanonicalForLoop(
        forloop, 
        &ivar,
        &begin_iter,
        &end_iter,
        &step,
        &loopbody,
        &hasIncrementalIterationSpace,
        &isInclusiveUpperBound
  );

  if (!retval)
  {
    /* cannot put it in canonical form */
    Sg_File_Info* fileInfo = forloop->get_file_info();
    std::cerr << "****[kaapi_c2c] wraning. Loop is not in canonical form."
              << "     #pragma kaapi loop ignored"
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  
  SgGlobal* bigscope = SageInterface::getGlobalScope(loop);

  /* build the free variable list of the forloop
  */
  std::set<SgVariableSymbol*> listvar;
  std::set<SgVariableSymbol*>::iterator ivar_beg;
  std::set<SgVariableSymbol*>::iterator ivar_end;

  if (isSgScopeStatement(loopbody) == NULL)
  {
    std::cerr << "cannot find for loop body scope" << std::endl;
    return 0;
  }

  buildFreeVariable(isSgScopeStatement(loopbody), listvar);

  // for a method, find the class definition
  SgClassDefinition* this_class_def = NULL;
  if (SageInterface::is_Cxx_language())
  {
    SgFunctionDefinition* const enclosing_def =
      SageInterface::getEnclosingFunctionDefinition(loopbody);
    if (enclosing_def != NULL && enclosing_def->get_declaration())
    {
      SgMemberFunctionDeclaration* const member_decl =
	isSgMemberFunctionDeclaration(enclosing_def->get_declaration());
      if (member_decl != NULL)
      {
	// printf("FOUND_METHOD: %s\n", member_decl->get_name().str());
#if 0 // TODO
	member_decl->get_class_scope()->get_qualified_name().str();
	::member_decl->get_name().str();
#endif // TODO

	this_class_def = member_decl->get_class_scope();
      }
    }
  }
  
  /* suppress any ivar or __kaapi_thread instance 
  */
  ivar_beg = std::find_if( listvar.begin(), listvar.end(), compare_symbol_name(ivar->get_name()) );
  if (ivar_beg != listvar.end()) listvar.erase( ivar_beg );

  ivar_beg = std::find_if( listvar.begin(), listvar.end(), compare_symbol_name("__kaapi_thread") );
  if (ivar_beg != listvar.end()) listvar.erase( ivar_beg );

  /* Build the structure for the argument of most of the other function:
     The data structure contains
     - all free parameters required to call the entrypoint for loop execution
     - a __kaapi_this_ pointer, if we are in a method
     The structure store pointer to the actual parameters.
  */
  SgClassDeclaration* contexttype = buildOutlineArgStruct( listvar, bigscope, this_class_def );
  
  /* append the type for the splitter juste before the enclosing function definition
  */
  SageInterface::insertStatement (
    SageInterface::getEnclosingFunctionDeclaration(forloop, true),
    contexttype
  );

  /* generate the declaration + definition of the function that 
     will replace the loop statement.
        void loop_entrypoint( contexttype* context )
     The body will be fill after.
  */
  SgFunctionDeclaration* partial_decl;
  buildLoopEntrypoint(this_class_def, contexttype, bigscope, partial_decl, entrypoint, kta);

  /* The new block of instruction that will replace the forloop */
  SgBasicBlock* newbbcall = SageBuilder::buildBasicBlock();

  /* Generate the body of the entrypoint:
     kaapi_workqueue_t work;
     kaapi_workqueue_init( &work, 0, __kaapi_end );
     The interval [0,__kaapi_end] is the number of iteration in the loop.
     Each pop of range [i,j) will iterate to the original index [begin+i*incr
  */
  SgBasicBlock* body = partial_decl->get_definition()->get_body();

  // resolve the splitter
  {
    SgName splitter_name("kaapi_splitter_default");
    SgFunctionSymbol* const func_sym =
      bigscope->lookup_function_symbol(splitter_name);
    // assume(func_sym);
    splitter = func_sym->get_declaration();
  }

  /* Complete the body of the entry pointer with splitter information
  */
  buildLoopEntrypointBody(
    partial_decl,
    splitter,
    listvar,
    contexttype, 
    ivar,
    begin_iter,
    step,
    loopbody, 
    bigscope,
    hasIncrementalIterationSpace,
    isInclusiveUpperBound,
    affine_vars,
    kta
  );

  /* fwd declaration of the entry point */
  if (this_class_def == NULL)
  {
    SgFunctionDeclaration* const fwd_decl = 
      SageBuilder::buildNondefiningFunctionDeclaration
      (entrypoint, forloop->get_scope());
    fwd_decl->get_declarationModifier().get_storageModifier().setStatic();
    SageInterface::prependStatement(fwd_decl, bigscope);
  }

  /* Generate the calle new basic block
     2 variables: __kaapi_context and __kaapi_thread.
  */
  SgVariableSymbol* newkaapi_threadvar = 
      SageInterface::lookupVariableSymbolInParentScopes(
          "__kaapi_thread", 
          forloop 
  );
  if (newkaapi_threadvar ==0)
  {
    buildInsertDeclarationKaapiThread( scope, newbbcall);
    newkaapi_threadvar = 
          SageInterface::lookupVariableSymbolInParentScopes(
              "__kaapi_thread", 
              newbbcall 
    );
  }

  // allocate offsetof(type, data) + sizeof(context_type) on the thread stack
  std::ostringstream offsetof_stream;
  offsetof_stream << "offsetof(kaapi_splitter_context_t, data)";

  SgExpression* size_expr = SageBuilder::buildAddOp
    (
     SageBuilder::buildOpaqueVarRefExp(offsetof_stream.str(), newbbcall),
     SageBuilder::buildSizeOfOp(contexttype->get_type())
    );

  SgFunctionCallExp* call_expr = SageBuilder::buildFunctionCallExp
    (
     "kaapi_thread_pushdata_align",
     SageBuilder::buildPointerType(SageBuilder::buildVoidType()),
     SageBuilder::buildExprListExp
     (
      SageBuilder::buildVarRefExp("__kaapi_thread", newbbcall),
      SageBuilder::buildCastExp
      (size_expr, SageBuilder::buildUnsignedIntType()),
      SageBuilder::buildUnsignedLongVal(8)
     ),
     bigscope
    );

  SgVariableDeclaration* splitter_context = SageBuilder::buildVariableDeclaration
  (
   "__splitter_context",
   SageBuilder::buildPointerType(kaapi_splitter_context_ROSE_type),
   SageBuilder::buildAssignInitializer
   (
    SageBuilder::buildCastExp
    (
     call_expr,
     SageBuilder::buildPointerType(kaapi_splitter_context_ROSE_type)
    ),
    SageBuilder::buildPointerType(kaapi_splitter_context_ROSE_type)
   ),
   newbbcall
  );
  SageInterface::appendStatement(splitter_context, newbbcall);

  // __kaapi_context = __splitter_context->data;
  SgVariableDeclaration* local_context = SageBuilder::buildVariableDeclaration (
      "__kaapi_context",
      SageBuilder::buildPointerType(contexttype->get_type()),
      SageBuilder::buildAssignInitializer(
        SageBuilder::buildCastExp(
	  SageBuilder::buildOpaqueVarRefExp("__splitter_context->data", body),
          SageBuilder::buildPointerType(contexttype->get_type())
	  )
      ),
      newbbcall
  );
  SageInterface::appendStatement( local_context, newbbcall );

  // initialize __splitter_context fields

  // kaapi_workqueue_init(&__splitter_context->wq, i, j);
  {
    SgExpression* low_expr = begin_iter;
    SgExpression* hi_expr = end_iter;
    if (hasIncrementalIterationSpace == false)
    {
      low_expr = end_iter;
      hi_expr = begin_iter;
    }

    SgExpression* size_expr;
    if (isInclusiveUpperBound)
    {
      size_expr = SageBuilder::buildSubtractOp
	(
	 SageBuilder::buildAddOp(hi_expr, SageBuilder::buildLongIntVal(1)),
	 low_expr
        );
    }
    else 
    {
      size_expr = SageBuilder::buildSubtractOp(hi_expr, low_expr);
    }

    SgExprListExp* arg_list = SageBuilder::buildExprListExp
      (
       SageBuilder::buildAddressOfOp
       (SageBuilder::buildOpaqueVarRefExp("__splitter_context->wq", body)),
       SageBuilder::buildIntVal(0),
       size_expr
      );

    SgExprStatement* call_stmt = SageBuilder::buildFunctionCallStmt
      (
       "kaapi_workqueue_init",
       SageBuilder::buildVoidType(), 
       arg_list,
       scope
      );
    call_stmt->set_endOfConstruct(SOURCE_POSITION);
    SageInterface::appendStatement(call_stmt, newbbcall);
  }

  // __splitter_context->body = loopentry;
  {
    SgExprStatement* assign_stmt = SageBuilder::buildAssignStatement
      (
       SageBuilder::buildOpaqueVarRefExp
       ("__splitter_context->body", body),
       SageBuilder::buildCastExp
       (
	SageBuilder::buildAddressOfOp
	(SageBuilder::buildFunctionRefExp(entrypoint)),
	SageBuilder::buildPointerType(kaapi_task_body_ROSE_type)
       )
      );
    assign_stmt->set_endOfConstruct(SOURCE_POSITION);
    SageInterface::appendStatement(assign_stmt, newbbcall);
  }

  // __splitter_context->ktr_size = sizeof(type)
  {
    SgExprStatement* assign_stmt;
    if (kta->hasReduction())
    {
      // get the corresponding ktr type
      SgClassDeclaration* const type_decl =
	kta->buildInsertClassDeclaration(bigscope, forloop, this_class_def);
      if (type_decl == NULL) KaapiAbort("type_decl == NULL");

      assign_stmt = SageBuilder::buildAssignStatement
      (
       SageBuilder::buildOpaqueVarRefExp
       ("__splitter_context->ktr_size", body),
       SageBuilder::buildSizeOfOp(type_decl->get_type())
      );
    }
    else
    {
      assign_stmt = SageBuilder::buildAssignStatement
      (
       SageBuilder::buildOpaqueVarRefExp
       ("__splitter_context->ktr_size", body),
       SageBuilder::buildUnsignedLongVal(0)
      );
    }
    assign_stmt->set_endOfConstruct(SOURCE_POSITION);
    SageInterface::appendStatement(assign_stmt, newbbcall);
  }

  // __splitter_context->data_size = sizeof(type)
  {
    SgExprStatement* assign_stmt = SageBuilder::buildAssignStatement
      (
       SageBuilder::buildOpaqueVarRefExp
       ("__splitter_context->data_size", body),
       SageBuilder::buildSizeOfOp(contexttype->get_type())
      );
    assign_stmt->set_endOfConstruct(SOURCE_POSITION);
    SageInterface::appendStatement(assign_stmt, newbbcall);
  }

  /* Generate assignment of the of free variable to the context field */
  ivar_beg = listvar.begin();
  ivar_end = listvar.end();
  while (ivar_beg != ivar_end)
  {
    if ((*ivar_beg)->get_name() == ivar->get_name())
    {
      /* to nothing */
    }
    else if ((*ivar_beg)->get_name() == "__kaapi_thread")
    {
      /* to nothing */
    }
    else {
      SgExprStatement* exrpassign = SageBuilder::buildExprStatement(
        SageBuilder::buildAssignOp(
          SageBuilder::buildArrowExp( 
            SageBuilder::buildVarRefExp(local_context),
            SageBuilder::buildOpaqueVarRefExp
	    ("p_" + (*ivar_beg)->get_name(), contexttype->get_scope())
          ),
          SageBuilder::buildAddressOfOp(
            SageBuilder::buildVarRefExp(
              (*ivar_beg)->get_name(),
              SageInterface::getScope( forloop )
            )
          )
        )
      );
      exrpassign->set_endOfConstruct(SOURCE_POSITION);
      SageInterface::appendStatement( exrpassign, newbbcall );
    }
    ++ivar_beg;
  }

  // if needed, affect p_this to this
  if (this_class_def != NULL)
  {
    SgExprStatement* const assign_stmt = SageBuilder::buildExprStatement
      (
       SageBuilder::buildAssignOp
       (
	SageBuilder::buildArrowExp
	(
	 SageBuilder::buildVarRefExp(local_context),
	 SageBuilder::buildOpaqueVarRefExp("p_this", contexttype->get_scope())
	),

	SageBuilder::buildOpaqueVarRefExp
	("this", SageInterface::getScope(forloop))
        )
      );

    assign_stmt->set_endOfConstruct(SOURCE_POSITION);
    SageInterface::appendStatement(assign_stmt, newbbcall);
  }

  // call the entrypoint
  SgExprListExp* const callArgs = SageBuilder::buildExprListExp();
  callArgs->append_expression
    (
      SageBuilder::buildCastExp
      (
       SageBuilder::buildVarRefExp(splitter_context),
       SageBuilder::buildPointerType(SageBuilder::buildVoidType())
      )
    );

  callArgs->append_expression(SageBuilder::buildVarRefExp(newkaapi_threadvar));

  if (kta->hasReduction())
  {
    SgExpression* const null_expr = SageBuilder::buildCastExp
      (
       SageBuilder::buildUnsignedLongVal(0),
       SageBuilder::buildPointerType(SageBuilder::buildVoidType())
      );

    callArgs->append_expression(null_expr);
  }

  SgExprStatement* callStmt = SageBuilder::buildFunctionCallStmt
    (SageBuilder::buildFunctionRefExp(entrypoint), callArgs);

  callStmt->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( callStmt, newbbcall );

#if 0 //////////////////////////////////////////
{

  /* 
   * Generate the body of the entry point 
   */
  SgExpression* work = 
    SageBuilder::buildAddressOfOp(
      SageBuilder::buildArrowExp( 
        SageBuilder::buildVarRefExp(local_context),
        SageBuilder::buildOpaqueVarRefExp("__kaapi_work", contexttype->get_scope())
      )
    );

  SgVariableSymbol* local_ivar = entrypoint->get_definition()->get_body()->lookup_variable_symbol(ivar->get_name());

  SgVariableDeclaration* local_ivar_end = SageBuilder::buildVariableDeclaration (
      "__kaapi_iter_end",
      ivar->get_type(),
      0,
      body
  );
  local_ivar_end->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( local_ivar_end, body );
    
  SgVariableDeclaration* wc = SageBuilder::buildVariableDeclaration (
      "__kaapi_wc",
      SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type),
      0, 
      body
  );
  wc->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( wc, body );
  
  /* size wq */
  SgExpression* exp_size = 0;
  if (isInclusiveUpperBound)
  {
    exp_size = SageBuilder::buildDivideOp(
          SageBuilder::buildSubtractOp(
            SageBuilder::buildAddOp(
              end_iter,
              SageBuilder::buildLongIntVal(1)
            ),
            begin_iter
          ),
          step
        );
  }
  else 
  {
    exp_size = SageBuilder::buildDivideOp(
          SageBuilder::buildSubtractOp(
            end_iter,
            begin_iter
          ),
          step
        );
  }

  /* */
  SgVariableSymbol* local_beg = entrypoint->get_definition()->lookup_variable_symbol("__kaapi_range_beg");
  SgVariableSymbol* local_end = entrypoint->get_definition()->lookup_variable_symbol("__kaapi_range_end");


  SgVariableDeclaration* popsize = SageBuilder::buildVariableDeclaration (
      "__kaapi_popsize",
      SageBuilder::buildLongType(),
      SageBuilder::buildAssignInitializer(
        SageBuilder::buildDivideOp(
          SageBuilder::buildSubtractOp(
            SageBuilder::buildVarRefExp(local_end),
            SageBuilder::buildVarRefExp(local_beg)
          ),
          SageBuilder::buildMultiplyOp(
            SageBuilder::buildIntVal(42),
            SageBuilder::buildFunctionCallExp(    
              "kaapi_getconcurrency",
              SageBuilder::buildIntType(), 
              SageBuilder::buildExprListExp(),
              body
            )
          )
        )
      ),
      body
  );
  popsize->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( popsize, body );

  /* generate assignment of the local_context field */
  ivar_beg = listvar.begin();
  ivar_end = listvar.end();
  while (ivar_beg != ivar_end)
  {
    if ((*ivar_beg)->get_name() == ivar->get_name())
    {
      /* to nothing */
    }
    else if ((*ivar_beg)->get_name() == "__kaapi_thread")
    {
      /* to nothing */
    }
    else {
      SgExprStatement* exrpassign = SageBuilder::buildExprStatement(
        SageBuilder::buildAssignOp(
          SageBuilder::buildDotExp( 
            SageBuilder::buildVarRefExp(local_context),
            SageBuilder::buildOpaqueVarRefExp("p_" + (*ivar_beg)->get_name(), contexttype->get_scope())
          ),
          SageBuilder::buildAddressOfOp(
            SageBuilder::buildVarRefExp(
              (*ivar_beg)->get_name(),
              SageInterface::getScope( forloop )
            )
          )
        )
      );
      exrpassign->set_endOfConstruct(SOURCE_POSITION);
      SageInterface::appendStatement( exrpassign, bb );
    }
    ++ivar_beg;
  }

  /* begin adaptive section */
  SgExprStatement* wc_init = SageBuilder::buildExprStatement(
    SageBuilder::buildAssignOp(
      SageBuilder::buildVarRefExp(wc),
      SageBuilder::buildFunctionCallExp(    
        "kaapi_task_begin_adaptive",
        SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type), 
        SageBuilder::buildExprListExp(
          SageBuilder::buildVarRefExp(newkaapi_threadvar),
#if 0
          SageBuilder::buildIntVal(0), /* flag !! */
#else
          SageBuilder::buildBitOrOp(
            SageBuilder::buildOpaqueVarRefExp( "KAAPI_SC_CONCURRENT", bb),
            SageBuilder::buildOpaqueVarRefExp( "KAAPI_SC_NOPREEMPTION", bb)
          ),
#endif
          SageBuilder::buildAddressOfOp(SageBuilder::buildFunctionRefExp(splitter)),
          SageBuilder::buildAddressOfOp( SageBuilder::buildVarRefExp(splitter_context) )
        ),
        bb
      )
    )
  );
  wc_init->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( wc_init, bb );
  
  SgStatement* init_ivar_local =
    SageBuilder::buildAssignStatement(
      SageBuilder::buildVarRefExp(local_ivar),
      SageBuilder::buildAddOp(
        begin_iter,
        SageBuilder::buildMultiplyOp(
          SageBuilder::buildVarRefExp(local_beg),
          step
        )
      )
    );
  init_ivar_local->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( init_ivar_local, bb );

  SgWhileStmt* while_popsizeloop = 
    SageBuilder::buildWhileStmt(
      /* !kaapi_workqueue_pop() */
      SageBuilder::buildNotOp(
        SageBuilder::buildFunctionCallExp(    
          "kaapi_workqueue_pop",
          SageBuilder::buildIntType(), 
          SageBuilder::buildExprListExp(
            work,
            SageBuilder::buildAddressOfOp(SageBuilder::buildVarRefExp(local_beg)),
            SageBuilder::buildAddressOfOp(SageBuilder::buildVarRefExp(local_end)),
            SageBuilder::buildVarRefExp( popsize )
          ),
          bb
        )
      ),
      SageBuilder::buildBasicBlock( )
  );
  while_popsizeloop->set_endOfConstruct(SOURCE_POSITION);

  /* iter_end = affine function of range_end */
  SgStatement* init_ivar_local_end =
    SageBuilder::buildAssignStatement(
      SageBuilder::buildVarRefExp(local_ivar_end),
      SageBuilder::buildAddOp(
        begin_iter,
        SageBuilder::buildMultiplyOp(
          SageBuilder::buildVarRefExp(local_end),
          step
        )
      )
    );
  init_ivar_local_end->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( init_ivar_local_end, isSgBasicBlock(while_popsizeloop->get_body( )) );
  
  SgStatement* new_forloop = 
    SageBuilder::buildForStatement(
      0,
      SageBuilder::buildExprStatement(
        SageBuilder::buildLessThanOp(
          SageBuilder::buildVarRefExp(local_ivar),
          SageBuilder::buildVarRefExp(local_ivar_end)
        )
      ), 
      SageBuilder::buildPlusAssignOp(
        SageBuilder::buildVarRefExp(local_ivar),
        step
      ),
      loopbody
    );
  new_forloop->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( new_forloop, isSgBasicBlock(while_popsizeloop->get_body( ))  );

  /* iter = iter_end at the end of the local loop */
  SgStatement* swap_ivars_local =
    SageBuilder::buildAssignStatement(
      SageBuilder::buildVarRefExp(local_ivar),
      SageBuilder::buildVarRefExp(local_ivar_end)
    );
  swap_ivars_local->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( swap_ivars_local, isSgBasicBlock(while_popsizeloop->get_body( )) );

  SageInterface::appendStatement( while_popsizeloop, bb  );
  
  /* end_adaptive
  */
  SgExprStatement* wc_term = SageBuilder::buildFunctionCallStmt(    
      "kaapi_task_end_adaptive",
      SageBuilder::buildVoidType(), 
      SageBuilder::buildExprListExp(
        SageBuilder::buildVarRefExp(wc)
      ),
      bb
  );
  wc_term->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( wc_term, bb );
}
#endif //////////////////////////////////////////


  /* 
  */
  SageInterface::movePreprocessingInfo( forloop, entrypoint );
  SageInterface::movePreprocessingInfo( forloop, splitter );

  /* append the new function declaration at the end of the scope */
  if (this_class_def == NULL)
    SageInterface::appendStatement ( entrypoint, bigscope );

  // a test of the loop condition must be done since
  // the generated iterator may be positive even if
  // the loop condition is not initially met.
  SgIfStmt* const if_stmt = SageBuilder::buildIfStmt
  (
   isSgStatement(SageBuilder::buildExprStatement(saved_test_expr)),
   isSgStatement(newbbcall),
   NULL
  );
  return isSgStatement(if_stmt);
}


/** Generate the function declaration for the loop entrypoint to execute
    loop iteration over a range.
    The signature of the function is
      void entrypoint( context_type*, kaapi_thread*)
    such that it could be used as task body.
    
    The body of the function only contains
*/

static SgFunctionDeclaration* buildLoopMethodTrampoline
(
 SgClassDefinition* this_class_def,
 SgClassDeclaration* args_decl,
 SgFunctionDeclaration* called_decl
)
{
  // generate a method trampoline of the form:
  // static __kaapi_loop_entrypoint_0(void* p, kaapi_thread_t* t)
  // (TT*((T*)p)->data)->called_decl_name(p, t);

  SgFunctionParameterList* const param_list =
    SageBuilder::buildFunctionParameterList();

  SageInterface::appendArg
    (
     param_list,
     SageBuilder::buildInitializedName
     (
      "splitter_context",
      // SageBuilder::buildPointerType(kaapi_splitter_context_ROSE_type)
      SageBuilder::buildPointerType(SageBuilder::buildVoidType())
     )
    );

  SageInterface::appendArg
    (
     param_list, 
     SageBuilder::buildInitializedName
     (
      "thread",
      SageBuilder::buildPointerType(kaapi_thread_ROSE_type)
     )
    );

  SgName tramp_suffix("_tramp");
  SgName tramp_name(called_decl->get_name() + tramp_suffix);

  SgFunctionDeclaration* tramp_decl =
    SageBuilder::buildDefiningFunctionDeclaration
    (
     tramp_name,
     SageBuilder::buildVoidType(),
     param_list,
     this_class_def
    );
  tramp_decl->get_declarationModifier().get_storageModifier().setStatic();
  tramp_decl->set_endOfConstruct(SOURCE_POSITION);
  this_class_def->append_member(tramp_decl);

  SgBasicBlock* const tramp_body = tramp_decl->get_definition()->get_body();
  SgScopeStatement* const tramp_scope = SageInterface::getScope(tramp_body);
  // assume(tramp_body);

  // implement the method body as defined above
  // ((TT*)((T*)fu)->bar)->p_this->called_name(fu, thread);

  SgCastExp* const data_expr = SageBuilder::buildCastExp
    (
     SageBuilder::buildOpaqueVarRefExp
     ("((kaapi_splitter_context_t*)splitter_context)->data", tramp_scope),
     SageBuilder::buildPointerType(args_decl->get_type())
    );

  SgCastExp* const this_expr = SageBuilder::buildCastExp
    (
     SageBuilder::buildArrowExp
     (
      data_expr,
      SageBuilder::buildOpaqueVarRefExp("p_this", this_class_def)
     ),
     SageBuilder::buildPointerType
     (this_class_def->get_declaration()->get_type())
    );

  this_expr->set_need_paren(true);

#if 0
  SgExpression* const call_expr = SageBuilder::buildFunctionCallExp
    (
     SageBuilder::buildArrowExp
     (this_expr, SageBuilder::buildFunctionRefExp(called_decl)),
     SageBuilder::buildExprListExp
     (
      SageBuilder::buildVarRefExp("splitter_context", tramp_scope),
      SageBuilder::buildVarRefExp("thread", tramp_scope)
     )
    );
  SgExprStatement* const call_stmt =
    SageBuilder::buildFunctionCallStmt(call_expr);
#else
  std::string call_string;
  call_string.append("(");
  call_string.append(this_expr->unparseToString());
  call_string.append(")");
  call_string.append("->");
  call_string.append(called_decl->get_name().str());

  SgExprStatement* const call_stmt = SageBuilder::buildFunctionCallStmt
    (
     call_string,
     SageBuilder::buildVoidType(),
     SageBuilder::buildExprListExp
     (
      SageBuilder::buildOpaqueVarRefExp("splitter_context", tramp_scope),
      SageBuilder::buildOpaqueVarRefExp("thread", tramp_scope)
     ),
     tramp_scope
   );
#endif

  SageInterface::appendStatement(call_stmt, tramp_body);

  return tramp_decl;
}

static void buildLoopEntrypoint
(
 SgClassDefinition* this_class_def,
 SgClassDeclaration* contexttype,
 SgGlobal* scope,
 SgFunctionDeclaration*& partial_decl,
 SgFunctionDeclaration*& tramp_decl,
 KaapiTaskAttribute* kta
)
{
  // partial_decl is the function to be build
  // tramp_decl is the function to be called, which can be
  // the same as partial_decl if this is not a method

  static int cnt = 0;
  std::ostringstream func_name;
  func_name << "__kaapi_loop_entrypoint_" << cnt++;

  SgType *return_type = SageBuilder::buildVoidType();
  SgFunctionParameterList *parlist;

  /* Generate the parameter list */
  parlist = SageBuilder::buildFunctionParameterList();
  
  /* append 2 declarations : context & thread */
  SageInterface::appendArg
    (
     parlist, 
     SageBuilder::buildInitializedName
     (
      "__kaapi_vcontext", 
      SageBuilder::buildPointerType(SageBuilder::buildVoidType())
     )
    );

  SageInterface::appendArg
    (
     parlist, 
     SageBuilder::buildInitializedName
     (
      "__kaapi_thread", 
      SageBuilder::buildPointerType(kaapi_thread_ROSE_type)
     )
    );

  if (kta->hasReduction())
  {
    SageInterface::appendArg
      (
       parlist, 
       SageBuilder::buildInitializedName
       (
	"__kaapi_sc",
	SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type)
       )
      );
  }

  // loop body must be a method. generate a trampoline routine.
  if (this_class_def != NULL)
  {
    partial_decl = SageBuilder::buildDefiningMemberFunctionDeclaration
      (func_name.str(), return_type, parlist, this_class_def);
    partial_decl->set_endOfConstruct(SOURCE_POSITION);
    this_class_def->append_member(partial_decl);

    tramp_decl = buildLoopMethodTrampoline
      (this_class_def, contexttype, partial_decl);
  }
  else
  {
    tramp_decl = SageBuilder::buildDefiningFunctionDeclaration
      (func_name.str(), return_type, parlist, scope);
    tramp_decl->get_declarationModifier().get_storageModifier().setStatic();
    tramp_decl->set_endOfConstruct(SOURCE_POSITION);
    partial_decl = tramp_decl;
  }
}


static void buildLoopEntrypointBody( 
  SgFunctionDeclaration*       func,
  SgFunctionDeclaration*       splitter,
  std::set<SgVariableSymbol*>& listvar,
  SgClassDeclaration*          contexttype,
  SgInitializedName*           ivar,
  SgExpression*                global_begin_iter,
  SgExpression*                global_step,
  SgStatement*                 loopbody,
  SgGlobal*                    scope,
  bool			       hasIncrementalIterationSpace,
  bool			       isInclusiveUpperBound,
  forLoopCanonicalizer::AffineVariableList& affine_vars,
  KaapiTaskAttribute*	       kta
)
{
  SgBasicBlock* body = func->get_definition()->get_body();
  
  SgVariableSymbol* newkaapi_threadvar = 
      func->get_definition()->lookup_variable_symbol("__kaapi_thread");

  /* cast the void* vcontext to the true type */
  SgVariableDeclaration* splitter_context = SageBuilder::buildVariableDeclaration (
      "__splitter_context",
      SageBuilder::buildPointerType(kaapi_splitter_context_ROSE_type),
      SageBuilder::buildAssignInitializer(
        SageBuilder::buildCastExp(
          SageBuilder::buildVarRefExp(func->get_definition()->lookup_variable_symbol("__kaapi_vcontext")), 
          SageBuilder::buildPointerType(kaapi_splitter_context_ROSE_type)
        )
      ),
      body
  );

  SageInterface::appendStatement( splitter_context, body );
  
  SgVariableDeclaration* context = SageBuilder::buildVariableDeclaration (
      "__kaapi_context",
      SageBuilder::buildPointerType(contexttype->get_type()),
      SageBuilder::buildAssignInitializer(
        SageBuilder::buildCastExp(
	  SageBuilder::buildOpaqueVarRefExp("__splitter_context->data", body),
          SageBuilder::buildPointerType(contexttype->get_type())
	  )
      ),
      body
  );
  SageInterface::appendStatement( context, body );

  // __kaapi_result = __kaapi_sc ? kaapi_adaptive_result_data(__kaapi_sc) : 0
  SgVariableDeclaration* result_data = NULL;
  if (kta->hasReduction() == true)
  {
    SgClassDeclaration* const class_decl =
      kta->buildInsertClassDeclaration(scope, isSgScopeStatement(loopbody));
    if (class_decl == NULL) KaapiAbort("class_decl == NULL");

    SgExpression* const null_expr = SageBuilder::buildCastExp
      (
       SageBuilder::buildUnsignedLongVal(0),
       SageBuilder::buildPointerType(SageBuilder::buildVoidType())
      );

    SgExpression* const noteq_expr = SageBuilder::buildNotEqualOp
      (SageBuilder::buildOpaqueVarRefExp("__kaapi_sc", body), null_expr);

    SgFunctionCallExp* const call_expr = SageBuilder::buildFunctionCallExp
      (
       "kaapi_adaptive_result_data",
       SageBuilder::buildPointerType(SageBuilder::buildVoidType()),
       SageBuilder::buildExprListExp
       (SageBuilder::buildOpaqueVarRefExp("__kaapi_sc", body)),
       body
      );

    SgConditionalExp* const tern_expr =
      SageBuilder::buildConditionalExp(noteq_expr, call_expr, null_expr);

    result_data = SageBuilder::buildVariableDeclaration
    (
      "__kaapi_result",
      SageBuilder::buildPointerType(class_decl->get_type()),
      SageBuilder::buildAssignInitializer
      (
       SageBuilder::buildCastExp
       (tern_expr, SageBuilder::buildPointerType(class_decl->get_type()))
      ),
      body
    );
    SageInterface::appendStatement(result_data, body);
  }

  /* 
   * Generate the body of the entry point 
   */
  SgExpression* work = SageBuilder::buildAddressOfOp
    (
     SageBuilder::buildArrowExp
     ( 
      SageBuilder::buildVarRefExp(splitter_context),
      SageBuilder::buildOpaqueVarRefExp("wq", body)
     )
    );

  SgVariableDeclaration* local_ivar_end = SageBuilder::buildVariableDeclaration (
      "__kaapi_iter_end",
      ivar->get_type(),
      0,
      body
  );
  local_ivar_end->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( local_ivar_end, body );
    
  SgVariableDeclaration* wc = SageBuilder::buildVariableDeclaration (
      "__kaapi_wc",
      SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type),
      0, 
      body
  );
  wc->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( wc, body );
  
  /* the queue is already initilized by the callee */
  SgVariableDeclaration* local_beg = SageBuilder::buildVariableDeclaration (
      "__kaapi_range_beg",
      kaapi_workqueue_index_ROSE_type,
      0, 
      body
  );

  local_beg->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( local_beg, body );

  /* */
  SgVariableDeclaration* local_end = SageBuilder::buildVariableDeclaration (
      "__kaapi_range_end",
      kaapi_workqueue_index_ROSE_type,
      0, 
      body
  );
  local_end->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( local_end, body );
  
  SgVariableDeclaration* popsize = SageBuilder::buildVariableDeclaration (
      "__kaapi_popsize",
      SageBuilder::buildLongType(),
      SageBuilder::buildAssignInitializer(

        SageBuilder::buildAddOp(
        SageBuilder::buildLongIntVal(1),

        SageBuilder::buildDivideOp(
          SageBuilder::buildFunctionCallExp(    
            "kaapi_workqueue_size",
            SageBuilder::buildIntType(), 
            SageBuilder::buildExprListExp(work),
            body
          ),
          SageBuilder::buildMultiplyOp(
            SageBuilder::buildIntVal(4),
            SageBuilder::buildFunctionCallExp(    
              "kaapi_getconcurrency",
              SageBuilder::buildIntType(), 
              SageBuilder::buildExprListExp(),
              body
            )
          )
         )
        )
      ),
      body
  );
  popsize->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( popsize, body );

  /* Declare and assignment of the variable with the same name as in the free variable of the loop */
  std::set<SgVariableSymbol*>::iterator ivar_beg = listvar.begin();
  std::set<SgVariableSymbol*>::iterator ivar_end = listvar.end();
  while (ivar_beg != ivar_end)
  {
    if ((*ivar_beg)->get_name() == ivar->get_name())
    { /* to nothing */ }
    else if ((*ivar_beg)->get_name() == "__kaapi_thread")
    { /* to nothing */ }
    else {
      SgVariableDeclaration* alocalvar;

      // dont assign if affine, since it is done at each pop
      typedef forLoopCanonicalizer::AffineVariable AffineVariable;
      std::list<AffineVariable>::iterator pos = affine_vars.begin();
      std::list<AffineVariable>::iterator end = affine_vars.end();
      for (; pos != end; ++pos)
      {
	if (pos->name_->get_name() == (*ivar_beg)->get_name())
	  break ;
      }

      // not found, declare and assign
      if (pos == end)
      {
	alocalvar = SageBuilder::buildVariableDeclaration (
          (*ivar_beg)->get_name(),
          (*ivar_beg)->get_type(),
          SageBuilder::buildAssignInitializer(
            SageBuilder::buildPointerDerefExp(
              SageBuilder::buildArrowExp( 
                SageBuilder::buildVarRefExp(context),
                SageBuilder::buildOpaqueVarRefExp
		("p_" + (*ivar_beg)->get_name(), body)
              )
            )
          ),
          body
	 );
      }
      else // found, just declare
      {
	alocalvar = SageBuilder::buildVariableDeclaration (
          (*ivar_beg)->get_name(),
          (*ivar_beg)->get_type(),
	  0,
	  body
        );
      }

      SageInterface::appendStatement( alocalvar, body );
    }

    ++ivar_beg;
  }

  /* begin adaptive section */
  SgBinaryOp* flags_op;
  if (kta->hasReduction() == true)
  {
    flags_op = SageBuilder::buildBitOrOp
      (
       SageBuilder::buildOpaqueVarRefExp( "KAAPI_SC_CONCURRENT", body),
       SageBuilder::buildOpaqueVarRefExp( "KAAPI_SC_PREEMPTION", body)
      );
  }
  else
  {
    flags_op = SageBuilder::buildBitOrOp
      (
       SageBuilder::buildOpaqueVarRefExp( "KAAPI_SC_CONCURRENT", body),
       SageBuilder::buildOpaqueVarRefExp( "KAAPI_SC_NOPREEMPTION", body)
      );
  }

  SgExprStatement* wc_init = SageBuilder::buildExprStatement(
    SageBuilder::buildAssignOp(
      SageBuilder::buildVarRefExp(wc),
      SageBuilder::buildFunctionCallExp(    
        "kaapi_task_begin_adaptive",
        SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type), 
        SageBuilder::buildExprListExp(
          SageBuilder::buildVarRefExp(newkaapi_threadvar),
	  flags_op,
          SageBuilder::buildAddressOfOp(SageBuilder::buildFunctionRefExp(splitter)),
          SageBuilder::buildVarRefExp(splitter_context)
        ),
        body
      )
    )
  );
  wc_init->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( wc_init, body );

  SgWhileStmt* while_popsizeloop = 
    SageBuilder::buildWhileStmt(
      /* !kaapi_workqueue_pop() */
      SageBuilder::buildNotOp(
        SageBuilder::buildFunctionCallExp(    
          "kaapi_workqueue_pop",
          SageBuilder::buildIntType(), 
          SageBuilder::buildExprListExp(
            work,
            SageBuilder::buildAddressOfOp(SageBuilder::buildVarRefExp(local_beg)),
            SageBuilder::buildAddressOfOp(SageBuilder::buildVarRefExp(local_end)),
            SageBuilder::buildVarRefExp( popsize )
          ),
          body
        )
      ),
      SageBuilder::buildBasicBlock( )
  );
  while_popsizeloop->set_endOfConstruct(SOURCE_POSITION);

  { // affine variable affectation

    typedef forLoopCanonicalizer::AffineVariable AffineVariable;

    SgBasicBlock* const while_block =
      isSgBasicBlock(while_popsizeloop->get_body());

    SgVarRefExp* const beg_expr = SageBuilder::buildVarRefExp(local_beg);

    std::list<AffineVariable>::iterator pos = affine_vars.begin();
    std::list<AffineVariable>::iterator end = affine_vars.end();
    for (; pos != end; ++pos)
    {
      SgVariableSymbol* const var_sym =
	body->lookup_variable_symbol(pos->name_->get_name());

      // TO_REMOVE cannot occur
      if (var_sym == NULL)
      {
	printf("var_sym(%s) == NULL\n", pos->name_->get_name().str());
	continue ;
      }
      // TO_REMOVE cannot occur

      // create the var affectation statement:
      // var = __c->var + __i * incr;

      SgPointerDerefExp* const base_expr =
	SageBuilder::buildPointerDerefExp
	(
	 SageBuilder::buildArrowExp
	 ( 
	  SageBuilder::buildVarRefExp(context),
	  SageBuilder::buildOpaqueVarRefExp
	  ("p_" + var_sym->get_name(), body)
	 )
	);

      SgExpression* incr_expr = pos->incr_;

      // plus_plus or minus_minus have no expression
      if (incr_expr == NULL)
	incr_expr = SageBuilder::buildLongIntVal(1);

      SgExpression* const scaled_expr =
	SageBuilder::buildMultiplyOp(beg_expr, incr_expr);

      SgExpression* add_expr;
      // decreasing op
      if (forLoopCanonicalizer::isDecreasingOp(pos->op_))
	add_expr = SageBuilder::buildSubtractOp(base_expr, scaled_expr);
      else
	add_expr = SageBuilder::buildAddOp(base_expr, scaled_expr);

      SgStatement* const assign_stmt = SageBuilder::buildAssignStatement
	(SageBuilder::buildVarRefExp(var_sym), add_expr);
      assign_stmt->set_endOfConstruct(SOURCE_POSITION);
      SageInterface::prependStatement(assign_stmt, while_block);

    } // foreach affine variable

  } // affine variable affectation

  /* */
  SgStatement* new_forloop = 
    SageBuilder::buildForStatement(
      0,
      SageBuilder::buildExprStatement(
        SageBuilder::buildNotEqualOp(
          SageBuilder::buildVarRefExp(local_beg),
          SageBuilder::buildVarRefExp(local_end)
        )
      ), 
      SageBuilder::buildPlusPlusOp(
        SageBuilder::buildVarRefExp(local_beg)
      ),
      loopbody
    );
  new_forloop->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement( new_forloop, isSgBasicBlock(while_popsizeloop->get_body( ))  );

  SageInterface::appendStatement( while_popsizeloop, body );

  if (kta->hasReduction())
  {
    // there is reduction. if we are the master, wait for thieves
    // to terminate and reduce their result. When there is no more
    // thief, update the original variables.
    // if we are a slave, update the ktr.

    SgExpression* const null_expr = SageBuilder::buildCastExp
      (
       SageBuilder::buildUnsignedLongVal(0),
       SageBuilder::buildPointerType(SageBuilder::buildVoidType())
      );

    // while ((ktr = kaapi_get_thief(sc)) != NULL) kaapi_preempt_thief();

    SgVariableDeclaration* const ktr_decl = SageBuilder::buildVariableDeclaration
      (
       "__kaapi_ktr",
       SageBuilder::buildPointerType(kaapi_taskadaptive_result_ROSE_type),
       0, 
       isSgScopeStatement(body)
      );
    body->prepend_statement(ktr_decl);

    // we cannot use the original variable for reduction
    // since they may be used by the thieves during. Thus,
    // we create a second temporary __kaapi_context. This
    // context member point to the master local variable,
    // on which the reduction is made. when all the thieves
    // have been preempted, do original variable update.

    SgVariableDeclaration* const tmp_context =
      SageBuilder::buildVariableDeclaration
      (
       "__kaapi_tmp_context",
       contexttype->get_type(),
       0,
       body
      );
    body->prepend_statement(tmp_context);

    // build tmp_context
    {
      std::set<SgVariableSymbol*> symbol_set;
      kta->buildReductionSet(symbol_set);

      std::set<SgVariableSymbol*>::const_iterator pos = symbol_set.begin();
      std::set<SgVariableSymbol*>::const_iterator end = symbol_set.end();
      for (; pos != end; ++pos)
      {
	// __kaapi_tmp_context.p_xxx = &xxx;

	std::string lhs_string;
	lhs_string.append("__kaapi_tmp_context.p_");
	lhs_string.append((*pos)->get_name().str());

	std::string rhs_string;
	rhs_string.append((*pos)->get_name().str());

	SgExprStatement* const assign_stmt = SageBuilder::buildAssignStatement
	  (
	   SageBuilder::buildVarRefExp(lhs_string, body),
	   SageBuilder::buildAddressOfOp
	   (SageBuilder::buildVarRefExp(rhs_string, body))
	  );
	body->append_statement(assign_stmt);
      }
    } // build tmp_context

    // preempt and reduce thieves
    SgVarRefExp* const ktr_expr = SageBuilder::buildVarRefExp(ktr_decl);
    SgExpression* const getthief_expr = SageBuilder::buildFunctionCallExp
      (    
       "kaapi_get_thief_head",
       SageBuilder::buildPointerType(kaapi_taskadaptive_result_ROSE_type),
       SageBuilder::buildExprListExp(SageBuilder::buildVarRefExp(wc)),
       scope
      );

    SgExprStatement* const assign_stmt =
      SageBuilder::buildAssignStatement(ktr_expr, getthief_expr);

    SgNotEqualOp* cond_stmt = SageBuilder::buildNotEqualOp
      (assign_stmt->get_expression(), null_expr);

    SgBasicBlock* const while_bb = SageBuilder::buildBasicBlock();

    SgFunctionDeclaration* const reducer_decl = kta->buildInsertReducer
      (contexttype->get_type(), kta->class_decl->get_type(), scope);

    // kaapi_preempt_thief(sc, ktr, NULL, reducer, (void*)&work);
    SgExprStatement* const preempt_stmt = SageBuilder::buildFunctionCallStmt
      (
       "kaapi_preempt_thief",
       SageBuilder::buildIntType(),
       SageBuilder::buildExprListExp
       (
	SageBuilder::buildVarRefExp(wc),
	ktr_expr,
	null_expr,
	SageBuilder::buildFunctionRefExp(reducer_decl),
	SageBuilder::buildCastExp
	(
	 SageBuilder::buildAddressOfOp
	 (SageBuilder::buildVarRefExp(tmp_context)),
	 SageBuilder::buildPointerType(SageBuilder::buildVoidType())
        )
       ),
       scope
      );
    while_bb->append_statement(preempt_stmt);

    SgWhileStmt* const while_stmt =
      SageBuilder::buildWhileStmt(cond_stmt, isSgStatement(while_bb));
    body->append_statement(while_stmt);

    // if (__kaapi_result != NULL)

    cond_stmt = SageBuilder::buildNotEqualOp
      (SageBuilder::buildVarRefExp(result_data), null_expr);

    // true statment: thief, update data
    SgBasicBlock* const true_bb = SageBuilder::buildBasicBlock();
    {
      std::set<SgVariableSymbol*> symbol_set;
      kta->buildReductionSet(symbol_set);

      std::set<SgVariableSymbol*>::const_iterator pos = symbol_set.begin();
      std::set<SgVariableSymbol*>::const_iterator end = symbol_set.end();
      for (; pos != end; ++pos)
      {
	SgVarRefExp* const ref_expr =
	  SageBuilder::buildOpaqueVarRefExp((*pos)->get_name(), body);

	SgArrowExp* const arrow_expr = SageBuilder::buildArrowExp
	  (SageBuilder::buildVarRefExp(result_data), ref_expr);

	SgExprStatement* const assign_stmt =
	  SageBuilder::buildAssignStatement(arrow_expr, ref_expr);

	true_bb->append_statement(assign_stmt);
      }
    }

    // false statment, master, update variables with our local result
    // FIXME: we should reduce, not assign
    // TODO: should update output too.
    SgBasicBlock* const false_bb = SageBuilder::buildBasicBlock();
    {
      // update the original values

      std::set<SgVariableSymbol*> symbol_set;
      kta->buildReductionSet(symbol_set);

      std::set<SgVariableSymbol*>::const_iterator pos = symbol_set.begin();
      std::set<SgVariableSymbol*>::const_iterator end = symbol_set.end();

      for (; pos != end; ++pos)
      {
	// *__kaapi_context->p_xxx = xxx;

	std::string lhs_string;
	lhs_string.append("*__kaapi_context->p_");
	lhs_string.append((*pos)->get_name().str());

	std::string rhs_string;
	rhs_string.append((*pos)->get_name().str());

	SgExprStatement* const assign_stmt = SageBuilder::buildAssignStatement
	  (
	   SageBuilder::buildVarRefExp(lhs_string, body),
	   SageBuilder::buildVarRefExp(rhs_string, body)
	  );
	false_bb->append_statement(assign_stmt);
      }
    }

    SgIfStmt* const if_stmt = SageBuilder::buildIfStmt
      (cond_stmt, isSgStatement(true_bb), isSgStatement(false_bb));
    if_stmt->set_endOfConstruct(SOURCE_POSITION);
    SageInterface::appendStatement(if_stmt, body);

  } // kta->hasReduction() == true

  // unconditionnally call kaapi_task_end_adpative
  SgExprStatement* call_stmt = SageBuilder::buildFunctionCallStmt
    (    
     "kaapi_task_end_adaptive",
     SageBuilder::buildVoidType(), 
     SageBuilder::buildExprListExp(SageBuilder::buildVarRefExp(wc)),
     body
    );
  call_stmt->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement(call_stmt, body);
  
  return ;
}
