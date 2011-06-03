#include <rose.h>
#include <string>
#include <iostream>
#include <map>
#include <list>
#include <sstream>
#include <iomanip>
#include <ctype.h>
#include <time.h>

#define SOURCE_POSITION Sg_File_Info::generateDefaultFileInfoForTransformationNode()

struct KaapiTaskAttribute; 

static std::list<KaapiTaskAttribute*> all_tasks;
static std::map<std::string,KaapiTaskAttribute*> all_manglename2tasks;

typedef std::list<std::pair<SgFunctionDeclaration*, std::string> > ListTaskFunctionDeclaration;
static ListTaskFunctionDeclaration all_task_func_decl;

/* used to mark already instanciated template */
static std::set<std::string> all_template_instanciate; 
static std::set<SgFunctionDefinition*> all_template_instanciate_definition; 

/* list of all SgBasicBlock where a __kaapi_thread variable was insert */
static std::set<SgBasicBlock*>           all_listbb;

static SgType* kaapi_access_ROSE_type;
static SgType* kaapi_task_ROSE_type;
static SgType* kaapi_thread_ROSE_type;

static std::string ConvertCType2KaapiFormat(SgType* type);

SgClassDeclaration* buildClassDeclarationAndDefinition (
    const std::string& name, 
    SgScopeStatement* scope
);

/**
*/
SgClassDeclaration* buildStructDeclaration ( 
    SgScopeStatement* scope,
    const std::vector<SgType*>& memberTypes, 
    const std::vector<std::string>& memberNames,
    const std::string& structName
);

/**
*/
SgVariableDeclaration* buildStructVariable ( 
    SgScopeStatement* scope,
    const std::vector<SgType*>& memberTypes, 
    const std::vector<std::string>& memberNames,
    const std::string& structName = "", 
    const std::string& varName = "", 
    SgInitializer *initializer = NULL 
);

/**
*/
SgVariableDeclaration* buildStructVariable ( 
    SgScopeStatement* scope,
    SgClassDeclaration* classdecl,
    const std::string& varName = "", 
    SgInitializer *initializer = NULL 
);

SgVariableDeclaration* buildStructPointerVariable ( 
      SgScopeStatement* scope,
      SgClassDeclaration* classDeclaration,
      const std::string& varName, 
      SgInitializer *initializer = NULL
);


/* Build expression to declare a __kaapi_thread variable 
*/
void buildInsertDeclarationKaapiThread( SgBasicBlock* bbnode );

/*
*/
static void KaapiAbort( const std::string& msg )
{
  std::cerr << msg << std::endl;
  exit(1);
}

/* Store pragma_string to each task' template function declaration */
class KaapiPragmaString : public AstAttribute {
public:
  KaapiPragmaString (const std::string& s)
   : pragma_string(s)
  { }
  std::string pragma_string;
};


/* link a kaapi task AST function declaration to the pragma kaapi task */
enum KaapiAccessMode_t {
  KAAPI_VOID_MODE,
  KAAPI_W_MODE,
  KAAPI_R_MODE,
  KAAPI_RW_MODE,
  KAAPI_CW_MODE,
  KAAPI_V_MODE
};


/* Store view of each parameter */
class KaapiParamAttribute : public AstAttribute {
public:
  KaapiParamAttribute ()
   : lda(0), dim(0) 
  { }
  SgExpression* lda;
  size_t        dim;
  SgExpression* ndim[3];
};

struct KaapiTaskFormalParam {
  KaapiTaskFormalParam()
   : mode(KAAPI_VOID_MODE), attr(0), initname(0), type(0), kaapi_format()
  {}
  KaapiAccessMode_t    mode;
  KaapiParamAttribute* attr;
  SgInitializedName*   initname;
  SgType*              type;
  std::string          kaapi_format; 
};


/**
*/
class KaapiTaskAttribute : public AstAttribute {
public:
  KaapiTaskAttribute () 
  { }

  std::string                       mangled_name;      /* internal name of the task */
  std::vector<KaapiTaskFormalParam> formal_param;
  std::map<std::string,int>         lookup;
  SgScopeStatement                  scope;             /* for the formal parameter */
  std::string                       name_paramclass;   /* name of the type for arguments */
  SgClassDeclaration*               paramclass;        /* the class declaration for the parameters */
  SgTypedefDeclaration*             typedefparamclass; /* */
  SgFunctionDeclaration*            func_decl;         /* the function declaration */

  std::string                       name_wrapper;      /* name of the DFG wrapper */
  SgFunctionDeclaration*            wrapper_decl;      /* its declaration */
  SgFunctionDeclaration*            fwd_wrapper_decl;      /* its declaration */

  std::string                       name_format;       /* name of the format */
};


/* Attribut attached to a #pragma kaapi data alloca variable declaration */
class KaapiDataAttribute : public AstAttribute {
public:
  KaapiDataAttribute ( SgVariableSymbol* newsymbol ) 
   : symbol(newsymbol)
  { }

  SgVariableSymbol* symbol;
};


/** Utility: generate the rhs to get the dimension from the expression.
    Formal parameter in expr that corresponds to the i-th parameter is replaced by arg->fi 
*/
std::string GenerateGetDimensionExpression(
    KaapiTaskAttribute* kta, 
    SgExpression* expr
);

/** Utility: generate the lhs to get the dimension from the expression.
    Formal parameter in expr that corresponds to the i-th parameter is replaced by arg->fi 
*/
std::string GenerateSetDimensionExpression(
    KaapiTaskAttribute* kta, 
    SgExpression* expr,
    int index_in_view
);



/* fwd decl */
static SgExpression* ParseExpression( 
    Sg_File_Info* fileInfo, 
    std::istringstream& input, 
    SgScopeStatement* scope 
);

static inline int isletter( char c)
{ return (c == '_') || isalpha(c); }
static inline int isletternum( char c)
{ return (c == '_') || isalnum(c); }

/** Parse C/C++ identifier and return the last unrecognized char 
    which is putted back into the stream
    [a-zA-Z_]*[0-9]
*/
static int ParseIdentifier( std::string& ident, std::istringstream& input )
{
  char c;
  ident.clear();
  input >> c;
  if ( !isletter(c) )
  {
    input.putback(c);
    return c;
  }

  ident.push_back(c);
  do {
    if (!input.good()) 
      return EOF;
    input >> c;
    if (isletternum(c))
      ident.push_back(c);
    else if (isdigit(c))
      goto digits;
    else {
      input.putback(c);
      return c;
    }
  } while ( 1 );

digits:
  ident.push_back(c);
  do {
    if (!input.good()) 
      return EOF;
    input >> c;
    if (isdigit(c))
      ident.push_back(c);
    else {
      input.putback(c);
      return c;
    }
  } while ( 1 );
}


static SgUnsignedLongVal* ParseExprIntegralNumber( std::istringstream& input )
{
  unsigned long value = 0;
  char c;
  input >> c;
  while (isdigit(c))
  {
    value *= 10;
    value += c - '0';
    input >> c;
  }
  input.putback(c);
  return SageBuilder::buildUnsignedLongVal( value );
}

static SgVarRefExp* ParseExprIdentifier( 
    Sg_File_Info* fileInfo, 
    std::istringstream& input, 
    SgScopeStatement* scope 
)
{
  std::string ident;
  ParseIdentifier(ident, input);
  if (ident.empty())
  {
    std::cerr << "****[kaapi_c2c] Empty identifier."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  return SageBuilder::buildVarRefExp (ident, scope );
}

static SgExpression* ParseExprConstant( 
    Sg_File_Info* fileInfo, 
    std::istringstream& input, 
    SgScopeStatement* scope 
)
{
  char c;
  input >> c;
  input.putback(c);
  if (isdigit(c))
  {
    return ParseExprIntegralNumber( input );
  }
  return ParseExprIdentifier(fileInfo,input,scope);
}

/**/
static SgExpression* ParsePrimaryExpression( 
    Sg_File_Info* fileInfo, 
    std::istringstream& input, 
    SgScopeStatement* scope 
)
{
  std::cout << ">> ParsePrimaryExpression" << std::endl;
  SgExpression* expr;
  char c;
  input >> std::ws >> c;
  if (c == '(')
  {
    expr = ParseExpression(fileInfo, input, scope );
    if (c != ')') 
    {
      std::cerr << "****[kaapi_c2c] Error found '" << c 
                << "'. Missing ')' in primary expression."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    std::cout << ">> ParsePrimaryExpression" 
              << expr->unparseToString() << std::endl;
    return expr;
  }
  input.putback(c);
  expr = ParseExprConstant(fileInfo, input, scope );
  std::cout << ">> ParsePrimaryExpression: " 
            << expr->unparseToString() << std::endl;
  return expr;
}

/**/
static SgExpression* ParseUnaryExpression( 
    Sg_File_Info* fileInfo, 
    std::istringstream& input, 
    SgScopeStatement* scope 
)
{
  std::cout << ">> ParseUnaryExpression" << std::endl;
  SgExpression* expr;
  
#if 0 
  input >> std::ws;
/* currently do not handle sizeof expression 
   putback seems to be limited to 1 char.
*/
  char ci[7] = {0,0,0,0,0,0,0}; /* s i z e o f + 0 */
  input.getline(ci, 7);
  if (strcmp(ci,"sizeof") == 0) 
  {
    char c;
    input >> std::ws >> c;
    if (c == '(')
    {
      std::string type_name;
      ParseIdentifier( type_name, input );
      SgType* type = SageInterface::lookupNamedTypeInParentScopes(type_name, scope); 
      /* should be a type !!! */
      if (type ==0)
      {
        std::cerr << "****[kaapi_c2c] Error. Unknown type in cast expression.\n"
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }

      input >> std::ws >> c;
      if (c != ')') 
      {
        std::cerr << "****[kaapi_c2c] Error. Missing ')' in cast expression.\n"
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
      expr = SageBuilder::buildSizeOfOp(type);
      std::cout << ">> ParseUnaryExpression: " 
                << expr->unparseToString() << std::endl;
      return expr;
    }
    else 
    {
      input.putback(c);
      expr = ParseUnaryExpression(fileInfo, input, scope );
      expr = SageBuilder::buildSizeOfOp(expr);
      std::cout << ">> ParseUnaryExpression" 
                << expr->unparseToString() << std::endl;
      return expr;
    }
  }
  else {
    int rsize= input.gcount();
    for (int i=0; i<rsize; ++i)
      input.putback(ci[i]);
  }
#endif

  expr = ParsePrimaryExpression( fileInfo, input, scope);
  std::cout << ">> ParseUnaryExpression: " 
            << expr->unparseToString() << std::endl;
  return expr;
}

/**/
static SgExpression* ParseCastExpression( 
    Sg_File_Info* fileInfo, 
    std::istringstream& input, 
    SgScopeStatement* scope 
)
{
  std::cout << ">> ParseCastExpression" << std::endl;
  SgExpression* expr;
  char c;
  input >> std::ws >> c;
  if (c == '(') 
  {
    std::string type_name;
    ParseIdentifier( type_name, input );
    SgType* type = SageInterface::lookupNamedTypeInParentScopes(type_name, scope); 
    /* should be a type !!! */
    if (type ==0)
    {
      std::cerr << "****[kaapi_c2c] Error. Unknown type in cast expression."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }

    input >> std::ws >> c;
    if (c != ')')
    {
      std::cerr << "****[kaapi_c2c] Error. Missing ')' in cast expression."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    expr = ParseCastExpression(fileInfo, input, scope);
    expr = SageBuilder::buildCastExp (expr, type );
    std::cout << ">> ParseCastExpression: " << expr->unparseToString() << std::endl;
    return expr;
  }
  input.putback(c);
  expr= ParseUnaryExpression( fileInfo, input, scope);
  std::cout << ">> ParseCastExpression: " << expr->unparseToString() << std::endl;
  return expr;
}

/**/
static SgExpression* ParseMultiplicativeExpression( 
    Sg_File_Info* fileInfo, 
    std::istringstream& input, 
    SgScopeStatement* scope 
)
{
  std::cout << ">> ParseMultiplicativeExpression" << std::endl;
  char c;
  SgExpression* expr= ParseCastExpression( fileInfo, input, scope);
redo:
  input >> std::ws >> c;
  if (c == '*')
  {
    expr = SageBuilder::buildMultiplyOp( expr, ParseCastExpression( fileInfo, input, scope) );
    goto redo;
  }
  else if (c == '/')
  {
    expr = SageBuilder::buildDivideOp( expr, ParseCastExpression( fileInfo, input, scope) );
    goto redo;
  }
  else if (c == '%')
  {
    expr = SageBuilder::buildModOp( expr, ParseCastExpression( fileInfo, input, scope) );
    goto redo;
  }
  else 
    input.putback(c);
  std::cout << "<< ParseMultiplicativeExpression: " << expr->unparseToString() << std::endl;
  return expr;
}

/**/
static SgExpression* ParseAdditiveExpression( 
    Sg_File_Info* fileInfo, 
    std::istringstream& input, 
    SgScopeStatement* scope 
)
{
  std::cout << ">> ParseAdditiveExpression" << std::endl;
  char c;
  SgExpression* expr= ParseMultiplicativeExpression( fileInfo, input, scope);
redo:
  input >> std::ws >> c;
  if (c == '+')
  {
    expr = SageBuilder::buildAddOp( expr, ParseMultiplicativeExpression( fileInfo, input, scope) );
    goto redo;
  }
  else if (c == '-')
  {
    expr = SageBuilder::buildSubtractOp( expr, ParseMultiplicativeExpression( fileInfo, input, scope) );
    goto redo;
  }
  else 
    input.putback(c);
  std::cout << "<< ParseAdditiveExpression: " << expr->unparseToString() << std::endl;
  return expr;
}

static SgExpression* ParseExpression( 
    Sg_File_Info* fileInfo, 
    std::istringstream& input, 
    SgScopeStatement* scope 
)
{
  SgExpression* expr;
  std::cout << ">> Parse Expression" << std::endl;
  expr = ParseAdditiveExpression( fileInfo, input, scope);
  std::cout << "<< Parse Expression:" << expr->unparseToString() << std::endl;
  return expr;
}

static void ParseDimension( 
    Sg_File_Info* fileInfo, 
    KaapiParamAttribute* kpa, 
    std::istringstream& input, 
    SgScopeStatement* scope 
)
{
  char c;
redo_onedim:
  input >> std::ws >> c;
  if (c != '[') 
  {
    input.putback(c);
    return;
  }
  kpa->ndim[kpa->dim] = ParseExpression( fileInfo, input, scope );
  input >> std::ws >> c;
  if (c != ']') 
  {
    std::cerr << "****[kaapi_c2c] Error. Missing ']' in dimension expression."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  if (++kpa->dim == 3) 
  {
    std::cerr << "****[kaapi_c2c] Error. Too high dimension for array (hard limit=2)."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }

  goto redo_onedim;
}

/* Parse one formal parameter definition and return its attribut
   after parsing the expression defining the view.
        
      dimension :=
          one_dimension | dimension one_dimension
      one_dimension :=
          '[' expression ']'

      lda :=
        identifier

      expression :=
        additive_expression

      additive_expression :=
          multiplicative_expression
        | additive_expression '+' multiplicative_expression
        | additive_expression '-' multiplicative_expression
        
      multiplicative_expression
          cast_expression
        | multiplicative_expression '*' cast_expression
        | multiplicative_expression '/' cast_expression
        | multiplicative_expression '%' cast_expression

      cast_expression
          unary_expression
        | '(' type_name ')' cast_expression

      unary_expression
          primary_expression
        | SIZEOF unary_expression
        | SIZEOF '(' type_name ')'
      
      primary_expression
          identifier
        | integral
        | '(' expression ')'

*/
static KaapiParamAttribute* ParseParamDecl( 
    Sg_File_Info* fileInfo, 
    std::string& ident, 
    std::istringstream& input, 
    SgScopeStatement* scope 
)
{
  char c;
  ParseIdentifier( ident, input );
  input >> std::ws >> c;
  KaapiParamAttribute* kpa = new KaapiParamAttribute;
  if ((c == ',') || (c == ')'))
  {
    input.putback(c);
    return kpa;
  }
  
  if (c == '{') 
  { /* complex form */
    kpa->lda = ParseExprIdentifier(fileInfo,input,scope);
    input >> std::ws >> c;
    if (c != ';') {
      std::cerr << "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                << "\n Bad syntax for complex view description of a formal parameter. Waiting ';'. Found: '" << c << "'"
                << std::endl;
      KaapiAbort("*** Error");
    }
    ParseDimension( fileInfo, kpa, input, scope );
    input >> std::ws >> c;
    if (c != '}')
    {
      std::cerr << "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                << "\n Bad syntax for complex view description of a formal parameter. Waiting '}'. Found: '" << c << "'"
                << std::endl;
      KaapiAbort("*** Error");
    }
    return kpa;
  }
  if (c != '[')
  {
    std::cerr << "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << "\n Bad syntax for view description of a formal parameter. Waiting '['. Found: '" << c << "'"
              << std::endl;
    KaapiAbort("*** Error");
  }

  input.putback(c);
  ParseDimension( fileInfo, kpa, input, scope );
  input >> std::ws >> c;
  if ((c == ',') || (c == ')'))
  {
    input.putback(c);
    return kpa;
  }
  std::cerr << "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
            << "\n Bad syntax for view description of a formal parameter. Waiting ',' or ')'. Found: '" << c << "'"
            << std::endl;
  KaapiAbort("*** Error");
  return 0;
}


/* Parse the list of formal parameter definitions
   The grammar is:
      list_param :=
          access_param
        | list_param access_param
      
      access_param :=
          mode '(' list_declaration ')
      
      list_declaration :=
          range_declaration
        | list_declaration ',' range_declaration
        
      range_declaration :=
          identifier
        | identifier dimension
        | identifier '{' complex_view '}
        | '[' identifier '..' identifier ']'
      
      complex_view :=
          element_view
        | complex_dimension ',' element_view
      
      element_view :=
          storage
        | lda
        | dimension
        
      mode :=
          'write' | 'w' | 'output'
        | 'read' | 'r' | 'input'
        | 'exclusive' | 'x' | 'inout'
        | 'value' | 'v'
        | 'reduction' | 'cw'
      
      storage :=
          'rowmajor' | 'C'
        | 'columnmajor' | 'Fortran'
      
      lda :=
          identifier
          
      See in the grammar above for the remainders rules about dimension,
      lda etc.
      
*/
static void ParseListParamDecl( 
    Sg_File_Info* fileInfo, 
    KaapiTaskAttribute* kta, 
    std::istringstream& input 
)
{
  std::string name;
  std::string ident;
  char c;
  KaapiParamAttribute* kpa;
  KaapiAccessMode_t    mode;
  
  while (input.good())
  {
    input >> std::ws >> name;
    if (!input.good()) return;
    if ((name == "write") || (name == "w") || (name == "output"))
      mode = KAAPI_W_MODE;
    else if ((name == "read") || (name == "r") || (name == "input"))
      mode = KAAPI_R_MODE;
    else if ((name == "exclusive") || (name == "x") || (name == "inout"))
      mode = KAAPI_R_MODE;
    else if ((name == "value") || (name == "v"))
      mode = KAAPI_V_MODE;
    else if ((name == "reduction") || (name == "cw"))
      mode = KAAPI_CW_MODE;
    else
      KaapiAbort("*** Error list of parameters should defined access mode");

    input >> std::ws >> c;
    if (c != '(') 
    {
      std::cerr << "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                << "\n Bad syntax for view description of formal parameter. "
                << "Within scope of '" << name << "' access mode, missing '('. Found: '" << c << "'"
                << std::endl;
      KaapiAbort("*** Error");
    }

redo_oneparam:
    kpa = ParseParamDecl( fileInfo, ident, input, &kta->scope );
    if (kpa !=0) 
    {
      std::map<std::string,int>::iterator curr = kta->lookup.find( ident );
      if (curr == kta->lookup.end())
      {
        std::cerr << "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << "\n Cannot find formal parameter '" << ident << "' in the task's formal parameter list"
                  << std::endl;
        KaapiAbort("*** Error");
      }

      int ith = curr->second;
      kta->formal_param[ith].mode = mode;
      kta->formal_param[ith].attr = kpa;
    }

    input >> std::ws >> c;
    if (c == ')') continue;
    goto redo_oneparam;
  }
}
  

/** #pragma kaapi task parsing
*/
void DoKaapiPragmaTask( SgPragmaDeclaration* sgp, const std::string& pragma_string )
{
#if 0
  std::cout << "This is a kaapi task definition, parent class name: " 
            << sgp->get_parent()->class_name() 
            << std::endl;
#endif
  Sg_File_Info* fileInfo = sgp->get_file_info();

  /* #pragma kaapi task: find next function declaration to have the name */
  if (sgp->get_parent() ==0)
    KaapiAbort( "*** Internal compiler error: mal formed AST" );
    
  /* it is assumed that the statement following the #pragma kaapi task is a declaration */
  SgNode* fnode = 
          sgp->get_parent()-> get_traversalSuccessorByIndex( 
              1+ sgp->get_parent()->get_childIndex( sgp ) );

  if (isSgTemplateDeclaration(fnode))
  {
    if (isSgTemplateDeclaration(fnode)->get_template_kind() == SgTemplateDeclaration::e_template_function)
      fnode->setAttribute("kaapi_templatetask", new KaapiPragmaString(pragma_string));
    return;
  }
  
  SgFunctionDeclaration* functionDeclaration = isSgFunctionDeclaration( fnode );
  if (functionDeclaration ==0) 
  {
    std::cerr << "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << "\n#pragma kaapi task must be followed by a function declaration. Found: " << fnode->class_name()
              << std::endl;
    KaapiAbort("**** error");
  }
  SgMemberFunctionDeclaration* methodDeclaration = isSgMemberFunctionDeclaration(fnode );
  if (methodDeclaration !=0) 
  {
    std::cerr << "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << "\n#pragma kaapi task before a method: not yet implemented"
              << std::endl;
    KaapiAbort("**** error");
  } 
  all_task_func_decl.push_back( std::make_pair(functionDeclaration, pragma_string) );
}


void DoKaapiTaskDeclaration( SgFunctionDeclaration* functionDeclaration, std::istringstream& input )
{
  Sg_File_Info* fileInfo = functionDeclaration->get_file_info();

  /* */
  KaapiTaskAttribute* kta = new KaapiTaskAttribute;
  kta->func_decl = functionDeclaration;
  
  /* Mark the body of the function as a "kaapiisparallelregion" */
  if (functionDeclaration->get_definition() !=0)
  {
    SgBasicBlock* body = functionDeclaration->get_definition()->get_body();
    body->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
#if 0
    std::cout << "Set the body:" << body << " of the function: "<<functionDeclaration->get_name().str() 
              << " as an implicit parallel region"
              << std::endl;
#endif
  }


  /* Store formal parameter in the KaapiTaskAttribute structure */
  SgFunctionParameterList* func_param = functionDeclaration->get_parameterList();
  SgInitializedNamePtrList& args = func_param->get_args();
  SgScopeStatement* scope_declaration = functionDeclaration->get_scope();
  kta->formal_param.resize( args.size() );
  for (unsigned int i=0; i<args.size(); ++i)
  {
    SgInitializedName* initname   = args[i];
    kta->formal_param[i].initname = initname; //->get_qualified_name().str();
    kta->formal_param[i].type     = initname->get_typeptr();
    if (initname->get_name().is_null())
    {
      std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                << " #pragma kaapi task: function '" << functionDeclaration->get_mangled_name().str() << "'"
                << " has missing name for its " << i << "-th formal parameter"
                << std::endl;
      KaapiAbort("**** error");
    }
    /* try to find the parameter through the scope */
    kta->lookup.insert( std::make_pair(initname->get_name().str(), i ) );
  }
  
  /* parse the end of the stream of declaration
  */
  ParseListParamDecl( fileInfo, kta, input );
  
  /* Add the class declaration for the parameters */
  kta->name_paramclass = std::string("__kaapi_args_") + 
      functionDeclaration->get_name().str() + 
      functionDeclaration->get_mangled_name().str();

  kta->paramclass = 
    buildClassDeclarationAndDefinition( 
      kta->name_paramclass,
      scope_declaration
    );
  for (unsigned int i=0; i<kta->formal_param.size(); ++i)
  {
    
    std::ostringstream name;
    name << "f" << i;
    SgType* member_type = 0;
redo_selection:
    if (kta->formal_param[i].mode == KAAPI_V_MODE)
    {
      member_type = kta->formal_param[i].type;
      if (isSgModifierType(member_type))
        member_type = isSgModifierType(member_type)->get_base_type();
      kta->formal_param[i].kaapi_format = ConvertCType2KaapiFormat(kta->formal_param[i].type);
    }
    else
    {
      if (!isSgPointerType(kta->formal_param[i].type))
      { /* read/write/reduction should be pointer: else move them to be by value */
        std::cerr << "****[kaapi_c2c] Warning: incorrect access mode.\n"
                  << "     In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << " formal parameter '" << kta->formal_param[i].initname->get_name().str()
                  << "' is declared as read/write/reduction and should be declared as a pointer.\n"
                  << "      Change access mode to be a value.\n"
                  << std::endl;
        kta->formal_param[i].mode = KAAPI_V_MODE;
        goto redo_selection;
        KaapiAbort("**** error");
      }
      kta->formal_param[i].kaapi_format = ConvertCType2KaapiFormat(
        isSgPointerType(kta->formal_param[i].type)->get_base_type()
      );
      member_type = kaapi_access_ROSE_type;
    }

    SgVariableDeclaration* memberDeclaration =
      SageBuilder::buildVariableDeclaration (
        name.str(), 
        member_type,
        0, 
        kta->paramclass->get_definition()
    );
    memberDeclaration->set_endOfConstruct(SOURCE_POSITION);
    
    kta->paramclass->get_definition()->append_member(memberDeclaration);
  }

#if 0 // Here is a tentative to add extra member to represent a view: not good
      // should always be constant (and so define in the code of the format)
      // or an identifier of a formal parameter
  for (unsigned int i=0; i<kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].attr->dim >=1)
    {
      int d;
      switch(kta->formal_param[i].attr->dim)
      {
        case 1: d = 1; break; /* only store 1d size */
        case 2: d = 3; break; /* store n, m and lda */
        case 3: KaapiAbort("****To high dimension"); 
      }
      std::ostringstream name;
      name << "view" << i << "[" << d << "]";
      SgType* member_type = 0;
        member_type = SageBuilder::buildIntType();

        SgVariableDeclaration* memberDeclaration =
          SageBuilder::buildVariableDeclaration (
            name.str(), 
            member_type,
            0, 
            kta->paramclass->get_definition()
        );
        memberDeclaration->set_endOfConstruct(SOURCE_POSITION);
        
        kta->paramclass->get_definition()->append_member(memberDeclaration);
    }
  }
#endif

  kta->typedefparamclass = SageBuilder::buildTypedefDeclaration(
      kta->name_paramclass, 
      new SgClassType(kta->paramclass), //->get_definingDeclaration()), //get_firstNondefiningDeclaration()), 
      scope_declaration
  );
  SageInterface::insertStatement( functionDeclaration, kta->paramclass, true );
  SageInterface::insertStatement( kta->paramclass, kta->typedefparamclass, false );
  
  /* generate the wrapper function and its fwd declaration. Copy the instructions
     to generate the function parameter list because it cannot be shared between
     2 delcarations and deep copy == seg fault
   */
  kta->mangled_name = std::string(functionDeclaration->get_name().str()) + 
      functionDeclaration->get_mangled_name().str();
  kta->name_wrapper = std::string("__kaapi_wrapper_") + kta->mangled_name;

  SgFunctionParameterList* fwd_parameterList = SageBuilder::buildFunctionParameterList();
  SageInterface::appendArg(fwd_parameterList, 
    SageBuilder::buildInitializedName("_k_arg", SageBuilder::buildPointerType(SageBuilder::buildVoidType()) )
  );
  SageInterface::appendArg(fwd_parameterList, 
    SageBuilder::buildInitializedName("_k_thread", SageBuilder::buildPointerType(kaapi_thread_ROSE_type) )
  );

  /* fwd declaration before the body of the original function */
  kta->fwd_wrapper_decl =
    SageBuilder::buildNondefiningFunctionDeclaration (
      kta->name_wrapper, 
      SageBuilder::buildVoidType(),
      fwd_parameterList,
      scope_declaration
  );
  ((kta->fwd_wrapper_decl->get_declarationModifier()).get_storageModifier()).setExtern();

  /* declaration + definition: generated after the #pragma kaapi task */
  SgFunctionParameterList* parameterList = SageBuilder::buildFunctionParameterList();

  SageInterface::appendArg(parameterList, 
    SageBuilder::buildInitializedName("_k_arg", SageBuilder::buildPointerType(SageBuilder::buildVoidType()) )
  );
  SageInterface::appendArg(parameterList, 
    SageBuilder::buildInitializedName("_k_thread", SageBuilder::buildPointerType(kaapi_thread_ROSE_type) )
  );
  kta->wrapper_decl =
    SageBuilder::buildDefiningFunctionDeclaration (
      kta->name_wrapper, 
      SageBuilder::buildVoidType(),
      parameterList, 
      scope_declaration
  );
//DO not put it extern, fwd declaration already indicates that
//  ((kta->wrapper_decl->get_declarationModifier()).get_storageModifier()).setExtern();
  SgBasicBlock*  wrapper_body = kta->wrapper_decl->get_definition()->get_body();
  SgClassType* class_forarg = new SgClassType(kta->paramclass->get_firstNondefiningDeclaration());
  
  /* add call to the original function */
  SgVariableDeclaration* truearg_decl = SageBuilder::buildVariableDeclaration ( 
    "thearg", 
    SageBuilder::buildPointerType(class_forarg),
    SageBuilder::buildAssignInitializer(
      SageBuilder::buildCastExp(
        SageBuilder::buildVarRefExp ("_k_arg", wrapper_body ),
        SageBuilder::buildPointerType(class_forarg)
      ),
      0
    ),
    wrapper_body
  );
  SageInterface::prependStatement( truearg_decl, wrapper_body );
  
  SgExprListExp* argscall = SageBuilder::buildExprListExp();
  
  for (unsigned int i=0; i<kta->formal_param.size(); ++i)
  {
    std::ostringstream fieldname;
    fieldname << "(" << kta->formal_param[i].type->unparseToString() << ")thearg->f" << i;
    if (kta->formal_param[i].mode != KAAPI_V_MODE)
      fieldname << ".data";
    SageInterface::appendExpression(argscall,
      SageBuilder::buildOpaqueVarRefExp(fieldname.str(),wrapper_body)
    );
  }
  
  SgExprStatement* callStmt = SageBuilder::buildFunctionCallStmt(    
      functionDeclaration->get_name(),
      SageBuilder::buildVoidType(), 
      argscall,
      wrapper_body
  );
  callStmt->setAttribute("kaapiwrappercall", (AstAttribute*)-1);
  SageInterface::insertStatement( truearg_decl, callStmt, false );
    
//  SageInterface::insertStatement( kta->typedefparamclass, kta->wrapper_decl, false );
  SageInterface::insertStatement( kta->typedefparamclass, kta->fwd_wrapper_decl, false );

  SageInterface::insertStatement( kta->func_decl, kta->wrapper_decl, false );

  /* annotated the AST function declaration node to the Pragma node */
  SgTreeCopy helper;
  functionDeclaration->setAttribute("kaapitask", 
    kta
  );
  all_tasks.push_back( kta );
  all_manglename2tasks.insert( std::make_pair(kta->mangled_name, kta) );
#if 0
  std::cout << "***** Attach Task attribut to declaration: " << functionDeclaration 
            << " name: " << functionDeclaration->get_name().str()
            << " access mode is:" << input.str()
            << std::endl;
#endif
}


/**
*/
void DoKaapiPragmaNoTask( SgPragmaDeclaration* sgp, std::istringstream& input )
{
  /* #pragma kaapi task: find next function declaration to have the name */
  if (sgp->get_parent() ==0)
    KaapiAbort( "*** Internal compiler error: mal formed AST" );
    
  /* it is assumed that the statement following the #pragma kaapi task is a declaration */
  SgNode* fnode = 
          sgp->get_parent()-> get_traversalSuccessorByIndex( 
              1+ sgp->get_parent()->get_childIndex( sgp ) );

  SgExprStatement* exprstatement = isSgExprStatement(fnode);
  if ((exprstatement ==0) || (isSgFunctionCallExp( exprstatement->get_expression() ) ==0))
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi notask must be followed by a function call expression"
              << " it is a:" << fnode->class_name()
              << std::endl;
    KaapiAbort("**** error");
  }
  
//  SgFunctionCallExp* fc = isSgFunctionCallExp( exprstatement->get_expression() );
  exprstatement->setAttribute("kaapinotask",(AstAttribute*)1);
}


/**
*/
void DoKaapiGenerateFormat( std::ostream& fout, KaapiTaskAttribute* kta)
{
  std::cout << "****[kaapi_c2c] Task's generation format for function: " << kta->func_decl->get_name().str()
            << std::endl;
  
  kta->name_format = kta->name_paramclass + "_format";

  fout << "/*** Format for task argument:" 
       << kta->name_paramclass
       << "***/\n";
  
  SgUnparse_Info* sg_info = new SgUnparse_Info;
  sg_info->unset_forceQualifiedNames();
  
  fout << kta->paramclass->unparseToString(sg_info) << std::endl;
  fout << kta->typedefparamclass->unparseToString(sg_info) << std::endl << std::endl;
  fout << kta->fwd_wrapper_decl->unparseToString(sg_info) << std::endl << std::endl;
  
  /* format definition */
  fout << "/* format object*/\n"
       << "struct kaapi_format_t*" << kta->name_format << " = 0;\n\n\n";

  /* format::get_count_params */
  fout << "size_t " << kta->name_format << "_get_count_params(const struct kaapi_format_t* fmt, const void* sp)\n"
       << "{ return " << kta->formal_param.size() << "; }\n"
       << std::endl;

  /* format::get_mode_param */
  fout << "kaapi_access_mode_t " << kta->name_format << "_get_mode_param(const struct kaapi_format_t* fmt, unsigned int i, const void* sp)\n"
       << "{ \n"
       << "  static kaapi_access_mode_t mode_param[] = {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    switch (kta->formal_param[i].mode) {
      case KAAPI_V_MODE: fout << "    KAAPI_ACCESS_MODE_V, "; break;
      case KAAPI_W_MODE: fout << "    KAAPI_ACCESS_MODE_W, "; break;
      case KAAPI_R_MODE: fout << "    KAAPI_ACCESS_MODE_R, "; break;
      case KAAPI_RW_MODE:fout << "    KAAPI_ACCESS_MODE_RW, "; break;
      case KAAPI_CW_MODE:fout << "    KAAPI_ACCESS_MODE_CW, "; break;
      default:
        break;
    }
  }
  fout << "    KAAPI_ACCESS_MODE_VOID\n  };\n"; /* marker for end of mode */
  fout << "  return mode_param[i];\n"
       << "}\n" 
       << std::endl;
  
  /* format::get_off_param */
  fout << "void* " << kta->name_format << "_get_off_param(const struct kaapi_format_t* fmt, unsigned int i, const void* sp)\n"
       << "{\n  " << kta->name_paramclass << "* arg = (" << kta->name_paramclass << "*)sp;\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    fout << "    case " << i << ": return &arg->f" << i << ";\n";
  }
  fout << "  }\n"
       << "  return 0;\n"
       << "}\n"
       << std::endl;

  /* format::get_cwflag */
  fout << "int* " << kta->name_format << "_get_cwflag(const struct kaapi_format_t* fmt, unsigned int i, const void* sp)\n"
       << "{ return 0; }\n"
       << std::endl;
       
  /* format::get_access_param */
  fout << "kaapi_access_t " << kta->name_format << "_get_access_param(const struct kaapi_format_t* fmt, unsigned int i, const void* sp)\n"
       << "{\n  " << kta->name_paramclass << "* arg = (" << kta->name_paramclass << "*)sp;\n"
       << "  kaapi_access_t retval = {0,0};\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].mode == KAAPI_V_MODE)
      fout << "    case " << i << ": break;\n";
    else
      fout << "    case " << i << ": retval = arg->f" << i << "; break; \n" ;/* because it is an access here */
  }
  fout << "  }\n"
       << "  return retval;\n"
       << "}\n"
       << std::endl;
  
  /* format::set_access_param */
  fout << "void " << kta->name_format << "_set_access_param(const struct kaapi_format_t* fmt, unsigned int i, void* sp, const kaapi_access_t* a)\n"
       << "{\n  " << kta->name_paramclass << "* arg = (" << kta->name_paramclass << "*)sp;\n"
       << "  kaapi_access_t retval = {0,0};\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].mode != KAAPI_V_MODE)
      fout << "    case " << i << ": arg->f" << i << " = *a" << "; return; \n"; /* because it is an access here */
  }
  fout << "  }\n"
       << "}\n"
       << std::endl;

  /* format::set_access_param */
  fout << "void " << kta->name_format << "_set_cwaccess_param(const struct kaapi_format_t* fmt, unsigned int i, void* sp, const kaapi_access_t* a, int wa)\n"
       << "{ return; }\n"
       << std::endl;
  
  /* format::get_fmt_param */
  fout << "const struct kaapi_format_t* " << kta->name_format << "_get_fmt_param(const struct kaapi_format_t* fmt, unsigned int i, const void* sp)\n"
       << "{\n  " << kta->name_paramclass << "* arg = (" << kta->name_paramclass << "*)sp;\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
    fout << "    case " << i << ": return " << kta->formal_param[i].kaapi_format << ";\n";
  fout << "  }\n"
       << "}\n"
       << std::endl;
       
  /* format::get_view_param */
  fout << "kaapi_memory_view_t " << kta->name_format << "_get_view_param(const struct kaapi_format_t* fmt, unsigned int i, const void* sp)\n"
       << "{\n  " << kta->name_paramclass << "* arg = (" << kta->name_paramclass << "*)sp;\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].mode == KAAPI_V_MODE)
      fout << "    case " << i << ": return kaapi_memory_view_make1d( 1, sizeof(arg->f" << i << "));\n";
    else {
      if (kta->formal_param[i].attr->dim == 1)
        fout << "    case " << i << ": return kaapi_memory_view_make1d( " 
                            << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->ndim[0])
                            << ", sizeof(arg->f" << i << "));\n";
      else if (kta->formal_param[i].attr->dim == 2)
      {
        fout << "    case " << i << ": return kaapi_memory_view_make2d( "
                            << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->ndim[0]) << ","
                            << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->ndim[1]) << ",";
        if (kta->formal_param[i].attr->lda == 0) /* means contigous & row major == lda = dimj */
          fout              << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->ndim[1]);
        else
          fout              << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->lda);
        fout                << ", sizeof(arg->f" << i << "));\n";
      }
      else if (kta->formal_param[i].attr->dim == 3)
      {
        fout << "    case " << i << ": kaapi_abort();\n";
      }      
    }
  }
  fout << "  }\n"
       << "}\n"
       << std::endl;


  /* format::get_view_param */
  fout << "void " << kta->name_format << "_set_view_param(const struct kaapi_format_t* fmt, unsigned int i, void* sp, const kaapi_memory_view_t* view)\n"
       << "{\n  " << kta->name_paramclass << "* arg = (" << kta->name_paramclass << "*)sp;\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].mode != KAAPI_V_MODE)
    {
      if ((kta->formal_param[i].attr->dim == 2) && (kta->formal_param[i].attr->lda !=0))
        fout << "    case " << i << ": {\n "
             << "      " << GenerateSetDimensionExpression(kta, kta->formal_param[i].attr->lda, -1) << "\n"
             << "    return; }\n";
      else if (kta->formal_param[i].attr->dim == 3)
      {
        fout << "    case " << i << ": kaapi_abort();\n";
      }      
    }
  }
  fout << "  }\n"
       << "}\n"
       << std::endl;

  /* format::reducor */
  fout << "void " << kta->name_format << "_reducor(const struct kaapi_format_t* fmt, unsigned int i, const void* sp, void* result, const void* value)\n"
       << "{ return; }\n"
       << std::endl;

  /* format::reducor */
  fout << "kaapi_reducor_t " << kta->name_format << "_get_reducor(const struct kaapi_format_t* fmt, unsigned int i, const void* sp)\n"
       << "{ kaapi_reducor_t a; return a; }\n"
       << std::endl;

#if 0
  /* format::get_task_binding */
  fout << "void " << kta->name_format << "_get_task_binding(const struct kaapi_format_t* fmt, const kaapi_task_t* t, kaapi_task_binding_t* tb)\n"
       << "{ return; }\n"
       << std::endl;
#endif
       
  /* Generate constructor function that register the format */
  fout << "/* constructor method */\n" 
       << "static void __attribute__ ((constructor)) " << kta->name_format << "_constructor(void)\n"
       << "{\n"
       << "  if (" << kta->name_format << " !=0) return;\n"
       << "  " << kta->name_format << " = kaapi_format_allocate();\n"
       << "  kaapi_format_taskregister_func(\n"
       << "    " << kta->name_format << ",\n" /* format object */
       << "    " << kta->name_wrapper << ",\n" /* body */
       << "    " << 0 << ",\n" /* bodywh */
       << "    \"" << kta->name_format << "\",\n" /* name */
       << "    sizeof(" << kta->name_paramclass << "),\n" /* sizeof the arg struct */
       << "    " << kta->name_format << "_get_count_params,\n" /* get_count_params */
       << "    " << kta->name_format << "_get_mode_param,\n" /* get_mode_param */
       << "    " << kta->name_format << "_get_off_param,\n" /* get_off_param */
       << "    " << kta->name_format << "_get_cwflag,\n" /* get_cwflag */
       << "    " << kta->name_format << "_get_access_param,\n" /* get_access_param */
       << "    " << kta->name_format << "_set_access_param,\n" /* set_access_param */
       << "    " << kta->name_format << "_set_cwaccess_param,\n" /* set_cwaccess_param */
       << "    " << kta->name_format << "_get_fmt_param,\n" /* get_fmt_param */
       << "    " << kta->name_format << "_get_view_param,\n" /* get_view_param */
       << "    " << kta->name_format << "_set_view_param,\n" /* set_view_param */
       << "    " << kta->name_format << "_reducor,\n" /* reducor */
       << "    " << kta->name_format << "_get_reducor\n" /* get_reducor */
//NO BINDING IN THIS VERSION       << "    " << kta->name_format << "_get_task_binding\n" /* get_task_binding */
       << "  );\n"
       << "}\n"
       << std::endl;
}


/** #pragma kaapi data parsing
*/
class nestedVarRefVisitorTraversal : public AstSimpleProcessing {
public:
  nestedVarRefVisitorTraversal( )
  { }
  virtual void visit(SgNode* n)
  {
    SgVarRefExp* varref = isSgVarRefExp(n);
    if (varref != 0) 
    {
      SgVariableSymbol* varsym = varref->get_symbol();
      KaapiDataAttribute* kada = (KaapiDataAttribute*)varsym->getAttribute("kaapidata");
      if (kada !=0)
      {
  #if 0
        Sg_File_Info* fileInfo = varref->get_file_info();
        std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << " Found use of VarRef for variable name:" << varsym->get_name().str()
                  << std::endl;
  #endif
        
        SgVarRefExp*       newvarref = SageBuilder::buildVarRefExp (kada->symbol );
        SgPointerDerefExp* deref = SageBuilder::buildPointerDerefExp( newvarref );
        
        /* replace the expression */
        SgExpression* parentexptr = isSgExpression(varref->get_parent());
        parentexptr->replace_expression( varref, deref );
      }
    }    
  }
};

/**
*/
void DoKaapiPragmaData( SgNode* node, std::istringstream& input )
{
  std::string name;
  SgPragmaDeclaration* sgp = isSgPragmaDeclaration(node);
  
  input >> std::ws >> name;
  if (name != "alloca")
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi data: expecting 'alloca' clause, found '" << name << "'"
              << std::endl;
    KaapiAbort("**** error");
  }

  SgBasicBlock* bbnode = isSgBasicBlock(sgp->get_parent());
  if (bbnode ==0)
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi data: invalid scope declaration"
              << std::endl;
    KaapiAbort("**** error");
  }

  SgSymbolTable* symtable = bbnode->get_symbol_table();
  
  /* parse the variables declaration in the pragma string */
  input >> std::ws >> name;
  if (name != "(") 
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi data alloca clause: missing '('"
              << std::endl;
    KaapiAbort("**** error");
  }
  
  bool findend = false;
  while (input.good())
  {
    input >> std::ws;
    ParseIdentifier( name, input );
    if (name.size() >0)
    {
#if 0
      std::cout << "-- Look for variable: '" << name << "' separator:'" << c << "'" << std::endl;
#endif
      
      SgName varname (name);
      SgVariableSymbol* var = symtable->find_variable( varname );
      if (var == 0)
      {
        Sg_File_Info* fileInfo = sgp->get_file_info();
        std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << " #pragma kaapi data alloca clause: variable name '" << name << "' not found in the scope"
                  << std::endl;
        KaapiAbort("**** error");
      }

      /* add a variable with name _kaapi_varname and type the pointer of the current type */
      SgType* type = var->get_type();
      SgPointerType* newtype = SageBuilder::buildPointerType(type);
      
      /* if it is not alread done: add __kaapi_thread variable */
      SgVariableSymbol* newkaapi_threadvar = 
          SageInterface::lookupVariableSymbolInParentScopes(
              "__kaapi_thread", 
              bbnode 
      );
      if (newkaapi_threadvar ==0)
        buildInsertDeclarationKaapiThread( bbnode);
  
      SgVariableDeclaration* newvardecl = SageBuilder::buildVariableDeclaration ( 
        "_kaapi_"+name, 
        newtype, 
        SageBuilder::buildAssignInitializer(
          SageBuilder::buildCastExp(
            SageBuilder::buildFunctionCallExp(
              "kaapi_alloca",
              SageBuilder::buildPointerType(SageBuilder::buildVoidType()),
              SageBuilder::buildExprListExp(
                SageBuilder::buildVarRefExp ("__kaapi_thread", bbnode ),
                SageBuilder::buildSizeOfOp( type )
              ),
              bbnode
            ),
            newtype 
          ),
          0
        ),
        bbnode
      );
      SgVariableSymbol* newvar = symtable->find_variable( "_kaapi_"+name );

      var->setAttribute("kaapidata", 
        new KaapiDataAttribute( newvar )
      );
      SageInterface::insertStatement(sgp, newvardecl, false);
      SageInterface::removeStatement( var->get_declaration ()->get_declaration() );
    }
    else {
        Sg_File_Info* fileInfo = sgp->get_file_info();
        std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << " #pragma kaapi data alloca clause: mal formed variable list, missing variable name ?"
                  << std::endl;
        KaapiAbort("**** error");
    }
    char c;
    input >> std::ws >> c;
    if (c == ')') 
    {
      findend = true;
      break;
    }
    if ((c != ' ') && (c != ','))
    {
        Sg_File_Info* fileInfo = sgp->get_file_info();
        std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << " #pragma kaapi data alloca clause: unrecognized character '" << c << "'"
                  << std::endl;
        KaapiAbort("**** error");
    }
  }
  if (!findend) 
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi data alloca clause: missing ')'"
              << std::endl;
    KaapiAbort("**** error");
  }

  /* replace #pragma kaapi data alloca by the extern definition of kaapi_alloca(size_t) */
  //  SageInterface::prependStatement(decl_alloca, bbnode);
  SageInterface::removeStatement(sgp);
  
  /* traverse all the expression inside the bbnode to replace use of variables */
  nestedVarRefVisitorTraversal replace;
  replace.traverse(bbnode,postorder);
}


/** #pragma kaapi barrier parsing
*/
void DoKaapiPragmaBarrier( SgPragmaDeclaration* sgp, std::istringstream& input )
{
  std::string name;

  SgBasicBlock* bbnode = isSgBasicBlock(sgp->get_parent());
  if (bbnode ==0)
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi barrier: invalid scope declaration"
              << std::endl;
    KaapiAbort("**** error");
  }
  SgNode* nextnode = 
          sgp->get_parent()-> get_traversalSuccessorByIndex( 
              sgp->get_parent()->get_childIndex( sgp ) + 1);

  SgExprStatement* callStmt = SageBuilder::buildFunctionCallStmt
  (    "kaapi_sched_sync", 
       SageBuilder::buildVoidType(), 
       SageBuilder::buildExprListExp(),
       bbnode
  );
//  SageInterface::prependStatement(decl_sync,bbnode);
  SageInterface::replaceStatement(sgp,callStmt);

}


/** #pragma kaapi waiton parsing
*/
void DoKaapiPragmaWaiton( SgPragmaDeclaration* sgp, std::istringstream& input )
{
  std::string name;

  SgBasicBlock* bbnode = isSgBasicBlock(sgp->get_parent());
  if (bbnode ==0)
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi waiton: invalid scope declaration"
              << std::endl;
    KaapiAbort("**** error");
  }
  SgNode* nextnode = 
          sgp->get_parent()-> get_traversalSuccessorByIndex( 
              sgp->get_parent()->get_childIndex( sgp ) + 1);

  SgExprStatement* callStmt = SageBuilder::buildFunctionCallStmt
  (    "kaapi_sched_sync", 
       SageBuilder::buildVoidType(), 
       SageBuilder::buildExprListExp(),
       bbnode
  );
//  SageInterface::prependStatement(decl_sync,bbnode);
  SageInterface::replaceStatement(sgp,callStmt);
}


/** #pragma kaapi parallel
*/
void DoKaapiPragmaParallelRegion( SgPragmaDeclaration* sgp, std::istringstream& input )
{
  SgBasicBlock* bbnode = isSgBasicBlock(sgp->get_parent());
  if (bbnode ==0)
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi parallel: invalid scope declaration"
              << std::endl;
    KaapiAbort("**** error");
  }
  SgNode* nextnode = 
          sgp->get_parent()-> get_traversalSuccessorByIndex( 
              sgp->get_parent()->get_childIndex( sgp ) + 1);

  SgExprStatement* callinitStmt;
  SgExprStatement* callfinishStmt;
  if (isSgStatement(nextnode) !=0)
  {
    /* add a new fictif basicblock that contains only the Statement nextnode */
    SgStatement* nextstmt = isSgStatement(nextnode);
    SgBasicBlock* newbb = SageBuilder::buildBasicBlock();
    SageInterface::insertStatement( nextstmt, newbb, false );
    SageInterface::removeStatement( nextstmt );
    SageInterface::appendStatement( nextstmt, newbb );
    nextnode = newbb;
  }
  if (isSgBasicBlock(nextnode))
  {
    SgStatement* first_stmt;
    SgStatement* last_stmt;
    callinitStmt = SageBuilder::buildFunctionCallStmt
    (    "kaapi_begin_parallel", 
         SageBuilder::buildVoidType(), 
         SageBuilder::buildExprListExp(),
         bbnode
    );
    callfinishStmt = SageBuilder::buildFunctionCallStmt
    (    "kaapi_end_parallel", 
         SageBuilder::buildVoidType(), 
         SageBuilder::buildExprListExp(),
         bbnode
    );

    nextnode->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
    nextnode->setAttribute("kaapiparallelregion", (AstAttribute*)callinitStmt);
    first_stmt = SageInterface::getFirstStatement(isSgBasicBlock(nextnode), true);
    last_stmt = SageInterface::getLastStatement(isSgBasicBlock(nextnode));

    SageInterface::insertStatement( first_stmt, callinitStmt, true );
    SageInterface::insertStatement( last_stmt, callfinishStmt, false );
  } 
  else {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr << "****[kaapi_c2c] invalid scope for #pragma kaapi parallel."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
}

/** #pragma kaapi init/finish: flag == true if init
*/
void DoKaapiPragmaInit( SgPragmaDeclaration* sgp, std::istringstream& input, bool flag )
{
  std::string name;

  SgBasicBlock* bbnode = isSgBasicBlock(sgp->get_parent());
  if (bbnode ==0)
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi waiton: invalid scope declaration"
              << std::endl;
    KaapiAbort("**** error");
  }
  bbnode->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
  SgNode* nextnode = 
          sgp->get_parent()-> get_traversalSuccessorByIndex( 
              sgp->get_parent()->get_childIndex( sgp ) + 1);

  SgExprStatement* callStmt;
  
  if (flag)
  {
    callStmt = SageBuilder::buildFunctionCallStmt
    (    "kaapi_init", 
         SageBuilder::buildVoidType(), 
         SageBuilder::buildExprListExp(
           SageBuilder::buildIntVal (0),
           SageBuilder::buildIntVal (0)
         ),
         bbnode
    );
  }
  else {
    callStmt = SageBuilder::buildFunctionCallStmt
    (    "kaapi_finalize", 
         SageBuilder::buildVoidType(), 
         SageBuilder::buildExprListExp(
         ),
         bbnode
    );
  }
//  SageInterface::prependStatement(decl_sync,bbnode);
  SageInterface::replaceStatement(sgp,callStmt);
}




/* Find all calls to task 
*/
struct OneCall {
  OneCall( 
    SgBasicBlock*       _bb,
    SgFunctionCallExp*  _fc,
    SgExprStatement*    _s,
    KaapiTaskAttribute* _kta
  ) : bb(_bb), fc(_fc), statement(_s), kta(_kta) {}

  SgBasicBlock*       bb;
  SgFunctionCallExp*  fc;
  SgExprStatement*    statement;
  KaapiTaskAttribute* kta;
};

class KaapiTaskCallTraversal : public AstSimpleProcessing {
public:
  KaapiTaskCallTraversal()
  {}
  virtual void visit(SgNode* node)
  {
    /* propagate the parallelregion attribut to all nodes which has parent within */
    if ((node->get_parent() !=0) && (node->get_parent()->getAttribute("kaapiisparallelregion") !=0))
    {
      node->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
    }

#if 0
    if (isSgScopeStatement(node))
    {
      SgScopeStatement* bb = isSgScopeStatement(node);
      if (bb->get_parent()->getAttribute("kaapiisparallelregion") !=0)
      {
        bb->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
#if 0
        std::cout << "Set the scope statement: " << bb << ", name:" << bb->class_name()
                  << " as a parallel region because its parent:" << bb->get_parent() 
                  << " was also a parallel region"
                  << std::endl;
#endif
        switch (bb->variantT()) /* see ROSE documentation if missing node or node */
        {
          case V_SgIfStmt:
          {
            SgIfStmt* ifstmt = isSgIfStmt(bb);
            bb = isSgScopeStatement( ifstmt->get_true_body() );
            if (bb !=0)
              bb->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
            
            bb = isSgScopeStatement( ifstmt->get_false_body() );
            if (bb !=0)
              bb->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
            break;
          }
          case V_SgDoWhileStmt: 
          {
            SgDoWhileStmt* dostmt = isSgDoWhileStmt(bb);
            bb = isSgScopeStatement( dostmt->get_body() );
            if (bb !=0)
              bb->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
            break;
          }
          case V_SgWhileStmt:
          {
            SgWhileStmt* whilestmt = isSgWhileStmt(bb);
            bb = isSgScopeStatement( whilestmt->get_body() );
            if (bb !=0)
              bb->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
            break;
          }

          case V_SgForStatement:
          {
            SgForStatement* forstmt = isSgForStatement(bb);
            bb = isSgScopeStatement( forstmt->get_loop_body() );
            if (bb !=0)
              bb->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
            break;
          }

          case V_SgSwitchStatement:
          {
            SgSwitchStatement* switchstmt = isSgSwitchStatement(bb);
            bb = isSgScopeStatement( switchstmt->get_body() );
            if (bb !=0)
              bb->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
            break;
          }
          
          case V_SgForAllStatement: /* for fortran */
          case V_SgFortranDo:
          default: break;
        }
      }
    }
    if (isSgCaseOptionStmt(node))
    {
      SgCaseOptionStmt* casestmt = isSgCaseOptionStmt(node);
      if (casestmt->get_parent()->getAttribute("kaapiisparallelregion") !=0)
      {
        casestmt->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
        casestmt->get_body()->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
      }
    }
#endif
    
    if (isSgExprStatement(node))
    {
      SgExprStatement*    exprstatement = isSgExprStatement(node);
      if (SageInterface::getScope(exprstatement)->getAttribute("kaapiisparallelregion") ==0) 
      {
#if 0
        std::cout << "Function call is not a task, because its scope is not parallel:"
                  << SageInterface::getScope(exprstatement) 
                  << std::endl;
#endif
        return;
      }
      if (exprstatement->getAttribute("kaapinotask") !=0) 
        return;
      if (exprstatement->getAttribute("kaapiwrappercall") !=0) 
        return; /* call made by the wrapper */
        
      SgFunctionCallExp* fc = isSgFunctionCallExp( exprstatement->get_expression() );
      if (fc !=0)
      {
        SgFunctionDeclaration* fdecl = fc->getAssociatedFunctionDeclaration();
        if (fdecl !=0)
        {
  #if 0
          std::cout << "*** Found function call expr: '" << fdecl->get_name().str()
                    << "' decl: " << fdecl
                    << std::endl;
  #endif
          KaapiTaskAttribute* kta = (KaapiTaskAttribute*)fdecl->getAttribute("kaapitask");
          if (kta !=0)
          {
#if 0
            std::string mangled_name = fdecl->get_name().str();
            mangled_name = mangled_name + fdecl->get_mangled_name();
            Sg_File_Info* fileInfo = fc->get_file_info();
            std::cerr << "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                      << " \n*********** This is a task call to function: '" << mangled_name 
                      << "', Parent node is a: '" << fc->get_parent()->class_name()
                      << "', Grand Parent node is a: '" << fc->get_parent()->get_parent()->class_name() << "'\n"
                      << std::endl;
#endif
            /* store both the container (basic block) and the funccall expr */
            SgBasicBlock* bbnode = isSgBasicBlock( fc->get_parent()->get_parent() );
            all_listbb.insert( bbnode );
            _listcall.push_back( OneCall(bbnode, fc, exprstatement, kta) );

#if 0 // TG no important here: see below in the main: when TemplateInstance are processed
            SgTemplateInstantiationFunctionDecl* sg_tmpldecl = isSgTemplateInstantiationFunctionDecl( fdecl );
            if (sg_tmpldecl !=0)
            {
              std::cerr << "This is a call to a template function instanciation\n" << std::endl;
              KaapiPragmaString* kps 
                = (KaapiPragmaString*)sg_tmpldecl->get_templateDeclaration()->getAttribute("kaapi_templatetask");
              if (kps != 0)
              {
                std::cerr << "This is a call to a TASK template function instanciation, definition=" 
                          << sg_tmpldecl->get_definition() << "\n" << std::endl;
                if (sg_tmpldecl->get_definition() !=0)
                  all_template_instanciate_definition.insert(sg_tmpldecl->get_definition());
              }
            }
#endif
          }
        }
      }
    }
  }

public:
  std::list<OneCall> _listcall;
};


/* 
*/
void DoKaapiTaskCall( KaapiTaskCallTraversal* ktct, SgGlobal* gscope )
{
  /* Insert __kaapi_thread in each bbnode */
  std::set<SgBasicBlock*>::iterator ibb;
  for (ibb = all_listbb.begin(); ibb != all_listbb.end(); ++ibb)
  {
    /* if it is not alread done: add __kaapi_thread variable */
    SgVariableSymbol* newkaapi_threadvar = 
        SageInterface::lookupVariableSymbolInParentScopes(
            "__kaapi_thread", 
            *ibb 
    );
    if (newkaapi_threadvar ==0)
    {
      buildInsertDeclarationKaapiThread(*ibb);
    }
  }

  std::list<OneCall>::iterator ifc;
  for (ifc = ktct->_listcall.begin(); ifc != ktct->_listcall.end(); ++ifc)
  {
    OneCall& oc = *ifc;
    static int cnt = 0;
    std::ostringstream arg_name;
    arg_name << "__kaapi_arg_" << cnt++;
    SgClassType* classtype =new SgClassType(oc.kta->paramclass->get_firstNondefiningDeclaration());

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
                SageBuilder::buildVarRefExp ("__kaapi_thread", oc.bb ),
                SageBuilder::buildSizeOfOp( classtype )
              ),
              oc.bb
            ),
            SageBuilder::buildPointerType(classtype)
          )
        ),
        oc.bb
      );
    
    SageInterface::insertStatement(oc.statement, variableDeclaration, false);
    SageInterface::removeStatement(oc.statement);
    SgStatement* last_statement = variableDeclaration;

    SgExpressionPtrList& listexpr = oc.fc->get_args()->get_expressions();
    SgExpressionPtrList::iterator iebeg;
    int i = 0;
    for (iebeg = listexpr.begin(); iebeg != listexpr.end(); ++iebeg, ++i)
    {
      SgStatement* assign_statement;
      std::ostringstream fieldname;
      if (oc.kta->formal_param[i].mode == KAAPI_V_MODE)
        fieldname << arg_name.str() << "->f" << i;
      else 
        fieldname << arg_name.str() << "->f" << i << ".data";

      assign_statement = SageBuilder::buildExprStatement(
        SageBuilder::buildAssignOp(
          /* dummy-> */
          SageBuilder::buildOpaqueVarRefExp (fieldname.str(),oc.bb),
          /* expr */
          *iebeg
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
              SageBuilder::buildVarRefExp ("__kaapi_thread", oc.bb )
            ),
            oc.bb
          )
        ),
        oc.bb
      );
    
    SageInterface::insertStatement(last_statement, taskDeclaration, false);
    last_statement = taskDeclaration;

    SgStatement* init_task_statement = SageBuilder::buildExprStatement(
      SageBuilder::buildFunctionCallExp(
        "kaapi_task_initdfg",
        SageBuilder::buildVoidType(),
        SageBuilder::buildExprListExp(
          SageBuilder::buildVarRefExp (task_name.str(), oc.bb ),
          SageBuilder::buildFunctionRefExp (oc.kta->wrapper_decl),// oc.bb ),
          SageBuilder::buildVarRefExp (arg_name.str(), oc.bb )
        ),
        oc.bb
      )
    );
    SageInterface::insertStatement(last_statement, init_task_statement, false);
    last_statement = init_task_statement;

    SgStatement* push_task_statement = SageBuilder::buildExprStatement(
      SageBuilder::buildFunctionCallExp(
        "kaapi_thread_pushtask",
        SageBuilder::buildVoidType(),
        SageBuilder::buildExprListExp(
          SageBuilder::buildVarRefExp ("__kaapi_thread", oc.bb )
        ),
        oc.bb
      )
    );
    SageInterface::insertStatement(last_statement, push_task_statement, false);
    last_statement = push_task_statement;


#if 0
    kaapi_task_ROSE_type
    kaapi_task_initdfg( task1, fibo_body, kaapi_thread_pushdata(thread, sizeof(fibo_arg_t)) );
#endif
  }
}


/* Apply to every pragma to process #pragma kaapi 
   - for each #pragma kaapi task : stores the signature information and verify some assumption
   about the expression. When the function call has 
   - 
*/
class KaapiPragma {
public:
  KaapiPragma()
  {
  }
  
  void operator()( SgNode* node )
  {
    if (node->variantT() == V_SgPragmaDeclaration) 
    {
      SgPragmaDeclaration* sgp = isSgPragmaDeclaration(node);
      
      // Parse if it is a Kaapi Pragma
      std::string pragma_string = sgp->get_pragma()->get_pragma();

      std::istringstream input( pragma_string );
      std::string name;
      input >> std::ws >> name;
      
      /* css compatibility */
      if (name == "css") name = "kaapi";
      if (name != "kaapi") return;
      
      input >> std::ws >> name;
      
      /* Parallel kaapi region
         add begin_parallel_region and end_parallel_region between the basic block
      */
      if (name == "parallel")
      {
        DoKaapiPragmaParallelRegion( sgp, input );
      } 

      /* Case of task definition: because do no processing but register what to do
         then also pass the string in order the next step will be able to do parsing.
      */
      else if (name == "task")
      {
        DoKaapiPragmaTask( sgp, pragma_string );
      } /* end for task */

      else if (name == "notask")
      {
        DoKaapiPragmaNoTask( sgp, input );
      } 
       
      /* Case of barrier definition
      */
      else if (name == "barrier") 
      {
        DoKaapiPragmaBarrier( sgp, input );
      }

      /* Case of waiton clause
      */
      else if (name == "waiton") 
      {
        DoKaapiPragmaWaiton( sgp, input );
      }

      /* Case of variable scope
      */
      else if (name == "data") 
      {
        DoKaapiPragmaData( node, input );
      }

      /* Case of init/finalize
      */
      else if ((name == "init") || (name == "start"))
      {
        DoKaapiPragmaInit( sgp, input, true );
      }
      else if (name == "finish") 
      {
        DoKaapiPragmaInit( sgp, input, false );
      }
      else {
        Sg_File_Info* fileInfo = sgp->get_file_info();
        std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << " unknown #pragma kaapi " << name
                  << std::endl;
        KaapiAbort("*** Error");
      }
    }
  }  
};


int main(int argc, char **argv) 
{
  SgProject *project = frontend(argc, argv);
    
  KaapiPragma pragmaKaapi;
  
  /** Add #include <kaapi.h> to each input file
  */
  int nfile = project->numberOfFiles();
  for (int i=0; i<nfile; ++i)
  {
    SgSourceFile* file = isSgSourceFile((*project)[i]);
    if (file !=0)
    {
      /* else add extern definition in the scope */
      SgGlobal* gscope = file->get_globalScope();

      kaapi_access_ROSE_type = SageBuilder::buildOpaqueType ("kaapi_access_t", gscope);
      kaapi_task_ROSE_type = SageBuilder::buildOpaqueType ("kaapi_task_t", gscope);
      kaapi_thread_ROSE_type = SageBuilder::buildOpaqueType ("kaapi_thread_t", gscope);
      
#if 0
      SageInterface::insertHeader ("kaapi.h", PreprocessingInfo::after, false, gscope);
#else
      SageInterface::addTextForUnparser(file->get_globalScope(),
                  "#include <kaapi.h>\n",
                  AstUnparseAttribute::e_before
      );
#endif
 
      /* declare kaapi_init function */
      static SgName name_init("kaapi_init");
      SgFunctionDeclaration *decl_kaapi_init = SageBuilder::buildNondefiningFunctionDeclaration(
          name_init, 
          SageBuilder::buildPointerType(SageBuilder::buildVoidType()), 
          SageBuilder::buildFunctionParameterList( 
            SageBuilder::buildFunctionParameterTypeList( 
              SageBuilder::buildPointerType (SageBuilder::buildIntType()),
              SageBuilder::buildPointerType (SageBuilder::buildPointerType(
                        SageBuilder::buildPointerType(SageBuilder::buildCharType()))
              ) 
            ) 
          ),
          gscope
      );
      ((decl_kaapi_init->get_declarationModifier()).get_storageModifier()).setExtern();

      /* declare kaapi_finalize */
      static SgName name_finalize("kaapi_finalize");
      SgFunctionDeclaration *decl_kaapi_finalize = SageBuilder::buildNondefiningFunctionDeclaration
          (name_finalize, SageBuilder::buildVoidType(), SageBuilder::buildFunctionParameterList(),gscope);
      ((decl_kaapi_finalize->get_declarationModifier()).get_storageModifier()).setExtern();


      /* declare kaapi_begin_parallel */
      static SgName name_beginparallel("kaapi_begin_parallel");
      SgFunctionDeclaration *decl_kaapi_beginparallel 
        = SageBuilder::buildNondefiningFunctionDeclaration(
                name_beginparallel, 
                SageBuilder::buildVoidType(), SageBuilder::
                buildFunctionParameterList(),
                gscope
      );
      ((decl_kaapi_beginparallel->get_declarationModifier()).get_storageModifier()).setExtern();

      /* declare kaapi_end_parallel */
      static SgName name_endparallel("kaapi_end_parallel");
      SgFunctionDeclaration *decl_kaapi_endparallel 
        = SageBuilder::buildNondefiningFunctionDeclaration(
                name_endparallel, 
                SageBuilder::buildVoidType(), SageBuilder::
                buildFunctionParameterList(),
                gscope
      );
      ((decl_kaapi_endparallel->get_declarationModifier()).get_storageModifier()).setExtern();

      /* declare kaapi_sync */
      static SgName name_sync("kaapi_sched_sync");
      SgFunctionDeclaration *decl_kaapi_sync = SageBuilder::buildNondefiningFunctionDeclaration
          (name_sync, SageBuilder::buildVoidType(), SageBuilder::buildFunctionParameterList(),gscope);
      ((decl_kaapi_sync->get_declarationModifier()).get_storageModifier()).setExtern();


      /* declare kaapi_alloca function */
      static SgName name_alloca("kaapi_alloca");
      SgFunctionDeclaration *decl_alloca = SageBuilder::buildNondefiningFunctionDeclaration(
          name_alloca, 
          SageBuilder::buildPointerType(kaapi_thread_ROSE_type), 
          SageBuilder::buildFunctionParameterList( 
            SageBuilder::buildFunctionParameterTypeList( 
              SageBuilder::buildUnsignedLongType () 
            ) 
          ),
          gscope
      );
      ((decl_alloca->get_declarationModifier()).get_storageModifier()).setExtern();

      /* declare kaapi_self_thread function */
      static SgName name_selfthread("kaapi_self_thread");
      SgFunctionDeclaration *decl_selfthread = SageBuilder::buildNondefiningFunctionDeclaration(
          name_selfthread, 
          SageBuilder::buildPointerType(kaapi_thread_ROSE_type), 
          SageBuilder::buildFunctionParameterList( 
              SageBuilder::buildFunctionParameterTypeList( 
                SageBuilder::buildVoidType() 
              ) 
          ),
          gscope
      );
      ((decl_selfthread->get_declarationModifier()).get_storageModifier()).setExtern();

      /* declare kaapi_thread_toptask function */
      static SgName name_thread_toptask("kaapi_thread_toptask");
      SgFunctionDeclaration *decl_thread_toptask = SageBuilder::buildNondefiningFunctionDeclaration(
          name_thread_toptask, 
          SageBuilder::buildPointerType(kaapi_task_ROSE_type), 
          SageBuilder::buildFunctionParameterList( 
              SageBuilder::buildFunctionParameterTypeList( 
                SageBuilder::buildPointerType(kaapi_thread_ROSE_type)
              ) 
          ),
          gscope
      );
      ((decl_thread_toptask->get_declarationModifier()).get_storageModifier()).setExtern();

      /* declare kaapi_task_initdfg function */
      static SgName name_task_init("kaapi_task_initdfg");
      SgFunctionDeclaration *decl_task_init = SageBuilder::buildNondefiningFunctionDeclaration(
          name_task_init, 
          SageBuilder::buildVoidType(), 
          SageBuilder::buildFunctionParameterList( 
              SageBuilder::buildFunctionParameterTypeList( 
                SageBuilder::buildPointerType(kaapi_task_ROSE_type), 
                SageBuilder::buildPointerType(SageBuilder::buildVoidType()), 
                SageBuilder::buildPointerType(SageBuilder::buildVoidType())
              )
          ),
          gscope
      );
      ((decl_task_init->get_declarationModifier()).get_storageModifier()).setExtern();

      /* declare kaapi_thread_toptask function */
      static SgName name_thread_pushtask("kaapi_thread_pushtask");
      SgFunctionDeclaration *decl_thread_pushtask = SageBuilder::buildNondefiningFunctionDeclaration(
          name_thread_pushtask, 
          SageBuilder::buildVoidType(), 
          SageBuilder::buildFunctionParameterList( 
              SageBuilder::buildFunctionParameterTypeList( 
                SageBuilder::buildPointerType(kaapi_thread_ROSE_type)
              )
          ),
          gscope);
      ((decl_thread_pushtask->get_declarationModifier()).get_storageModifier()).setExtern();

      /* Process #pragma */
      Rose_STL_Container<SgNode*> pragmaDeclarationList = NodeQuery::querySubTree (project,V_SgPragmaDeclaration);
      Rose_STL_Container<SgNode*>::iterator i;
      for (  i = pragmaDeclarationList.begin(); i != pragmaDeclarationList.end(); i++)
      {
        pragmaKaapi( *i );
      }

      /* add the template function declaration in function task list iff the TemplateDeclaration
         was marked as "task"
      */
      Rose_STL_Container<SgNode*> templateDeclarationList 
        = NodeQuery::querySubTree (project,V_SgTemplateInstantiationFunctionDecl);
      Rose_STL_Container<SgNode*>::iterator itmpldecl;
      for (  itmpldecl = templateDeclarationList.begin(); itmpldecl != templateDeclarationList.end(); itmpldecl++)
      {
        SgTemplateInstantiationFunctionDecl* sg_tmpldecl = isSgTemplateInstantiationFunctionDecl( *itmpldecl );
        KaapiPragmaString* kps 
          = (KaapiPragmaString*)sg_tmpldecl->get_templateDeclaration()->getAttribute("kaapi_templatetask");
        if (kps != 0)
        {
          /* here two cases: 
              1/ add the declaration of an instance to generate the wrapper & task arguments
              2/ add the definition to be generated if null
          */
          std::string mangled_name = sg_tmpldecl->get_templateName().str();
          mangled_name = mangled_name + sg_tmpldecl->get_mangled_name();
          
          if (all_template_instanciate.find(mangled_name) == all_template_instanciate.end())
          {
            all_task_func_decl.push_back( std::make_pair(sg_tmpldecl, kps->pragma_string) );
            all_template_instanciate.insert( mangled_name );
#if 0
            std::cout << "*** Found Task TemplateFunctionDeclaration !!!!!, node=" << sg_tmpldecl 
                      << ", name: '" << sg_tmpldecl->get_mangled_name().str() << "'"
                      << ", definition: " << sg_tmpldecl->get_definition()
                      << std::endl;
#endif
          }

          if (sg_tmpldecl->get_definition() !=0)
          {
            all_template_instanciate_definition.insert(sg_tmpldecl->get_definition());
          }
        }
      }

      /* Generate information for each task */
      ListTaskFunctionDeclaration::iterator func_decl_i;
      for (func_decl_i =all_task_func_decl.begin(); func_decl_i != all_task_func_decl.end(); ++func_decl_i)
      {
        std::string& pragma_string = func_decl_i->second;
        std::istringstream input(pragma_string);
        /* re-parse 'kaapi' or 'css' and 'task' */
        std::string name;
        input >> std::ws >> name >> std::ws >> name;
        
        /* ok inputstream is correctly positionned */
        DoKaapiTaskDeclaration( func_decl_i->first, input );
      }      


      /* traverse all the expression inside the project and try to find function call expr or statement 
         and replace them by task creation if the instruction is in a kaapiisparallelregion.
         Preorder traversal in order to traverse root nodes before childs.
      */
      KaapiTaskCallTraversal taskcall_replace;
      taskcall_replace.traverse(project,preorder);
      DoKaapiTaskCall( &taskcall_replace, gscope );
      
      /* Add explicit instanciation to template instance */
      std::set<SgFunctionDefinition*>::iterator func_def_i;
      for ( func_def_i =all_template_instanciate_definition.begin(); 
            func_def_i != all_template_instanciate_definition.end(); 
            ++func_def_i
      )
      {
//        std::cerr << "Found a template instanciation function to generate" << std::endl;
        SgTemplateInstantiationFunctionDecl* sg_tmpldecl 
            = isSgTemplateInstantiationFunctionDecl((*func_def_i)->get_parent() );
        if (sg_tmpldecl ==0)
          KaapiAbort("*** Error: bad assertion, should be a SgTemplateInstantiationFunctionDecl");

        KaapiPragmaString* kps 
          = (KaapiPragmaString*)sg_tmpldecl->get_templateDeclaration()->getAttribute("kaapi_templatetask");
        if (kps ==0)
          KaapiAbort("*** Error: bad assertion, should be a TemplateFunction declaration task");

        /* ok add textual definition after the forward declaration of the wrapper */
        std::string mangled_name = sg_tmpldecl->get_templateName().str();
        mangled_name = mangled_name + sg_tmpldecl->get_mangled_name();
        std::map<std::string,KaapiTaskAttribute*>::iterator itask = all_manglename2tasks.find(mangled_name);
        if (itask == all_manglename2tasks.end())
          KaapiAbort("*** Error: bad assertion, should found the task definition");
        
        SgUnparse_Info* sg_info = new SgUnparse_Info;
        sg_info->unset_forceQualifiedNames();            
#if 0
        std::cerr << "Core=='\n" 
                  << sg_tmpldecl->get_definition()->unparseToString(sg_info)
                  << "\n'" << std::endl;
#endif
        SageInterface::addTextForUnparser(itask->second->wrapper_decl,
                    sg_tmpldecl->get_definition()->unparseToString(sg_info),
                    AstUnparseAttribute::e_before
        );
      } 
    }
  }

  /* Generate the format */
  time_t t;
  time(&t);
  std::ofstream fout("kaapi-format.cpp");
  fout << "#include \"kaapi.h\"" << std::endl;
  fout << "/* This file is automatically generated */\n"
       << "/***** Date: " << ctime(&t) << " *****/\n" << std::endl;
    
  std::list<KaapiTaskAttribute*>::iterator begin_task;
  for (begin_task = all_tasks.begin(); begin_task != all_tasks.end(); ++begin_task)
  {
    DoKaapiGenerateFormat( fout, *begin_task );
  }

  project->unparse();
  return 0;
}




/** Utility
*/
/**
*/
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
     SgClassDeclaration* classDeclaration = new SgClassDeclaration(fileInfo,name.c_str(),SgClassDeclaration::e_struct,NULL,classDefinition);
     assert(classDeclaration != NULL);

  // Set the defining declaration in the defining declaration!
     classDeclaration->set_definingDeclaration(classDeclaration);

  // Set the non defining declaration in the defining declaration (both are required)
     SgClassDeclaration* nondefiningClassDeclaration = new SgClassDeclaration(fileInfo,name.c_str(),SgClassDeclaration::e_struct,NULL,NULL);
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


#if 0
SgClassDeclaration*
buildClassDeclarationAndDefinition (const std::string& name, SgScopeStatement* scope)
   {
  // This function builds a class declaration and definition 
  // (both the defining and nondefining declarations as required).

  // This is the class definition (the fileInfo is the position of the opening brace)
     SgClassDefinition* classDefinition   = new SgClassDefinition(SOURCE_POSITION);
     assert(classDefinition != NULL);

  // Set the end of construct explictly (where not a transformation this is the location of the closing brace)
     classDefinition->set_endOfConstruct(SOURCE_POSITION);

  // This is the defining declaration for the class (with a reference to the class definition)
     SgClassDeclaration* classDeclaration = new SgClassDeclaration(SOURCE_POSITION,name.c_str(),SgClassDeclaration::e_struct,NULL,classDefinition);
     assert(classDeclaration != NULL);
     classDeclaration->set_endOfConstruct(SOURCE_POSITION);

  // Set the defining declaration in the defining declaration!
     classDeclaration->set_definingDeclaration(classDeclaration);

  // Set the non defining declaration in the defining declaration (both are required)
     SgClassDeclaration* nondefiningClassDeclaration = new SgClassDeclaration(SOURCE_POSITION,name.c_str(),SgClassDeclaration::e_struct,NULL,NULL);
     assert(classDeclaration != NULL);
     nondefiningClassDeclaration->set_endOfConstruct(SOURCE_POSITION);
     nondefiningClassDeclaration->set_type(SgClassType::createType(nondefiningClassDeclaration));

  // Set the internal reference to the non-defining declaration
     classDeclaration->set_firstNondefiningDeclaration(nondefiningClassDeclaration);
     classDeclaration->set_type (nondefiningClassDeclaration->get_type());

  // Set the defining and no-defining declarations in the non-defining class declaration!
     nondefiningClassDeclaration->set_firstNondefiningDeclaration(nondefiningClassDeclaration);
     nondefiningClassDeclaration->set_definingDeclaration(classDeclaration);

  // Set the nondefining declaration as a forward declaration!
     nondefiningClassDeclaration->setForward();

  // Liao (2/13/2008), symbol for the declaration
     SgClassSymbol* mysymbol = new SgClassSymbol(nondefiningClassDeclaration);
     scope->insert_symbol(name, mysymbol);

  // Don't forget the set the declaration in the definition (IR node constructors are side-effect free!)!
     classDefinition->set_declaration(classDeclaration);

  // set the scope explicitly (name qualification tricks can imply it is not always the parent IR node!)
     classDeclaration->set_scope(scope);
     nondefiningClassDeclaration->set_scope(scope);

  //set parent
     classDeclaration->set_parent(scope);
     nondefiningClassDeclaration->set_parent(scope);

  // some error checking
     assert(classDeclaration->get_definingDeclaration() != NULL);
     assert(classDeclaration->get_firstNondefiningDeclaration() != NULL);
     assert(classDeclaration->get_definition() != NULL);

     ROSE_ASSERT(classDeclaration->get_definition()->get_parent() != NULL);

     return classDeclaration;
   }
#endif


SgClassDeclaration* buildStructDeclaration ( 
    SgScopeStatement* scope,
    const std::vector<SgType*>& memberTypes, 
    const std::vector<std::string>& memberNames,
    const std::string& structName
)
{
#if 0
     SgClassDefinition* classDefinition   = new SgClassDefinition(SOURCE_POSITION);
     assert(classDefinition != NULL);

     classDefinition->set_endOfConstruct(SOURCE_POSITION);
     SgClassDeclaration* classDeclaration = new SgClassDeclaration(
         SOURCE_POSITION,name.c_str(),SgClassDeclaration::e_struct,NULL,classDefinition);
     assert(classDeclaration != NULL);
     classDeclaration->set_endOfConstruct(SOURCE_POSITION);

     classDeclaration->set_definingDeclaration(classDeclaration);

#else
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
#endif
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
void buildInsertDeclarationKaapiThread( SgBasicBlock* bbnode )
{
  SgExprStatement* initregion = (SgExprStatement*)bbnode->getAttribute("kaapiparallelregion");
  if (initregion == 0)
  {
    SgVariableDeclaration* newkaapi_thread = SageBuilder::buildVariableDeclaration ( 
      "__kaapi_thread", 
      SageBuilder::buildPointerType( kaapi_thread_ROSE_type ), 
      SageBuilder::buildAssignInitializer(
        SageBuilder::buildFunctionCallExp(
          "kaapi_self_thread",
          SageBuilder::buildPointerType(kaapi_thread_ROSE_type),
          SageBuilder::buildExprListExp(
          ),
          bbnode
        ),
        0
      ),
      bbnode
    );
    SageInterface::prependStatement(newkaapi_thread, bbnode);
  }
  else {
    /* declare the variable and initialize it after the begin_parallel_region*/
    SgVariableDeclaration* newkaapi_thread = SageBuilder::buildVariableDeclaration ( 
      "__kaapi_thread", 
      SageBuilder::buildPointerType( kaapi_thread_ROSE_type ), 
      0,
      bbnode
    );
    SageInterface::prependStatement(newkaapi_thread, bbnode);
    
    SgExprStatement* exprstmt = SageBuilder::buildExprStatement(
      SageBuilder::buildAssignOp(
        SageBuilder::buildVarRefExp(
          "__kaapi_thread", 
          bbnode
        ),
        SageBuilder::buildFunctionCallExp(
          "kaapi_self_thread",
          SageBuilder::buildPointerType(kaapi_thread_ROSE_type),
          SageBuilder::buildExprListExp(
          ),
          bbnode
        )
      )
    );
    SageInterface::insertStatement( initregion, exprstmt, false );
  }
}


/***/
static void RecGenerateGetDimensionExpression(
    std::ostringstream& sout,
    KaapiTaskAttribute* kta, 
    SgExpression*       expr
)
{
  if (isSgVarRefExp(expr))
  {
    SgVarRefExp* var = isSgVarRefExp(expr);
    SgVariableSymbol * sym = var->get_symbol();
    /* look up sym->get_name() in the list of formal parameter of the declaration */
    std::map<std::string,int>::iterator iparam = kta->lookup.find( sym->get_name().str() );
    if (iparam == kta->lookup.end())
      KaapiAbort("*** Cannot find which parameter is involved in dimension expression");
    sout << "((" 
         << kta->formal_param[iparam->second].type->unparseToString() 
         << ")" << "arg->f" << iparam->second;
    if (kta->formal_param[iparam->second].mode != KAAPI_V_MODE)
      sout << ".data";
    sout << ")";
  }
  else if (isSgUnsignedLongVal(expr))
  {
    SgUnsignedLongVal* value = isSgUnsignedLongVal(expr);
    sout << value->get_value();
  }
  else if (isSgSizeOfOp(expr))
  {
    SgSizeOfOp* esizeof = isSgSizeOfOp(expr);
    sout << " sizeof( ";
    if (esizeof->get_operand_expr() !=0)
      RecGenerateGetDimensionExpression(sout, kta, esizeof->get_operand_expr() );
    else /* here ??? */
      sout << SageInterface::get_name(esizeof->get_type());
  }
  else if (isSgCastExp(expr))
  {
    SgCastExp* cast = isSgCastExp(expr);
    sout << " ( " << SageInterface::get_name(cast->get_type()) << ")";
    RecGenerateGetDimensionExpression(sout, kta, cast->get_originalExpressionTree());
  }
  else if (isSgMultiplyOp(expr))
  {
    SgMultiplyOp* op = isSgMultiplyOp(expr);
    sout << "(";
    RecGenerateGetDimensionExpression(sout, kta, op->get_lhs_operand());
    sout << " * ";
    RecGenerateGetDimensionExpression(sout, kta, op->get_rhs_operand());
    sout << ")";
  }
  else if (isSgDivideOp(expr))
  {
    SgDivideOp* op = isSgDivideOp(expr);
    sout << "(";
    RecGenerateGetDimensionExpression(sout, kta, op->get_lhs_operand());
    sout << " / ";
    RecGenerateGetDimensionExpression(sout, kta, op->get_rhs_operand());
    sout << ")";
  }
  else if (isSgModOp(expr))
  {
    SgModOp* op = isSgModOp(expr);
    sout << "(";
    RecGenerateGetDimensionExpression(sout, kta, op->get_lhs_operand());
    sout << " % ";
    RecGenerateGetDimensionExpression(sout, kta, op->get_rhs_operand());
    sout << ")";
  }
  else if (isSgAddOp(expr))
  {
    SgAddOp* op = isSgAddOp(expr);
    sout << "(";
    RecGenerateGetDimensionExpression(sout, kta, op->get_lhs_operand());
    sout << " + ";
    RecGenerateGetDimensionExpression(sout, kta, op->get_rhs_operand());
    sout << ")";
  }
  else if (isSgSubtractOp(expr))
  {
    SgSubtractOp* op = isSgSubtractOp(expr);
    sout << "(";
    RecGenerateGetDimensionExpression(sout, kta, op->get_lhs_operand());
    sout << " - ";
    RecGenerateGetDimensionExpression(sout, kta, op->get_rhs_operand());
    sout << ")";
  }
  else {
    std::cerr << "Found an expression: @expr:" << expr << "  " << (expr == 0? "" : expr->class_name()) << std::endl;
    KaapiAbort("*** Bad expression for dimension");
  }
}


/***/
std::string GenerateGetDimensionExpression(
    KaapiTaskAttribute* kta, 
    SgExpression*       expr
)
{
  std::ostringstream expr_str;
  RecGenerateGetDimensionExpression( expr_str, kta, expr );
  return expr_str.str();
}


/***/
std::string GenerateSetDimensionExpression(
    KaapiTaskAttribute* kta, 
    SgExpression* expr,
    int index_inview
)
{
  if (index_inview != -1) return "";
  /* only generate change in the lda field */
  std::ostringstream expr_str;

  if (isSgVarRefExp(expr))
  {
    SgVarRefExp* var = isSgVarRefExp(expr);
    SgVariableSymbol * sym = var->get_symbol();
    /* look up sym->get_name() in the list of formal parameter of the declaration */
    std::map<std::string,int>::iterator iparam = kta->lookup.find( sym->get_name().str() );
    if (iparam == kta->lookup.end())
      KaapiAbort("****[kaapi_c2c] Cannot find which parameter is involved in lda expression");
    expr_str << "arg->f" << iparam->second;
    expr_str << " = view->lda;\n";
  }
  else {
    std::cerr << "****[kaapi_c2c] error. Cannot found lhs expression type for lda: @expr:" 
              << expr << "  " << (expr == 0? "" : expr->class_name()) << std::endl;
    KaapiAbort("****[kaapi_c2c] Expression for lda should be an identifier.");
  }

  return expr_str.str();
}

std::string ConvertCType2KaapiFormat(SgType* type)
{
  switch (type->variantT()) 
  {
    case V_SgTypeChar: return "kaapi_char_format";
    case V_SgTypeSignedChar: return "kaapi_char_format";
    case V_SgTypeUnsignedChar: return "kaapi_uchar_format";

    case V_SgTypeShort: return "kaapi_short_format";
    case V_SgTypeSignedShort: return "kaapi_short_format";
    case V_SgTypeUnsignedShort: return "kaapi_ushort_format";

    case V_SgTypeInt: return "kaapi_int_format";
    case V_SgTypeSignedInt: return "kaapi_int_format";
    case V_SgTypeUnsignedInt: return "kaapi_uint_format";

    case V_SgTypeLong: return "kaapi_long_format";
    case V_SgTypeSignedLong: return "kaapi_long_format";
    case V_SgTypeUnsignedLong: return "kaapi_ulong_format";

    case V_SgTypeLongLong: return "kaapi_longlong_format";
    case V_SgTypeSignedLongLong: return "kaapi_longlong_format";
    case V_SgTypeUnsignedLongLong: return "kaapi_ulonglong_format";

    case V_SgTypeFloat: return "kaapi_float_format";
    case V_SgTypeDouble: return "kaapi_double_format";
    case V_SgTypeLongDouble: return "kaapi_longdouble_format";

    case V_SgPointerType: return "kaapi_voidp_format";
    
    case V_SgTypedefType: return ConvertCType2KaapiFormat( isSgTypedefType(type)->get_base_type() );
    case V_SgModifierType: return ConvertCType2KaapiFormat( isSgModifierType(type)->get_base_type() );

    default: {
      std::ostringstream sout;
      sout << "<NO KAAPI FORMAT for: " << type->class_name() << ">";
      return sout.str();
    }
  }
}
