/*
** kaapi_staticsched.h
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

#include <rose.h>
#include <string>
#include <iostream>
#include <map>
#include <list>
#include <sstream>
#include <iomanip>
#include <ctype.h>
#include <time.h>
#include <stdexcept>
/* TODO LIST
  OK : fait et un peu tester
  NOK: not OK
  
OK    - gestion des args si paramètre range: begin -> access, end -> size_t (taille)
OK    - reduction & parsing de la declaration des fonctions + support moteur executif
OK/NOK- les boucles sont gérés comme suit:
        __kaapi_thread = kaapi_push_frame()
        for(... ) { ... }
        kaapi_sched_sync();
        __kaapi_thread = kaapi_pop_frame()
OK     Les break ne pose pas de problème. 
OK     Les return non plus (l'appelant exécutant les tâches créées).
NOK    Le cas des gotos n'est pas géré.
     
NOK   - vérification des accès effectués en fonction des modes déclarés
        - mode d'accès R/W/RW/CW
          * arithmetique de pointeurs
          * accès: R/RW/CW/W: rhs
          * accès: W/CW/RW: lhs
          * accès: CW pas de vérification supplémentaire sur l'ensemble des opérations 
          séquentielle.
        - portée des variables:
          * scope de déclaration supérieur à l'appel de fonction
            - ok pour les paramètres formel
            - ok si sync ou fin de boucle (sync implicite) après l'utilisation par une tâche
          * sinon émettre une erreur / warning ?

NOK   - parsing de fin des pragmas: on ne vérifie pas la fin de la phrase...

   - adaptive loop. See OpenMP canonical form of Loop. Not all kinds of expression
   can occurs in expressions of the for loop (init_expr; cond_expr; incr_expr )
        - for_each: ok

NOK   - global variable:
        - add extra parameter (ok in struct args + function call)
        - recover the global variable in the task execution body: NOK
        
   - generation of serialization operators
*/

#define SOURCE_POSITION Sg_File_Info::generateDefaultFileInfoForTransformationNode()

struct KaapiTaskAttribute; 
struct OneCall;

static std::list<KaapiTaskAttribute*> all_tasks;
static std::map<std::string,KaapiTaskAttribute*> all_manglename2tasks;

typedef std::list<std::pair<SgFunctionDeclaration*, std::string> > ListTaskFunctionDeclaration;
static ListTaskFunctionDeclaration all_task_func_decl;

/* used to mark already instanciated template */
static std::set<std::string> all_template_instanciate; 
static std::set<SgFunctionDefinition*> all_template_instanciate_definition; 

static SgType* kaapi_access_ROSE_type;
static SgType* kaapi_task_ROSE_type;
static SgType* kaapi_thread_ROSE_type;
static SgType* kaapi_frame_ROSE_type;
static SgType* kaapi_workqueue_ROSE_type;
static SgType* kaapi_stealcontext_ROSE_type;

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


/* Replace function call by task spawn
*/
void buildFunCall2TaskSpawn( OneCall* oc );
void buildInsertSaveRestoreFrame( SgScopeStatement* forloop );
SgScopeStatement* buildConvertLoop2Adaptative( SgScopeStatement* loop );

/* Build expression to declare a __kaapi_thread variable 
*/
SgVariableDeclaration* buildInsertDeclarationKaapiThread( SgScopeStatement* bbnode );


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
  KAAPI_VOID_MODE = 0,
  KAAPI_W_MODE,
  KAAPI_R_MODE,
  KAAPI_RW_MODE,
  KAAPI_CW_MODE,
  KAAPI_V_MODE,
  KAAPI_GLOBAL_MODE,
  
  KAAPI_HIGHPRIORITY_MODE /* not really a mode. Only used to parse css program */
};


enum KaapiOperator_t {
  KAAPI_NO_OPERATOR = 0,
  KAAPI_ADD_OPERATOR,
  KAAPI_SUB_OPERATOR,
  KAAPI_MUL_OPERATOR,
  KAAPI_AND_OPERATOR,
  KAAPI_OR_OPERATOR,
  KAAPI_XOR_OPERATOR,
  KAAPI_LAND_OPERATOR, /* logical and */
  KAAPI_LOR_OPERATOR, /* logical and */
  
  KAAPI_USER_OPERATOR
};

struct KaapiReduceOperator_t {
  KaapiReduceOperator_t() 
   : isbuiltin(false), name(0), name_reducor(0), name_redinit(0)
  {}
  KaapiReduceOperator_t(const char* name, const char* reducor, const char* redinit) 
   : isbuiltin(true), name(name), name_reducor(reducor), name_redinit(redinit)
  {}
  bool  isbuiltin;
  std::string name;
  std::string name_reducor;
  std::string name_redinit;
};
static std::map<std::string,KaapiReduceOperator_t*> kaapi_user_definedoperator;

enum KaapiStorage_t {
  KAAPI_BAD_STORAGE = 0,
  KAAPI_ROW_MAJOR = 1,
  KAAPI_COL_MAJOR = 2
};

enum KaapiParamAttribute_t {
  KAAPI_BAD_PARAM_TYPE = 0,
  KAAPI_ARRAY_NDIM_TYPE = 1,
  KAAPI_OPEN_RANGE_TYPE = 2,
  KAAPI_CLOSE_RANGE_TYPE = 3
};

/* Store view of each parameter */
class KaapiParamAttribute : public AstAttribute {
public:
  KaapiParamAttribute ()
   : type(KAAPI_BAD_PARAM_TYPE),  storage(KAAPI_ROW_MAJOR), lda(0), dim(0) 
  { 
    memset(ndim, 0, sizeof(ndim) );
  }
  KaapiParamAttribute_t  type;
  union {
    struct {
      const char*           name;
      KaapiStorage_t        storage;
      SgExpression*         lda;
      size_t                dim;
      SgExpression*         ndim[3];
    };
    struct {
      const char*           first_bound;
      SgExpression*         expr_firstbound;
      int                   index_firstbound;
      const char*           second_bound;
      SgExpression*         expr_secondbound;
      int                   index_secondbound;
    };
  };
};

struct KaapiTaskFormalParam {
  KaapiTaskFormalParam()
   : mode(KAAPI_VOID_MODE), attr(0), initname(0), type(0), kaapi_format()
  {}
  KaapiAccessMode_t      mode;
  KaapiReduceOperator_t* redop;
  KaapiParamAttribute*   attr;
  SgInitializedName*     initname;
  SgType*                type;
  std::string            kaapi_format; 
};


/**
*/
class KaapiTaskAttribute : public AstAttribute {
public:
  KaapiTaskAttribute () 
  { }

  std::string                       mangled_name;      /* internal name of the task */
  bool                              has_retval;
  KaapiTaskFormalParam              retval;
  std::vector<KaapiTaskFormalParam> formal_param;
  std::vector<KaapiTaskFormalParam> extra_param;
  std::vector<int>                  israngedecl;       /* for range declaration: 0 no, 1:begin, 2: end*/
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


/* Find all calls to task 
*/
struct OneCall {
  OneCall( 
    SgScopeStatement*   _scope,
    SgFunctionCallExp*  _fc,
    SgStatement*        _s,
    KaapiTaskAttribute* _kta,
    SgScopeStatement*   _fl
  ) : fc(_fc), scope(_scope), statement(_s), kta(_kta), forloop(_fl) {}

  SgFunctionCallExp*  fc;         /* function call exp to a task */
  SgScopeStatement*   scope;      /* enclosing scope of fc */
  SgStatement*        statement;  /* enclosing statement of fc */
  KaapiTaskAttribute* kta;        /* attached task attribute */
  SgScopeStatement*   forloop;    /* enclosing loop of fc */
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



static inline int isletter( char c)
{ return (c == '_') || isalpha(c); }
static inline int isletternum( char c)
{ return (c == '_') || isalnum(c); }


class Parser {
  const char* buffer;
  const char* rpos;
  const char* rlast;
  
  void skip_ws()
  {
    while ((*rpos == ' ') || (*rpos == '\t') || (*rpos == '\n'))
      ++rpos;
  }

  char readchar() throw(std::overflow_error)
  {
    if (rpos == rlast) 
    {
      ++rpos;
      return EOF;
    }
    if (rpos > rlast) throw std::overflow_error("empty buffer");
    return *rpos++;
  }
  void putback() 
  { --rpos; }

public:
  Parser( const char* line )
   : buffer(line), rpos(line)
  { 
    rlast = rpos + strlen(line);
  }
  
  /** Return the string of the remainder part of the line
  */
  void ParseGetLine( std::string& ident );

  /** Parse C/C++ identifier and return the last unrecognized char 
      which is putted back into the stream
      [a-zA-Z_]*[0-9]
  */
  int ParseIdentifier( std::string& ident );

  /* arithmetic grammar ... */
  /**/
  SgUnsignedLongVal* ParseExprIntegralNumber( );

  /**/
  SgVarRefExp* ParseExprIdentifier( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  /**/
  SgExpression* ParseExprConstant( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  /**/
  SgExpression* ParsePrimaryExpression( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );
  
  /**/
  SgExpression* ParseUnaryExpression( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  /**/
  SgExpression* ParseCastExpression( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  /**/
  SgExpression* ParseMultiplicativeExpression( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  /**/
  SgExpression* ParseAdditiveExpression( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  SgExpression* ParseExpression( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  KaapiStorage_t ParseStorage( 
      Sg_File_Info*       fileInfo, 
      SgScopeStatement*   scope 
  );


  KaapiAccessMode_t ParseAccessMode( 
      Sg_File_Info* fileInfo
  );

  KaapiReduceOperator_t* ParseReduceOperator( 
      Sg_File_Info* fileInfo
  );

  KaapiReduceOperator_t* ParseReductionDeclaration( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  void ParseDimension( 
      Sg_File_Info* fileInfo, 
      KaapiParamAttribute* kpa, 
      SgScopeStatement* scope 
  );

  void ParseNDimensions( 
      Sg_File_Info* fileInfo, 
      KaapiParamAttribute* kpa, 
      SgScopeStatement* scope 
  );

  void ParseRangeDeclaration( 
      Sg_File_Info*        fileInfo, 
      KaapiParamAttribute* kpa, 
      SgScopeStatement*    scope 
  );

  void ParseComplexView( 
      Sg_File_Info*        fileInfo, 
      KaapiParamAttribute* kpa, 
      SgScopeStatement*    scope 
  );

  void ParseListParamDecl( 
      Sg_File_Info*       fileInfo, 
      KaapiTaskAttribute* kta,
      KaapiAccessMode_t   mode,
      SgScopeStatement*   scope 
  );

  void ParseListParam( 
      Sg_File_Info*       fileInfo, 
      KaapiTaskAttribute* kta,
      SgScopeStatement*   scope 
  );

  void DoKaapiPragmaTask( SgPragmaDeclaration* sgp );
  void DoKaapiPragmaData( SgNode* node );
  void DoKaapiPragmaLoop( SgPragmaDeclaration* sgp );
  void DoKaapiTaskDeclaration( SgFunctionDeclaration* functionDeclaration );
  void DoKaapiPragmaNoTask( SgPragmaDeclaration* sgp );
  void DoKaapiPragmaBarrier( SgPragmaDeclaration* sgp );
  void DoKaapiPragmaWaiton( SgPragmaDeclaration* sgp );
  void DoKaapiPragmaParallelRegion( SgPragmaDeclaration* sgp );
  void DoKaapiPragmaInit( SgPragmaDeclaration* sgp, bool flag );
  void DoKaapiPragmaDeclare( SgPragmaDeclaration* sgp );
  
}; /* end parser class */




/** #pragma kaapi task parsing
*/
void Parser::DoKaapiPragmaTask( SgPragmaDeclaration* sgp )
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
  SgStatement* fnode = SageInterface::getNextStatement ( sgp );

  std::string pragma_string;
  ParseGetLine( pragma_string );

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

void Parser::DoKaapiPragmaLoop( SgPragmaDeclaration* sgp )
{
  Sg_File_Info* fileInfo = sgp->get_file_info();
  std::cerr << "****[kaapi_c2c] Found #pragma kaapi loop directive !!!!."
            << "     In filename '" << fileInfo->get_filename() 
            << "' LINE: " << fileInfo->get_line()
            << std::endl;
  SgForStatement* forloop = isSgForStatement(SageInterface::getNextStatement ( sgp ));
  if (forloop ==0)
  {
    std::cerr << "****[kaapi_c2c] #pragma kaapi loop directive must be followed by a for loop statement."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  SgScopeStatement* newloop = buildConvertLoop2Adaptative( forloop );
  if (newloop !=0)
  {
    SageInterface::insertStatement( forloop, newloop );
    SageInterface::removeStatement( forloop );
  }
}


void Parser::DoKaapiTaskDeclaration( SgFunctionDeclaration* functionDeclaration )
{
  Sg_File_Info* fileInfo = functionDeclaration->get_file_info();

  /* */
  KaapiTaskAttribute* kta = new KaapiTaskAttribute;
  kta->func_decl = functionDeclaration;
  kta->has_retval = false;

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

  const unsigned int args_size = args.size();
  SgScopeStatement* scope_declaration = functionDeclaration->get_scope();
  kta->formal_param.resize( args_size );
  kta->israngedecl.resize( args_size );
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
  ParseListParam( fileInfo, kta, scope_declaration );

  /* Add the class declaration for the parameters */
  kta->name_paramclass = std::string("__kaapi_args_") + 
      functionDeclaration->get_name().str() + 
      functionDeclaration->get_mangled_name().str();

  kta->paramclass = 
    buildClassDeclarationAndDefinition( 
      kta->name_paramclass,
      scope_declaration
    );
  
  /* Add member for each parameter */
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
        std::cerr << "****[kaapi_c2c] Warning: incorrect access mode: not a pointer type. \n"
                  << "                         Change access mode declaration to value.\n"
                  << "     In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << " formal parameter '" << kta->formal_param[i].initname->get_name().str()
                  << "' is declared as read/write/reduction but is not a pointer type. Should be declared as a value.\n"
                  << std::endl;
        kta->formal_param[i].mode = KAAPI_V_MODE;
        goto redo_selection;
        KaapiAbort("**** error");
      }
      if (kta->israngedecl[i] <= 1) /* 2 was the size */
      {
        kta->formal_param[i].kaapi_format = ConvertCType2KaapiFormat(
          isSgPointerType(kta->formal_param[i].type)->get_base_type()
        );
        member_type = kaapi_access_ROSE_type;
      }
      else {
        /* This is the end bound of a 1D range: store an integer in place of a type */
        kta->formal_param[i].mode = KAAPI_V_MODE;
        kta->formal_param[i].kaapi_format = "kaapi_int_format";
        member_type = SageBuilder::buildIntType();
      }
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


  /* Add extra parameter */
  for (unsigned int i=0; i<kta->extra_param.size(); ++i)
  {
    
    std::ostringstream name;
    name << "e" << i;
    SgType* member_type = 0;

    member_type = kta->extra_param[i].type;
    if (isSgModifierType(member_type)) 
    {
      member_type = isSgModifierType(member_type)->get_base_type();
      kta->formal_param[i].kaapi_format = ConvertCType2KaapiFormat(kta->formal_param[i].type);
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

#if 1 // handle return value
  SgType* ret_type = functionDeclaration->get_orig_return_type();
  if (isSgTypeVoid(ret_type) == NULL)
  {
    static const char* const retval_name = "__kaapi_retval";

    SgPointerType* const ret_ptrtype =
      SageBuilder::buildPointerType(ret_type);

    SgInitializedName* const init_name =
      new SgInitializedName(SgName(retval_name), ret_ptrtype);

    kta->has_retval = true;

    KaapiTaskFormalParam& param = kta->retval;

    param.mode = KAAPI_W_MODE;
    param.redop = NULL;
    param.attr = NULL;
    param.initname = init_name;
    param.type = ret_ptrtype;
    param.kaapi_format = ConvertCType2KaapiFormat
      (isSgPointerType(ret_ptrtype)->get_base_type());

    kta->lookup.insert
      (std::make_pair(init_name->get_name().str(), args_size));

    std::ostringstream member_name;
    member_name << "r";

    SgVariableDeclaration* const member_decl =
      SageBuilder::buildVariableDeclaration
      (
       member_name.str(), 
       kaapi_access_ROSE_type,
       0,
       kta->paramclass->get_definition()
       );
    member_decl->set_endOfConstruct(SOURCE_POSITION);
    kta->paramclass->get_definition()->append_member(member_decl);
  }
#endif // handle return value

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
    if (kta->israngedecl[i] <=1)
    {
      if (kta->formal_param[i].mode == KAAPI_V_MODE)
        fieldname << "thearg->f" << i;
      else {
        fieldname << "(" << kta->formal_param[i].type->unparseToString() << ")"
                   << "thearg->f" << i << ".data";
      }
    }
    else { /* bound end of a range store the size: rebuild the pointer */
      KaapiParamAttribute* kpa = kta->formal_param[i].attr;
      fieldname << "((" << kta->formal_param[kpa->index_firstbound].type->unparseToString() << ")"
                << "thearg->f" << kpa->index_firstbound << ".data)"
                << " + thearg->f" << i;
    }
    SageInterface::appendExpression(argscall,
      SageBuilder::buildOpaqueVarRefExp(fieldname.str(),wrapper_body)
    );
  }

#if 1 // handle return value
  if (kta->has_retval)
  {
    KaapiTaskFormalParam& param = kta->retval;
    std::ostringstream fieldname;
    fieldname
      << "*((" << param.type->unparseToString() << ")"
      << "thearg->r.data)";

    SgVarRefExp* const retval_expr =
      SageBuilder::buildOpaqueVarRefExp(fieldname.str(), wrapper_body);

    SgExprStatement* const callStmt = SageBuilder::buildFunctionCallStmt
      (functionDeclaration->get_name(), ret_type, argscall, wrapper_body);

    SgExprStatement* const assign_stmt = SageBuilder::buildAssignStatement
      (retval_expr, callStmt->get_expression());
    SageInterface::insertStatement( truearg_decl, assign_stmt, false );

    assign_stmt->setAttribute("kaapiwrappercall", (AstAttribute*)-1);
  }
  else
#endif // handle return value
  {
    SgExprStatement* const callStmt =
      SageBuilder::buildFunctionCallStmt
      (
       functionDeclaration->get_name(),
       SageBuilder::buildVoidType(), 
       argscall,
       wrapper_body
      );

    callStmt->setAttribute("kaapiwrappercall", (AstAttribute*)-1);
    SageInterface::insertStatement( truearg_decl, callStmt, false );
  }
  
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
void Parser::DoKaapiPragmaNoTask( SgPragmaDeclaration* sgp )
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
    {
      if (kta->israngedecl[i] <= 1)
      {
        fout << "    case " << i << ": return kaapi_memory_view_make1d( 1, sizeof("
             << kta->formal_param[i].type->unparseToString()
             << "));\n";
      }
      else { /* end of range type == int */
        fout << "    case " << i << ": return kaapi_memory_view_make1d( 1, sizeof(int));\n"; 
      }
    }
    else 
    {
      SgPointerType* ptrtype = isSgPointerType(kta->formal_param[i].type);
      if (ptrtype ==0) KaapiAbort("**** Error: bad internal assertion");
      SgType* type = ptrtype->get_base_type();
      // TODO here: add definition of the type, else we cannot compile it */
      while (isSgTypedefType( type ))
      {
        type = isSgTypedefType(type)->get_base_type();
      }
      
      if (kta->formal_param[i].attr->type == KAAPI_ARRAY_NDIM_TYPE)
      {          
        if (kta->formal_param[i].attr->dim == 0) /* in fact single element */
        {
          fout << "    case " << i << ": return kaapi_memory_view_make1d( 1" 
                              << ", sizeof(" << type->unparseToString() << "));\n";
        } 
        else if (kta->formal_param[i].attr->dim == 1)
        {
          fout << "    case " << i << ": return kaapi_memory_view_make1d( " 
                              << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->ndim[0])
                              << ", sizeof(" << type->unparseToString() << "));\n";
        } 
        else if (kta->formal_param[i].attr->dim == 2)
        {
          fout << "    case " << i << ": return kaapi_memory_view_make2d( "
                              << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->ndim[0]) << ","
                              << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->ndim[1]) << ",";
          if (kta->formal_param[i].attr->lda == 0) /* means contigous & row major == lda = dimj */
            fout              << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->ndim[1]);
          else
            fout              << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->lda);
          fout                << ", sizeof(" << type->unparseToString() << "));\n";
        }
        else if (kta->formal_param[i].attr->dim == 3)
        {
          fout << "    case " << i << ": kaapi_abort();\n";
        }
      }
      else { /* this is a begin of a range */
        fout << "    case " << i << ": return kaapi_memory_view_make1d( " 
                            << "arg->f" << kta->formal_param[i].attr->index_secondbound
                            << ",  sizeof(" << type->unparseToString() << "));\n";
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
  fout << "void " << kta->name_format << "_reducor(const struct kaapi_format_t* fmt, unsigned int i, void* sp, const void* v)\n"
       << "{\n  " << kta->name_paramclass << "* arg = (" << kta->name_paramclass << "*)sp;\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].mode == KAAPI_CW_MODE)
    {
      fout << "    case " << i << ": {\n";
      SgPointerType* ptrtype = isSgPointerType(kta->formal_param[i].type);
      if (ptrtype ==0) KaapiAbort("**** Error: bad internal assertion");
      SgType* type = ptrtype->get_base_type();
      // TODO here: add definition of the type, else we cannot compile it */
      while (isSgTypedefType( type ))
      {
        type = isSgTypedefType(type)->get_base_type();
      }

      /* pointer to the data: var */
      fout << "      " << type->unparseToString() 
           << "* var = (" << type->unparseToString() << "*) arg->f" << i << ".data;\n";
      fout << "      const " << type->unparseToString() 
           << "* value = ( const " << type->unparseToString() << "*)v;\n";
           
      KaapiReduceOperator_t* redop = kta->formal_param[i].redop;
      /* the name of the variable is known: 
         from the type, find the righ operator or function.
         If it is a builtin operator, we generate the code for the reduction
      */
      if (redop->isbuiltin)
      {
        if (kta->formal_param[i].attr->type == KAAPI_ARRAY_NDIM_TYPE)
        {
          if (kta->formal_param[i].attr->dim ==0)
          {
            fout << "      *var " << redop->name_reducor << " *value;\n";
          }
          else {
            fout << "      kaapi_memory_view_t view = " << kta->name_format << "_get_view_param(fmt, i, sp);\n";

            /* nested loop */
            for (unsigned int k=0; k<kta->formal_param[i].attr->dim; ++k)
            {
              std::ostringstream varindexloop;
              varindexloop << "ikak" << k;
              fout << "      for ( unsigned int " << varindexloop.str() << "=0; " 
                   << varindexloop.str() << " < view.size[" << k << "]; "
                   << "++" << varindexloop.str() << " )\n"
                   << "      {\n";
            }

            /* body */
            fout << "          var[ikak" << kta->formal_param[i].attr->dim-1
                 << "] " << redop->name_reducor 
                 << " value[ikak" << kta->formal_param[i].attr->dim-1 << "];\n";

            /* nested loop */
            for (unsigned int k=0; k<kta->formal_param[i].attr->dim; ++k)
            {
              std::ostringstream varindexloop;
              varindexloop << "ikak" << k;
              fout << "      } /*" << varindexloop.str() << "*/\n";
              if ((k == 0) && (kta->formal_param[i].attr->dim ==2))
              {
                fout << "      var += view.kda;\n";
              }
            }
          } /* else dim >0 */
        } /* else it is range 1d */
        else {
          fout << "      for (unsigned int k=0; k < view.size[0]; ++k)\n"
               << "        *var++ " << redop->name_reducor << " *value++;\n";
        }
      }      
      else /* not a builtin */
      {
        
      }
      fout << "    } break;\n";
    }
  }
  fout << "  }\n"
       << "}\n"
       << std::endl;

  /* format::redinit */
  //isIntegerType
  fout << "void " << kta->name_format << "_redinit(const struct kaapi_format_t* fmt, unsigned int i, const void* sp, void* v)\n"
       << "{\n  " << kta->name_paramclass << "* arg = (" << kta->name_paramclass << "*)sp;\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].mode == KAAPI_CW_MODE)
    {
      fout << "    case " << i << ": {\n";
      SgPointerType* ptrtype = isSgPointerType(kta->formal_param[i].type);
      if (ptrtype ==0) KaapiAbort("**** Error: bad internal assertion");
      SgType* type = ptrtype->get_base_type();
      // TODO here: add definition of the type, else we cannot compile it */
      while (isSgTypedefType( type ))
      {
        type = isSgTypedefType(type)->get_base_type();
      }

      /* pointer to the data: var */
      fout << "      " << type->unparseToString() 
           << "* var = (" << type->unparseToString() << "*) v;\n";
           
      KaapiReduceOperator_t* redop = kta->formal_param[i].redop;

      /* the name of the variable is known: 
         from the type, find the righ operator or function.
         If it is a builtin operator, we generate the code for the reduction
      */
      if (redop->isbuiltin)
      {
        fout << "      *var = " << redop->name_redinit << ";\n";
      }
      else {
        
      }
      fout << "    } break;\n";
    }
  }
  fout << "  }\n"
       << "}\n"
       << std::endl;


  /* format::get_task_binding */
  fout << "void " << kta->name_format << "_get_task_binding(const struct kaapi_format_t* fmt, const kaapi_task_t* t, kaapi_task_binding_t* tb)\n"
       << "{ return; }\n"
       << std::endl;
       
  /* Generate constructor function that register the format */
  fout << "/* constructor method */\n" 
       << "__attribute__ ((constructor)) void " << kta->name_format << "_constructor(void)\n"
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
       << "    " << kta->name_format << "_get_access_param,\n" /* get_access_param */
       << "    " << kta->name_format << "_set_access_param,\n" /* set_access_param */
       << "    " << kta->name_format << "_get_fmt_param,\n" /* get_fmt_param */
       << "    " << kta->name_format << "_get_view_param,\n" /* get_view_param */
       << "    " << kta->name_format << "_set_view_param,\n" /* set_view_param */
       << "    " << kta->name_format << "_reducor,\n" /* reducor */
       << "    " << kta->name_format << "_redinit,\n" /* reducor */
       << "    " << kta->name_format << "_get_task_binding\n" /* get_task_binding */
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
void Parser::DoKaapiPragmaData( SgNode* node )
{
  std::string name;
  char c;
  SgPragmaDeclaration* sgp = isSgPragmaDeclaration(node);
  
  ParseIdentifier( name );
  if (name != "alloca")
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi data: expecting 'alloca' clause, found '" << name << "'"
              << std::endl;
    KaapiAbort("**** error");
  }

//  SgBasicBlock* bbnode = isSgBasicBlock(sgp->get_parent());
  SgScopeStatement* bbnode = SageInterface::getScope(sgp);
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
  skip_ws();
  c = readchar();
  if (c != '(') 
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi data alloca clause: missing '('"
              << std::endl;
    KaapiAbort("**** error");
  }
  
  bool findend = false;
  while (1)
  {
    skip_ws();
    ParseIdentifier( name );
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
      
      /* if it is not alread done: add __kaapi_thread variable 
         in the top scope:
      */
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
    skip_ws();
    c = readchar();
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
    if (c == EOF)
      break;
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
//  SageInterface::removeStatement(sgp);
  
  /* traverse all the expression inside the bbnode to replace use of variables */
  nestedVarRefVisitorTraversal replace;
  replace.traverse(bbnode,postorder);
}


/** #pragma kaapi barrier parsing
*/
void Parser::DoKaapiPragmaBarrier( SgPragmaDeclaration* sgp )
{
  std::string name;

  SgScopeStatement* bbnode = isSgBasicBlock(SageInterface::getScope(sgp));
  if (bbnode ==0)
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi sync: invalid scope declaration"
              << std::endl;
    KaapiAbort("**** error");
  }

  SgExprStatement* callStmt = SageBuilder::buildFunctionCallStmt
  (    "kaapi_sched_sync", 
       SageBuilder::buildIntType(), 
       SageBuilder::buildExprListExp(),
       bbnode
  );
  /* insert after */
  SageInterface::insertStatement(sgp,callStmt,false);
//  SageInterface::replaceStatement(sgp,callStmt);
}


/** #pragma kaapi waiton parsing
*/
void Parser::DoKaapiPragmaWaiton( SgPragmaDeclaration* sgp )
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
//  SgNode* nextnode = 
          sgp->get_parent()-> get_traversalSuccessorByIndex( 
              sgp->get_parent()->get_childIndex( sgp ) + 1);

  SgExprStatement* callStmt = SageBuilder::buildFunctionCallStmt
  (    "kaapi_sched_sync", 
       SageBuilder::buildIntType(), 
       SageBuilder::buildExprListExp(),
       bbnode
  );

  /* Insert after */
  SageInterface::insertStatement(sgp,callStmt,false);
//  SageInterface::replaceStatement(sgp,callStmt);

// TODO: parse list of variables 
}


/** #pragma kaapi parallel
    TODO: parse num_threads(num)
*/
void Parser::DoKaapiPragmaParallelRegion( SgPragmaDeclaration* sgp )
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
void Parser::DoKaapiPragmaInit( SgPragmaDeclaration* sgp, bool flag )
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
//  SgNode* nextnode = 
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
  /* Insert after */
  SageInterface::insertStatement(sgp,callStmt,false);
}


void Parser::DoKaapiPragmaDeclare( SgPragmaDeclaration* sgp )
{
  SgScopeStatement* scope = SageInterface::getScope(sgp);
  Sg_File_Info* fileInfo = sgp->get_file_info();
  
  const char* save_rpos = rpos;
  std::string name;
  ParseIdentifier(name);
  if (name == "reduction")
  {
    KaapiReduceOperator_t* redop = ParseReductionDeclaration(
        fileInfo, 
        scope 
    );
    if (redop ==0)
    {
      KaapiAbort("**** error");
    }
    kaapi_user_definedoperator.insert( std::make_pair( redop->name, redop  ) );

    std::cout << "Found declaration of reduction operator:"
              << redop->name
              << " freduce=" << redop->name_reducor
              << " finit=" << redop->name_redinit
              << std::endl;
    return;
  }
  else {
    std::cerr << "****[kaapi_c2c] #pragma kaapi declare '" << name << "' clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
    rpos = save_rpos;
    return;
  }

  return;
}


/** Return True if it is a valid expression with a function call to a task
    statement should be the enclosing statement of fc.
*/
bool isValidTaskWithRetValCallExpression(SgFunctionCallExp* fc, SgStatement* statement)
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
      if (isSgAssignOp(expr) == 0)
        break;

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
    case V_SgReturnStmt:        /* return expr , expr == sg->get_expression() */
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

/** This method traverse all the AST tree to find call to function task:
    - the call is registered into the _listcall of the object
*/
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

    
    /* */
    if (isSgFunctionCallExp(node))
    {
      SgFunctionCallExp* fc = isSgFunctionCallExp(node);
      SgStatement* exprstatement = SageInterface::getEnclosingStatement( fc );
      
      if (exprstatement->getAttribute("kaapinotask") !=0) 
        return;
      if (exprstatement->getAttribute("kaapiwrappercall") !=0) 
        return; /* call made by the wrapper do not replace it by task creation */
        
      if (fc !=0)
      {
        SgFunctionDeclaration* fdecl = fc->getAssociatedFunctionDeclaration();
        if (fdecl !=0)
        {
          KaapiTaskAttribute* kta = (KaapiTaskAttribute*)fdecl->getAttribute("kaapitask");
          if (kta !=0)
          {
            /* store both the container (basic block) and the funccall expr */
            SgScopeStatement* scope = SageInterface::getScope( fc );
            SgScopeStatement* loop = SageInterface::findEnclosingLoop( exprstatement );

            if ((kta->has_retval) && isValidTaskWithRetValCallExpression(fc, exprstatement ))
              _listcall.push_back( OneCall(scope, fc, exprstatement, kta, loop) );

if (loop !=0)
std::cout << "Find enclosing loop of a task declaration:" << loop->class_name() << std::endl
          << "At line: " << loop->get_file_info()->get_line()
          << std::endl;

#if 0 // TG no important here: see below in the main: when TemplateInstance are processed
{            SgTemplateInstantiationFunctionDecl* sg_tmpldecl = isSgTemplateInstantiationFunctionDecl( fdecl );
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
}
#endif
          } // decl !=0 & kta != 0
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
  std::list<OneCall>::iterator ifc;
  for (ifc = ktct->_listcall.begin(); ifc != ktct->_listcall.end(); ++ifc)
  {
    OneCall& oc = *ifc;

    /* add save restore frame for task creation inside loop 
       before buildFunCall2TaskSpawn, because it may declared __kaapi_thread
       before task spawn in the loop !
    */
    if (oc.forloop != 0)
      buildInsertSaveRestoreFrame( oc.forloop );

    /* replace function call by task spawn */
    buildFunCall2TaskSpawn( &oc );
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
      Sg_File_Info* fileInfo = node->get_file_info();
      SgPragmaDeclaration* sgp = isSgPragmaDeclaration(node);
      
      // Parse if it is a Kaapi Pragma
      std::string pragma_string = sgp->get_pragma()->get_pragma();
      Parser parser( pragma_string.c_str() );

      std::string name;
      parser.ParseIdentifier( name );
      
      /* css compatibility */
      if (name == "css") name = "kaapi";
      if (name != "kaapi") return;
      
      parser.ParseIdentifier( name );

      /* Parallel kaapi region
         add begin_parallel_region and end_parallel_region between the basic block
      */
      if (name == "parallel")
      {
        parser.DoKaapiPragmaParallelRegion( sgp );
      } 

      /* Case of task definition: because do no processing but register what to do
         then also pass the string in order the next step will be able to do parsing.
      */
      else if (name == "task")
      {
        parser.DoKaapiPragmaTask( sgp );
      } /* end for task */

      else if (name == "notask")
      {
        parser.DoKaapiPragmaNoTask( sgp );
      } 
       
      /* Case of barrier definition
      */
      else if ((name == "barrier") || (name == "sync")) // css compability
      {
        parser.DoKaapiPragmaBarrier( sgp );
      }

      /* Case of adaptive loop 
      */
      else if (name == "loop")
      {
        parser.DoKaapiPragmaLoop( sgp );
      } /* end for task */

      /* Case of waiton clause
      */
      else if (name == "waiton") 
      {
        parser.DoKaapiPragmaWaiton( sgp );
      }

      /* Case of variable scope
      */
      else if (name == "data") 
      {
        parser.DoKaapiPragmaData( node );
      }

      /* Case of init/finalize
      */
      else if ((name == "init") || (name == "start"))
      {
        parser.DoKaapiPragmaInit( sgp, true );
      }
      else if (name == "finish") 
      {
        parser.DoKaapiPragmaInit( sgp, false );
      }
      else if (name == "declare") 
      {
        parser.DoKaapiPragmaDeclare( sgp );
      }
      else if (name == "mutex") 
      {
        std::cerr << "****[kaapi_c2c] Warning: #pragma mutex is ignored.\n"
                  << "     In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << std::endl;
      }
      else {
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
  try {
    SgProject *project = frontend(argc, argv);
      
    KaapiPragma pragmaKaapi;
    
    
    /* Insert builting operator */
    kaapi_user_definedoperator.insert( 
      std::make_pair("+", 
        new KaapiReduceOperator_t("+", "+=", "0") 
      ) 
    );
    kaapi_user_definedoperator.insert( 
      std::make_pair("-", 
        new KaapiReduceOperator_t("-", "-=", "0") 
      ) 
    );
    kaapi_user_definedoperator.insert( 
      std::make_pair("*", 
        new KaapiReduceOperator_t("*", "*=", "1") 
      ) 
    );
    kaapi_user_definedoperator.insert( 
      std::make_pair("&", 
        new KaapiReduceOperator_t("&", "&=", "~0") 
      ) 
    );
    kaapi_user_definedoperator.insert( 
      std::make_pair("|", 
        new KaapiReduceOperator_t("|", "|=", "0") 
      ) 
    );
    kaapi_user_definedoperator.insert( 
      std::make_pair("^", 
        new KaapiReduceOperator_t("^", "^=", "0") 
      ) 
    );
    kaapi_user_definedoperator.insert( 
      std::make_pair("&&", 
        new KaapiReduceOperator_t("&&", "&&=", "1") 
      ) 
    );
    kaapi_user_definedoperator.insert( 
      std::make_pair("||", 
        new KaapiReduceOperator_t("||", "||=", "0") 
      ) 
    );
    
    
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
        kaapi_frame_ROSE_type = SageBuilder::buildOpaqueType ("kaapi_frame_t", gscope);
        kaapi_workqueue_ROSE_type = SageBuilder::buildOpaqueType ("kaapi_workqueue_t", gscope);
        kaapi_stealcontext_ROSE_type = SageBuilder::buildOpaqueType ("kaapi_stealcontext_t", gscope);
        
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
            (name_sync, SageBuilder::buildIntType(), SageBuilder::buildFunctionParameterList(),gscope);
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

        /* declare kaapi_thread_pushtask function */
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

        /* declare kaapi_thread_save_frame function */
        static SgName name_save_frame("kaapi_thread_save_frame");
        SgFunctionDeclaration *decl_save_frame = SageBuilder::buildNondefiningFunctionDeclaration(
            name_save_frame, 
            SageBuilder::buildIntType(), 
            SageBuilder::buildFunctionParameterList( 
                SageBuilder::buildFunctionParameterTypeList( 
                  SageBuilder::buildPointerType(kaapi_thread_ROSE_type),
                  SageBuilder::buildPointerType(kaapi_frame_ROSE_type)
                )
            ),
            gscope);
        ((decl_save_frame->get_declarationModifier()).get_storageModifier()).setExtern();

        /* declare kaapi_thread_restore_frame function */
        static SgName name_restore_frame("kaapi_thread_restore_frame");
        SgFunctionDeclaration *decl_restore_frame = SageBuilder::buildNondefiningFunctionDeclaration(
            name_restore_frame, 
            SageBuilder::buildIntType(), 
            SageBuilder::buildFunctionParameterList( 
                SageBuilder::buildFunctionParameterTypeList( 
                  SageBuilder::buildPointerType(kaapi_thread_ROSE_type),
                  SageBuilder::buildPointerType(kaapi_frame_ROSE_type)
                )
            ),
            gscope);
        ((decl_restore_frame->get_declarationModifier()).get_storageModifier()).setExtern();

        /* declare kaapi_thread_push_frame function */
        static SgName name_push_frame("kaapi_thread_push_frame");
        SgFunctionDeclaration *decl_push_frame = SageBuilder::buildNondefiningFunctionDeclaration(
            name_push_frame, 
            SageBuilder::buildPointerType(kaapi_thread_ROSE_type),
            SageBuilder::buildFunctionParameterList( 
                SageBuilder::buildFunctionParameterTypeList( )
            ),
            gscope);
        ((decl_push_frame->get_declarationModifier()).get_storageModifier()).setExtern();

        /* declare kaapi_thread_pop_frame function */
        static SgName name_pop_frame("kaapi_thread_pop_frame");
        SgFunctionDeclaration *decl_pop_frame = SageBuilder::buildNondefiningFunctionDeclaration(
            name_pop_frame, 
            SageBuilder::buildPointerType(kaapi_thread_ROSE_type),
            SageBuilder::buildFunctionParameterList( 
                SageBuilder::buildFunctionParameterTypeList( )
            ),
            gscope);
        ((decl_pop_frame->get_declarationModifier()).get_storageModifier()).setExtern();

        /* declare kaapi_workqueue_init function */
          static SgName name_wq_init("kaapi_workqueue_init");
          SgFunctionDeclaration *decl_wq_init = SageBuilder::buildNondefiningFunctionDeclaration(
              name_wq_init, 
              SageBuilder::buildVoidType(),
              SageBuilder::buildFunctionParameterList( 
                  SageBuilder::buildFunctionParameterTypeList( 
                    SageBuilder::buildPointerType(kaapi_workqueue_ROSE_type),
                    SageBuilder::buildLongType(),
                    SageBuilder::buildLongType()
                  )
              ),
              gscope);
          ((decl_wq_init->get_declarationModifier()).get_storageModifier()).setExtern();
        

        {/* declare kaapi_task_begin_adaptive function */
          static SgName name("kaapi_task_begin_adaptive");
          SgFunctionDeclaration *decl = SageBuilder::buildNondefiningFunctionDeclaration(
              name, 
              SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type),
              SageBuilder::buildFunctionParameterList( 
                  SageBuilder::buildFunctionParameterTypeList( 
                    SageBuilder::buildPointerType(kaapi_thread_ROSE_type),
                    SageBuilder::buildIntType(), /* flag */
                    SageBuilder::buildIntType(), /* splitter */
                    SageBuilder::buildPointerType(SageBuilder::buildVoidType())
                  )
              ),
              gscope);
          ((decl->get_declarationModifier()).get_storageModifier()).setExtern();
        }

        {/* declare kaapi_task_end_adaptive function */
          static SgName name("kaapi_task_end_adaptive");
          SgFunctionDeclaration *decl = SageBuilder::buildNondefiningFunctionDeclaration(
              name, 
              SageBuilder::buildVoidType(),
              SageBuilder::buildFunctionParameterList( 
                  SageBuilder::buildFunctionParameterTypeList( 
                    SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type)
                  )
              ),
              gscope);
          ((decl->get_declarationModifier()).get_storageModifier()).setExtern();
        }


        /* Process all #pragma */
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
          std::string name;
          std::string& pragma_string = func_decl_i->second;
          Parser parser(pragma_string.c_str());

  #if 0 // was suppressed with new parsing strategy
          /* re-parse 'kaapi' or 'css' and 'task' */
          Parser.parseIdentifier( name ); // == kaapi

          Parser.parseIdentifier( name ); // == task
  #endif        
          
          /* ok inputstream is correctly positionned */
          parser.DoKaapiTaskDeclaration( func_decl_i->first );
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
  } catch (...)
  {
    return -1;
  }
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
SgVariableDeclaration* buildInsertDeclarationKaapiThread( SgScopeStatement* scope )
{
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
    scope
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
/*    
    SgVariableDeclaration* const var_decl = buildThreadVariableDecl(oc->scope);
    buildVariableAssignment(oc->statement, var_decl, oc->scope);
*/
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

#if 1 // handle return value
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
      "kaapi_task_initdfg",
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
      SgStatement* node = isSgStatement(out[i]->get_from()->get_SgNode());
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
  
   On success, return a SgScopeStatement that correspond to rewrite of the loop execution.  
*/
SgScopeStatement* buildConvertLoop2Adaptative( SgScopeStatement* loop )
{
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

  SgInitializedName* ivar;
  SgExpression* begin_iter;
  SgExpression* end_iter;
  SgExpression* step;
  SgStatement*  body;
  bool hasIncrementalIterationSpace;
  bool isInclusiveUpperBound;
  
  bool retval = SageInterface::isCanonicalForLoop (forloop, 
        &ivar,
        &begin_iter,
        &end_iter,
        &step,
        &body,
        &hasIncrementalIterationSpace,
        &isInclusiveUpperBound
  );
  if (!retval)
  {
    /* cannot put it in canonical form */
    Sg_File_Info* fileInfo = forloop->get_file_info();
    std::cerr << "****[kaapi_c2c] wraning. Loop is not in canonical form."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
    
  /* Generate the header:
     kaapi_workqueue_t work;
     kaapi_workqueue_init( &work, 0, __kaapi_end );
     The interval [0,__kaapi_end] is the number of iteration in the loop.
     Each pop of range [i,j) will iterate to the original index [begin+i*incr
  */
  SgBasicBlock* bb = SageBuilder::buildBasicBlock( );
  bb->set_parent( loop->get_parent() );

  SgVariableSymbol* newkaapi_threadvar = 
    SageInterface::lookupVariableSymbolInParentScopes(
        "__kaapi_thread", 
        scope 
  );
  if (newkaapi_threadvar ==0)
  {
    buildInsertDeclarationKaapiThread( scope );
    newkaapi_threadvar = 
        SageInterface::lookupVariableSymbolInParentScopes(
            "__kaapi_thread", 
            scope 
      );
    if (newkaapi_threadvar ==0) 
      KaapiAbort("*** internal error");
  }

  SgVariableDeclaration* work = SageBuilder::buildVariableDeclaration (
      "__kaapi_work",
      kaapi_workqueue_ROSE_type,
      0, 
      bb
  );
  SageInterface::appendStatement( work, bb );

  SgVariableDeclaration* local_ivar = SageBuilder::buildVariableDeclaration (
      ivar->get_name(),
      ivar->get_type(),
      0,
      bb
  );
  /* do not include: already included in the scope */

  SgVariableDeclaration* local_ivar_end = SageBuilder::buildVariableDeclaration (
      ivar->get_name()+"_end",
      ivar->get_type(),
      0,
      bb
  );
  SageInterface::appendStatement( local_ivar_end, bb );
  
  SgVariableDeclaration* local_beg = SageBuilder::buildVariableDeclaration (
      "__kaapi_range_beg",
      SageBuilder::buildLongType(),
      0,
      bb
  );
  SageInterface::appendStatement( local_beg, bb );
  
  SgVariableDeclaration* local_end = SageBuilder::buildVariableDeclaration (
      "__kaapi_range_end",
      SageBuilder::buildLongType(),
      0,
      bb
  );
  SageInterface::appendStatement( local_end, bb );
  
  SgVariableDeclaration* wc = SageBuilder::buildVariableDeclaration (
      "__kaapi_wc",
      SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type),
      0, 
      bb
  );
  SageInterface::appendStatement( wc, bb );
  
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

  SgVariableDeclaration* wqsize = SageBuilder::buildVariableDeclaration (
      "__kaapi_wqsize",
      SageBuilder::buildLongType(),
      SageBuilder::buildAssignInitializer(
        exp_size
      ),
      bb
  );
  SageInterface::appendStatement( wqsize, bb );

  SgVariableDeclaration* popsize = SageBuilder::buildVariableDeclaration (
      "__kaapi_popsize",
      SageBuilder::buildLongType(),
      SageBuilder::buildAssignInitializer(
        SageBuilder::buildDivideOp(
          SageBuilder::buildVarRefExp(wqsize),
          SageBuilder::buildMultiplyOp(
            SageBuilder::buildIntVal(4),
            SageBuilder::buildFunctionCallExp(    
              "kaapi_getconcurrency",
              SageBuilder::buildIntType(), 
              SageBuilder::buildExprListExp(),
              bb
            )
          )
        )
      ),
      bb
  );
  SageInterface::appendStatement( popsize, bb );

  SgExprStatement* callinit_stmt = SageBuilder::buildFunctionCallStmt(    
      "kaapi_workqueue_init",
      SageBuilder::buildVoidType(), 
      SageBuilder::buildExprListExp(
        SageBuilder::buildAddressOfOp(
          SageBuilder::buildVarRefExp(work)
        ),
        SageBuilder::buildLongIntVal(0),
        SageBuilder::buildVarRefExp(wqsize)
      ),
      bb
  );
  SageInterface::appendStatement( callinit_stmt, bb );

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
          SageBuilder::buildIntVal(0),
          SageBuilder::buildAddressOfOp( SageBuilder::buildVarRefExp(work) )
        ),
        bb
      )
    )
  );
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
  SageInterface::appendStatement( init_ivar_local, bb );

  SgWhileStmt* while_popsizeloop = 
    SageBuilder::buildWhileStmt(
      /* !kaapi_workqueue_pop() */
      SageBuilder::buildNotOp(
        SageBuilder::buildFunctionCallExp(    
          "kaapi_workqueue_pop",
          SageBuilder::buildIntType(), 
          SageBuilder::buildExprListExp(
            SageBuilder::buildAddressOfOp(SageBuilder::buildVarRefExp(work)),
            SageBuilder::buildAddressOfOp(SageBuilder::buildVarRefExp(local_beg)),
            SageBuilder::buildAddressOfOp(SageBuilder::buildVarRefExp(local_end)),
            SageBuilder::buildVarRefExp( popsize )
          ),
          bb
        )
      ),
      SageBuilder::buildBasicBlock( )
  );
  
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
      body
    );
  SageInterface::appendStatement( new_forloop, isSgBasicBlock(while_popsizeloop->get_body( ))  );

  /* iter = iter_end at the end of the local loop */
  SgStatement* swap_ivars_local =
    SageBuilder::buildAssignStatement(
      SageBuilder::buildVarRefExp(local_ivar),
      SageBuilder::buildVarRefExp(local_ivar_end)
    );
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
  SageInterface::appendStatement( wc_term, bb );


  /* 
  */
  SageInterface::movePreprocessingInfo( forloop, bb );

  return bb;
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
    {
      std::cerr << "Parameter:" << sym->get_name().str() << " unknown in parameter list" << std::endl;
      KaapiAbort("*** Cannot find which parameter is involved in dimension expression");
    }
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



/*** Parser
*/
void Parser::ParseGetLine( std::string& ident )
{
  ident = std::string(rpos);
  rpos = rlast;
}


int Parser::ParseIdentifier( std::string& ident )
{
  char c;
  ident.clear();
  skip_ws();
  c = readchar();
  if ( !isletter(c) )
  {
    putback();
    return c;
  }

  ident.push_back(c);
  do {
    c = readchar();
    if (isletternum(c))
      ident.push_back(c);
    else if (isdigit(c))
      goto digits;
    else {
      putback();
      return c;
    }
  } while ( 1 );

digits:
  ident.push_back(c);
  do {
    c = readchar();
    if (isdigit(c))
      ident.push_back(c);
    else {
      putback();
      return c;
    }
  } while ( 1 );
}


SgUnsignedLongVal* Parser::ParseExprIntegralNumber( )
{
  unsigned long value = 0;
  char c;
  skip_ws();
  c = readchar();
  while (isdigit(c))
  {
    value *= 10;
    value += c - '0';
    c = readchar();
  }
  --rpos;
  return SageBuilder::buildUnsignedLongVal( value );
}

SgVarRefExp* Parser::ParseExprIdentifier( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  std::string ident;
  ParseIdentifier(ident);
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

SgExpression* Parser::ParseExprConstant( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  char c;
  c = readchar();
  putback();
  if (isdigit(c))
  {
    return ParseExprIntegralNumber( );
  }
  return ParseExprIdentifier(fileInfo,scope);
}

SgExpression* Parser::ParsePrimaryExpression( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  SgExpression* expr;
  char c;
  skip_ws();
  c = readchar();
  if (c == '(')
  {
    expr = ParseExpression(fileInfo, scope );
    skip_ws();
    c = readchar();
    if (c != ')') 
    {
      std::cerr << "****[kaapi_c2c] Error found '" << c 
                << "'. Missing ')' in primary expression."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    return expr;
  }
  putback();
  expr = ParseExprConstant(fileInfo, scope );
  return expr;
}


/**/
SgExpression* Parser::ParseUnaryExpression( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  SgExpression* expr;

/* currently do not handle sizeof expression 
   putback seems to be limited to 1 char.
*/
  const char* save_rpos = rpos;
  skip_ws();
  char sizeof_name[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int nitem = sscanf(rpos, "%7s", sizeof_name);
  if ((nitem !=0) && (strcmp(sizeof_name, "sizeof") == 0))
  {
    char c;
    skip_ws();
    c = readchar();
    if (c == '(')
    {
      std::string type_name;
      ParseIdentifier( type_name );
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

      skip_ws();
      c = readchar();
      if (c != ')') 
      {
        std::cerr << "****[kaapi_c2c] Error. Missing ')' in cast expression.\n"
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
      expr = SageBuilder::buildSizeOfOp(type);
      return expr;
    }
    else 
    {
      putback();
      expr = ParseUnaryExpression(fileInfo, scope );
      expr = SageBuilder::buildSizeOfOp(expr);
      return expr;
    }
  }
  else {
    /* putback what has been read */
    rpos = save_rpos;
  }
  
  expr = ParsePrimaryExpression( fileInfo, scope);

  return expr;
}


/**/
SgExpression* Parser::ParseCastExpression( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  SgExpression* expr;
  char c;
  skip_ws();
  c = readchar();
  if (c == '(') 
  {
    std::string type_name;
    ParseIdentifier( type_name );
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

    skip_ws();
    c = readchar();
    if (c != ')')
    {
      std::cerr << "****[kaapi_c2c] Error. Missing ')' in cast expression."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    expr = ParseCastExpression(fileInfo, scope);
    expr = SageBuilder::buildCastExp (expr, type );
    return expr;
  }
  putback();
  expr= ParseUnaryExpression( fileInfo, scope);
  return expr;
}


/**/
SgExpression* Parser::ParseMultiplicativeExpression( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  char c;
  SgExpression* expr= ParseCastExpression( fileInfo, scope);
redo:
  skip_ws();
  c = readchar();
  if (c == '*')
  {
    expr = SageBuilder::buildMultiplyOp( expr, ParseCastExpression( fileInfo, scope) );
    goto redo;
  }
  else if (c == '/')
  {
    expr = SageBuilder::buildDivideOp( expr, ParseCastExpression( fileInfo, scope) );
    goto redo;
  }
  else if (c == '%')
  {
    expr = SageBuilder::buildModOp( expr, ParseCastExpression( fileInfo, scope) );
    goto redo;
  }
  else 
    putback();
  return expr;
}


SgExpression* Parser::ParseAdditiveExpression( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  char c;
  SgExpression* expr= ParseMultiplicativeExpression( fileInfo, scope);
redo:
  skip_ws();
  c = readchar();
  if (c == '+')
  {
    expr = SageBuilder::buildAddOp( expr, ParseMultiplicativeExpression( fileInfo, scope) );
    goto redo;
  }
  else if (c == '-')
  {
    expr = SageBuilder::buildSubtractOp( expr, ParseMultiplicativeExpression( fileInfo, scope) );
    goto redo;
  }
  else 
    putback();
  return expr;
}


SgExpression* Parser::ParseExpression( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  SgExpression* expr;
  expr = ParseAdditiveExpression( fileInfo, scope);
  return expr;
}


KaapiStorage_t Parser::ParseStorage( 
    Sg_File_Info*       fileInfo, 
    SgScopeStatement*   scope 
)
{
  std::string name;
  const char* save_rpos = rpos;
  ParseIdentifier( name );
  for (size_t i=0; i<name.size(); ++i)
    name[i] = tolower(name[i]);

  if ((name == "c") || (name == "rowmajor")) return KAAPI_ROW_MAJOR;
  if ((name == "fortran") || (name == "columnmajor")) return KAAPI_COL_MAJOR;

  rpos = save_rpos;
  return KAAPI_BAD_STORAGE;
}


KaapiAccessMode_t Parser::ParseAccessMode( 
    Sg_File_Info* fileInfo
)
{
  std::string name;
  const char* save_rpos = rpos;
  ParseIdentifier( name );
  for (size_t i=0; i<name.size(); ++i)
    name[i] = tolower(name[i]);

  if ((name == "write") || (name == "w") || (name == "output")) return KAAPI_W_MODE;
  if ((name == "read") || (name == "r") || (name == "input")) return KAAPI_R_MODE;
  if ((name == "readwrite") || (name == "rw") || (name == "inout")) return KAAPI_RW_MODE;
  if ((name == "reduction") || (name == "cw")) return KAAPI_CW_MODE;
  if ((name == "value") || (name == "v")) return KAAPI_V_MODE;
  if ((name == "global") || (name == "g")) return KAAPI_GLOBAL_MODE;
  if (name == "highpriority") return KAAPI_HIGHPRIORITY_MODE;

  rpos = save_rpos;
  return KAAPI_VOID_MODE;
}


KaapiReduceOperator_t* Parser::ParseReduceOperator( 
    Sg_File_Info* fileInfo
)
{
  std::string name;
  const char* save_rpos = rpos;
  ParseIdentifier( name );
  
  if (name.empty()) 
  {
    /* try to read char -> basic operator +, - etc */
    char c;
    c = readchar();
    if ( 
         (c == '+') || (c == '-') || (c == '*')
      || (c == '^'))
    {
      name = c;
      c = readchar();
      if (c != ' ')
      {
        rpos= save_rpos;
        return 0;
      }
      putback();
    }
    else if ((c == '&') || (c == '|'))
    { /* may be && or || */
      char c1;
      c1 = readchar();
      if (c != c1)
      {
        putback();
        name = c;
      }
      else {
        if (c == '&') name = "&&";
        else name = "||";
      }
    }
    else {
      name = "ne cherchepas_ca_n'existe pas";
      rpos = save_rpos;
      return 0;
    }
  }
  std::map<std::string,KaapiReduceOperator_t*>::iterator curr =
    kaapi_user_definedoperator.find(name);
  if (curr == kaapi_user_definedoperator.end())
  {
    std::cerr << "****[kaapi_c2c] Error. Unknown reduce operator name '" << name << "'."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  
  return curr->second;
}



KaapiReduceOperator_t* Parser::ParseReductionDeclaration( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  std::string name;
  char c;
  const char* save_rpos = rpos;
  
  skip_ws();
  c = readchar();
  if (c != '(')
  {
    std::cerr << "****[kaapi_c2c] Error. Missing '(' in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  ParseIdentifier(name);
  if (name.empty())
  {
    std::cerr << "****[kaapi_c2c] Error. Invalid name in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  KaapiReduceOperator_t* newop = new KaapiReduceOperator_t;
  newop->name = strdup(name.c_str());
  
  skip_ws();
  c = readchar();
  if (c != ':')
  {
    delete newop;
    std::cerr << "****[kaapi_c2c] Error. Missing ':' in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  ParseIdentifier(name);
  if (name.empty())
  {
    std::cerr << "****[kaapi_c2c] Error. Invalid name in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  /* look up for name as a function 
     - not that due to overloading (C++) a function may appears multiple time.
     - the concrete type is only known during utilization of a variable reduction
     Thus we postpone the verification until the definition of variable in reduciton clause
  */
  newop->name_reducor = strdup(name.c_str());
  
  skip_ws();
  c = readchar();
  if (c != ')')
  {
    delete newop;
    std::cerr << "****[kaapi_c2c] Error. Missing ')' in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  skip_ws();
  c = readchar();
  if (c == EOF) 
   return newop;
  if (!isletter(c))
  {
    delete newop;
    std::cerr << "****[kaapi_c2c] Error. Waiting for 'identity'  in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  putback();
  ParseIdentifier(name);
  if (name != "identity")
  {
    delete newop;
    std::cerr << "****[kaapi_c2c] Error. Waiting for 'identity'  in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  
  skip_ws();
  c = readchar();
  if (c != '(')
  {
    std::cerr << "****[kaapi_c2c] Error. Missing '(' in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  
  ParseIdentifier(name);
  if (name.empty())
  {
    std::cerr << "****[kaapi_c2c] Error. Invalid name for identity function in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  newop->name_redinit = strdup(name.c_str());
  
  skip_ws();
  c = readchar();
  if (c != ')')
  {
    delete newop;
    std::cerr << "****[kaapi_c2c] Error. Missing ')' in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  return newop;
}


void Parser::ParseDimension( 
    Sg_File_Info* fileInfo, 
    KaapiParamAttribute* kpa, 
    SgScopeStatement* scope 
)
{
  char c;

  skip_ws();
  c = readchar();
  if (c != '[') 
  {
    std::cerr << "****[kaapi_c2c] Error. Missing '[' in dimension expression."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  kpa->ndim[kpa->dim] = ParseExpression( fileInfo, scope );
  if (kpa->ndim[kpa->dim] ==0)
  {
    std::cerr << "****[kaapi_c2c] Error. BAd expression in dimension."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  
  skip_ws();
  c = readchar();
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
}


void Parser::ParseNDimensions( 
    Sg_File_Info* fileInfo, 
    KaapiParamAttribute* kpa, 
    SgScopeStatement* scope 
)
{
  char c;

next_dimension:
  skip_ws();
  c = readchar();
  if (c == '[')
  {
    putback();
    ParseDimension( fileInfo, kpa, scope );
    goto next_dimension;
  }
  putback();
  return;
}


void Parser::ParseComplexView( 
    Sg_File_Info*        fileInfo, 
    KaapiParamAttribute* kpa, 
    SgScopeStatement*    scope 
)
{
  std::string name;
  char c;
  const char* save_rpos;
  
  skip_ws();
  c = readchar();
  if (c != '{')
  {
    putback();
    return;
  }
  
redo:
  skip_ws();
  save_rpos = rpos;
  ParseIdentifier( name );
  
  if ((name == "storage") || (name == "lda"))
  {
    skip_ws();
    c = readchar();
    if (c != '=') 
    {
      std::cerr << "****[kaapi_c2c] Error. Missing '=' after '" << name << "'."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    if (name =="storage")
    {
      if (kpa->storage != KAAPI_BAD_STORAGE) 
      {
        std::cerr << "****[kaapi_c2c] Error. Storage attribut already defined for same identifier."
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
      kpa->storage = ParseStorage( fileInfo, scope );
      if (kpa->storage != KAAPI_BAD_STORAGE) 
      {
        std::cerr << "****[kaapi_c2c] Error. Bad storage name: waiting 'C', 'RowMajor', 'Fortran' or 'ColumnMajor'."
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
    }
    else 
    { /* name == lda */
      if (kpa->lda != 0) 
      {
        std::cerr << "****[kaapi_c2c] Error. LDA attribut already defined for same identifier."
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
      kpa->lda = ParseExprIdentifier(fileInfo,scope);
      if (kpa->lda == 0) 
      {
        std::cerr << "****[kaapi_c2c] Error. LDA attribut, waiting identifier."
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
    }
    
    /* here: c==';' or '}' */
    skip_ws();
    c = readchar();
    if (c == ';') goto redo; /* next attribut */
    if (c == '}') return;
    std::cerr << "****[kaapi_c2c] Error. Missing ';' or '}' in complex view definition. Found: '" << c << "'."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  else {
    rpos = save_rpos; /* putback token */
  }

  ParseNDimensions( fileInfo, kpa, scope );
  skip_ws();
  c = readchar();
  if (c == ';') goto redo;
  if (c == '}') return;
  
  std::cerr << "****[kaapi_c2c] Error. Missing ';' or '}' in complex view definition. Found: '" << c << "'."
            << "     In filename '" << fileInfo->get_filename() 
            << "' LINE: " << fileInfo->get_line()
            << std::endl;
  KaapiAbort("**** error");
}


void Parser::ParseRangeDeclaration( 
    Sg_File_Info*        fileInfo, 
    KaapiParamAttribute* kpa, 
    SgScopeStatement*    scope 
)
{
  std::string name;
  char c;

  skip_ws();
  c = readchar();
  if (c == '[')
  {
    SgVarRefExp* expr_first_bound;
    SgVarRefExp* expr_second_bound;

    /* range 1D definition */
    expr_first_bound = ParseExprIdentifier( fileInfo, scope );
    /* parse .. */
    skip_ws();
    c= readchar();
    if (c != ':')
    {
      std::cerr << "****[kaapi_c2c] Error. Missing ':' for range declaration. Found'" << c << "'."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    expr_second_bound = ParseExprIdentifier( fileInfo, scope );

    skip_ws();
    c= readchar();
    /* parse ')' or ']' */
    if ( (c != ')') && ( c != ']'))
    {
      std::cerr << "****[kaapi_c2c] Error. Missing ')' or ']' for end of range declaration. Found'" << c << "'."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    
    if (c == ')') /* open range declaration */
      kpa->type = KAAPI_OPEN_RANGE_TYPE;
    if (c == ']') /* closed range declaration */
      kpa->type = KAAPI_CLOSE_RANGE_TYPE;
    kpa->expr_firstbound  = expr_first_bound;
    kpa->first_bound = expr_first_bound->get_symbol()->get_name().str();
    kpa->expr_secondbound = expr_second_bound;
    kpa->second_bound = expr_second_bound->get_symbol()->get_name().str();
  }
  else {
    putback();
    ParseIdentifier( name );
    if (name == "")
    {
      std::cerr << "****[kaapi_c2c] Error. Missing identifier for range declaration."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    kpa->type = KAAPI_ARRAY_NDIM_TYPE;
    kpa->name = strdup(name.c_str());
    skip_ws();
    c = readchar();
    if (c == '{')
    {
      putback();
      ParseComplexView( fileInfo, kpa, scope );
      return;
    }
    else if (c == '[')
    {
      putback();
      ParseNDimensions( fileInfo, kpa, scope );
      return;
    }
    else if ((c == ')') || (c == ','))
    {
      kpa->dim = 0; /* means dim=1, but single element */
      putback();
      return;
    }
    std::cerr << "****[kaapi_c2c] Error. Missing end ')' or ','."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
}



/* Parse the list of formal parameter definitions      
*/
void Parser::ParseListParamDecl( 
    Sg_File_Info*       fileInfo, 
    KaapiTaskAttribute* kta,
    KaapiAccessMode_t   mode,
    SgScopeStatement*   scope 
)
{
  char c;
  KaapiParamAttribute* kpa;
  KaapiReduceOperator_t* redop = 0;

  if (mode == KAAPI_CW_MODE)
  {
    /* parse operator */
    redop = ParseReduceOperator(fileInfo );
    if (redop ==0)
    {
      std::cerr << "****[kaapi_c2c] Error. Unknown reduction operator."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    
    /* parse : */
    skip_ws();
    c = readchar();
    if (c != ':')
    {
      std::cerr << "****[kaapi_c2c] Error. Missing ':' in reduction clause"
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
  }
    
redo:
  kpa = new KaapiParamAttribute;
  ParseRangeDeclaration( fileInfo, kpa, scope );
  switch (kpa->type)
  {
    case KAAPI_ARRAY_NDIM_TYPE:
    {
      if (mode != KAAPI_GLOBAL_MODE)
      {
        std::map<std::string,int>::iterator curr = kta->lookup.find( kpa->name );
        if (curr == kta->lookup.end())
        {
          std::cerr << "****[kaapi_c2c] Error. Unkown formal parameter identifier in declaration '" << kpa->name << "'."
                    << "     In filename '" << fileInfo->get_filename() 
                    << "' LINE: " << fileInfo->get_line()
                    << std::endl;
          KaapiAbort("**** error");
        }
        int ith = curr->second;
        kta->formal_param[ith].mode = mode;
        kta->formal_param[ith].attr = kpa;
        kta->formal_param[ith].redop = redop;
        kta->israngedecl[ith]       = 0;
      }
      else { /* global variable */
        SgVariableSymbol* var = SageInterface::lookupVariableSymbolInParentScopes( kpa->name, scope );
        if (var ==0)
        {
          std::cerr << "****[kaapi_c2c] Error. Unkown global variable identifier '" << kpa->name << "'."
                    << "     In filename '" << fileInfo->get_filename() 
                    << "' LINE: " << fileInfo->get_line()
                    << std::endl;
          KaapiAbort("**** error");
        }
        KaapiTaskFormalParam kpa_extra;
        kpa_extra.mode     = mode;
        kpa_extra.initname = var->get_declaration();
        kpa_extra.type     = var->get_type();
        kta->extra_param.push_back( kpa_extra );
      }

    } break;
    
    case KAAPI_OPEN_RANGE_TYPE:
    case KAAPI_CLOSE_RANGE_TYPE:
    {
      std::map<std::string,int>::iterator curr1 = kta->lookup.find( kpa->first_bound );
      if (curr1 == kta->lookup.end())
      {
        std::cerr << "****[kaapi_c2c] Error. Unkown formal parameter identifier in range declaration '" << kpa->first_bound << "'."
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
      std::map<std::string,int>::iterator curr2 = kta->lookup.find( kpa->second_bound );
      if (curr2 == kta->lookup.end())
      {
        std::cerr << "****[kaapi_c2c] Error. Unkown formal parameter identifier in range declaration '" << kpa->second_bound << "'."
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
      kpa->index_firstbound  = curr1->second;
      kpa->index_secondbound = curr2->second;
      kta->formal_param[curr1->second].mode  = mode;
      kta->formal_param[curr1->second].attr  = kpa;
      kta->israngedecl[curr1->second]        = 1;
      kta->formal_param[curr1->second].redop = redop;

      kta->formal_param[curr2->second].mode  = mode;
      kta->formal_param[curr2->second].attr  = kpa;
      kta->israngedecl[curr2->second]        = 2;
      kta->formal_param[curr2->second].redop = redop;
    } break;

    
    case KAAPI_BAD_PARAM_TYPE:
    default:
      std::cerr << "****[kaapi_c2c] Error. Bad list of declaration."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
  }
  
  skip_ws();
  c = readchar();
  if (c == ',') goto redo;
  if (c == ')') 
  {
    putback();
    return;
  }

  std::cerr << "****[kaapi_c2c] Error. Missing ',' or ')' at the end of list of declaration."
            << "     In filename '" << fileInfo->get_filename() 
            << "' LINE: " << fileInfo->get_line()
            << std::endl;
  KaapiAbort("**** error");
}


void Parser::ParseListParam( 
    Sg_File_Info*       fileInfo, 
    KaapiTaskAttribute* kta,
    SgScopeStatement*   scope 
)
{
  char c;
  KaapiAccessMode_t mode;

redo:
  mode = ParseAccessMode( fileInfo );
  
  if (mode == KAAPI_VOID_MODE)
  {
    std::cerr << "****[kaapi_c2c] Error. Bad access mode."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  
  skip_ws();
  c = readchar();
  if (c == EOF) return;
  
  if (mode == KAAPI_HIGHPRIORITY_MODE) 
  { /* ignore the mode */
    goto redo;
  }
  
  if (c != '(')
  {
    std::cerr << "****[kaapi_c2c] Error. Missing '(' in access mode list."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }

  ParseListParamDecl( fileInfo, kta, mode, scope );
  
  skip_ws();
  c = readchar();
  if (c != ')')
  {
    std::cerr << "****[kaapi_c2c] Error. Missing ')' at the end of access mode list."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  
  c = readchar();
  if (c == EOF) return;
  goto redo;
}

