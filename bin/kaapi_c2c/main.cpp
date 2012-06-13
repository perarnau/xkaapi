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


#include <string>
#include <iostream>
#include <algorithm>
#include <map>
#include <list>
#include <set>
#include <sstream>
#include <iomanip>
#include <ctype.h>
#include <time.h>
#include <stdexcept>
#include "rose_headers.h"
#include "globals.h"
#include "parser.h"
#include "kaapi_c2c_task.h"
#include "kaapi_pragma.h"
#include "kaapi_abort.h"
#include "kaapi_ssa.h"
#include "kaapi_taskcall.h"
#include "kaapi_initializer.h"
#include "kaapi_mode_analysis.h"
#include "kaapi_finalization.h"


int main(int argc, char **argv) 
{
  try {
    // debugging
    // SgProject::set_verbose(10);

    SgProject* project = new SgProject();
    // project->set_template_instantiation_mode(SgProject::e_none);
    project->parse(argc, argv);

    KaapiPragma pragmaKaapi;
    
    /* Insert builtin reduction operator */
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
        kaapi_workqueue_index_ROSE_type = SageBuilder::buildOpaqueType ("kaapi_workqueue_index_t", gscope);
        kaapi_stealcontext_ROSE_type = SageBuilder::buildOpaqueType ("kaapi_stealcontext_t", gscope);
        kaapi_request_ROSE_type = SageBuilder::buildOpaqueType ("kaapi_request_t", gscope);
        kaapi_taskadaptive_result_ROSE_type = SageBuilder::buildOpaqueType ("kaapi_taskadaptive_result_t", gscope);
        kaapi_splitter_context_ROSE_type = SageBuilder::buildOpaqueType ("kaapi_splitter_context_t", gscope);
        kaapi_task_body_ROSE_type = SageBuilder::buildFunctionType(
            SageBuilder::buildVoidType(), 
            SageBuilder::buildFunctionParameterTypeList( 
              SageBuilder::buildPointerType(SageBuilder::buildVoidType()),
              SageBuilder::buildPointerType(kaapi_thread_ROSE_type)
            )
        );

#if 0
	// add stddef.h
        SageInterface::addTextForUnparser(file->get_globalScope(),
                    "#include <stddef.h>\n",
                    AstUnparseAttribute::e_before
        );
#endif
        
#if 0 // do not work properly
        SageInterface::insertHeader ("kaapi.h", PreprocessingInfo::after, false, gscope);
#else
        /** Add #include <kaapi.h> to each input file
        */
        SageInterface::addTextForUnparser(file->get_globalScope(),
                    "#include <kaapic.h>\n",
                    AstUnparseAttribute::e_before
        );
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
                SageBuilder::buildIntType(),
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
                  SageBuilder::buildVoidType(), 
                  SageBuilder::buildFunctionParameterList(
                    SageBuilder::buildFunctionParameterTypeList(
                      SageBuilder::buildIntType()
                    )
                  ),
                  gscope
        );
        ((decl_kaapi_beginparallel->get_declarationModifier()).get_storageModifier()).setExtern();

        /* declare kaapi_end_parallel */
        static SgName name_endparallel("kaapi_end_parallel");
        SgFunctionDeclaration *decl_kaapi_endparallel 
          = SageBuilder::buildNondefiningFunctionDeclaration(
                  name_endparallel, 
                  SageBuilder::buildVoidType(), 
                  SageBuilder::buildFunctionParameterList(
                    SageBuilder::buildFunctionParameterTypeList(
                      SageBuilder::buildIntType()
                    )
                  ),
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

        /* declare kaapi_task_init function */
        static SgName name_task_init("kaapi_task_init");
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
                    SageBuilder::buildPointerType(kaapi_thread_ROSE_type),
                    SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type)
                  )
              ),
              gscope);
          ((decl->get_declarationModifier()).get_storageModifier()).setExtern();
        }

        {/* declare kaapi_task_begin_adaptive function */
          static SgName name("kaapi_reply_init_adaptive_task");
          SgFunctionDeclaration *decl = SageBuilder::buildNondefiningFunctionDeclaration(
              name, 
              SageBuilder::buildPointerType(SageBuilder::buildVoidType()),
              SageBuilder::buildFunctionParameterList( 
                  SageBuilder::buildFunctionParameterTypeList( 
                    SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type),
                    SageBuilder::buildPointerType(kaapi_request_ROSE_type),
                    SageBuilder::buildPointerType(kaapi_task_body_ROSE_type),
                    SageBuilder::buildIntType(), /* size */
                    SageBuilder::buildPointerType(kaapi_taskadaptive_result_ROSE_type)
                  )
              ),
              gscope);
          ((decl->get_declarationModifier()).get_storageModifier()).setExtern();
        }

        {/* declare kaapi_reply_push_adaptive_task function */
          static SgName name("kaapi_reply_push_adaptive_task");
          SgFunctionDeclaration *decl = SageBuilder::buildNondefiningFunctionDeclaration(
              name, 
              SageBuilder::buildVoidType(),
              SageBuilder::buildFunctionParameterList( 
                  SageBuilder::buildFunctionParameterTypeList( 
                    SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type),
                    SageBuilder::buildPointerType(kaapi_request_ROSE_type)
                  )
              ),
              gscope);
          ((decl->get_declarationModifier()).get_storageModifier()).setStatic();
        }

        {/* declare kaapi_splitter_default function */
          static SgName name("kaapi_splitter_default");
          SgFunctionDeclaration *decl = SageBuilder::buildNondefiningFunctionDeclaration(
              name, 
              SageBuilder::buildIntType(),
              SageBuilder::buildFunctionParameterList( 
                  SageBuilder::buildFunctionParameterTypeList( 
                    SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type),
                    SageBuilder::buildIntType(),
                    SageBuilder::buildPointerType(kaapi_request_ROSE_type),
                    SageBuilder::buildPointerType(SageBuilder::buildVoidType())
                  )
              ),
              gscope);
          ((decl->get_declarationModifier()).get_storageModifier()).setExtern();
        }

        {/* declare kaapi_thread_pushdata_align function */
          static SgName name("kaapi_thread_pushdata_align");
          SgFunctionDeclaration *decl = SageBuilder::buildNondefiningFunctionDeclaration(
              name, 
              SageBuilder::buildPointerType(SageBuilder::buildVoidType()),
              SageBuilder::buildFunctionParameterList( 
                  SageBuilder::buildFunctionParameterTypeList( 
                    SageBuilder::buildPointerType(kaapi_thread_ROSE_type),
                    SageBuilder::buildUnsignedIntType(),
                    SageBuilder::buildUnsignedLongType()
		)
              ),
              gscope);
          ((decl->get_declarationModifier()).get_storageModifier()).setExtern();
        }

        {/* declare kaapi_adaptive_result_data function */
          static SgName name("kaapi_adaptive_result_data");
          SgFunctionDeclaration *decl = SageBuilder::buildNondefiningFunctionDeclaration(
              name, 
              SageBuilder::buildPointerType(SageBuilder::buildVoidType()),
              SageBuilder::buildFunctionParameterList( 
                  SageBuilder::buildFunctionParameterTypeList( 
                    SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type)
		)
              ),
              gscope);
          ((decl->get_declarationModifier()).get_storageModifier()).setExtern();
        }

        {/* declare kaapi_get_thief_head function */
          static SgName name("kaapi_get_thief_head");
          SgFunctionDeclaration *decl = SageBuilder::buildNondefiningFunctionDeclaration(
              name, 
              SageBuilder::buildPointerType(SageBuilder::buildPointerType(kaapi_taskadaptive_result_ROSE_type)),
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

          /* ok inputstream is correctly positionned */
          parser.DoKaapiTaskDeclaration( func_decl_i->first );
        }

        // do mode analysis
        DoKaapiModeAnalysis(project);

        // turn call into ssa form
        KaapiSSATraversal ssa_traversal;
        ssa_traversal.traverse(project,preorder);

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

        // finalization pass
        DoKaapiFinalization(project);
      }

      /* Generate the format */
      time_t t;
      time(&t);
      std::ostringstream fout; //("kaapi-format.cpp");
//      fout << "#include \"kaapi.h\"" << std::endl;
      fout << "/* Format part for the predefined task */\n"
           << "/***** Date: " << ctime(&t) << "*****/\n" << std::endl;

      DoKaapiGenerateInitializer(fout);
      
      std::list<KaapiTaskAttribute*>::iterator begin_task;
      for (begin_task = all_tasks.begin(); begin_task != all_tasks.end(); ++begin_task)
      {
        DoKaapiGenerateFormat( fout, *begin_task );
      }

      SageInterface::addTextForUnparser(file->get_globalScope(),
                  fout.str(),
                  AstUnparseAttribute::e_after
      );
    }

    project->unparse();
  } catch (...)
  {
    KaapiAbort("****[kaapi_c2c] catch unknown exception");
    return -1;
  }
  return 0;
}
