/*
** kaapi_init.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:03 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@inrialpes.fr
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
#include "kaapi_impl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <errno.h>


/*
*/
kaapi_rtparam_t default_param;

/** Predefined format
*/
kaapi_format_t kaapi_shared_format;

kaapi_format_t kaapi_char_format;
kaapi_format_t kaapi_short_format;
kaapi_format_t kaapi_int_format;
kaapi_format_t kaapi_long_format;
kaapi_format_t kaapi_longlong_format;

kaapi_format_t kaapi_uchar_format;
kaapi_format_t kaapi_ushort_format;
kaapi_format_t kaapi_uint_format;
kaapi_format_t kaapi_ulong_format;
kaapi_format_t kaapi_ulonglong_format;

kaapi_format_t kaapi_float_format;
kaapi_format_t kaapi_double_format;


kaapi_task_body_t kaapi_bodies[1+KAAPI_TASK_BODY_MAX];


/** cpuset related routines
 -- grammar
    cpu_set   : cpu_expr ( ',' cpu_expr ) *
    cpu_expr  : '!'* cpu_list
    cpu_list  : cpu_index ':' cpu_index |
                ':' cpu_index |
                ':'
    cpu_index : num
    num       : [ '0' '9' ]+
 */

struct cpuset_parser
{
#define CPU_BIT_IS_DISABLED (1 << 0)
#define CPU_BIT_IS_USABLE (1 << 1)
  unsigned char cpu_bits[KAAPI_MAX_PROCESSOR];

  unsigned int* kid_map;

  unsigned int used_ncpus;
  unsigned int sys_ncpus;
  unsigned int total_ncpus;

  const char* str_pos;
  int err_no;
};


typedef struct cpuset_parser cpuset_parser_t;


static void init_parser
(
 cpuset_parser_t* parser,
 unsigned int* kid_map,
 unsigned int sys_ncpus,
 unsigned int total_ncpus,
 const char* str_pos
)
{
  memset(parser, 0, sizeof(cpuset_parser_t));

  parser->kid_map = kid_map;

  parser->total_ncpus = total_ncpus;
  parser->sys_ncpus = sys_ncpus;

  parser->str_pos = str_pos;
}


static inline void set_parser_error
(
 cpuset_parser_t* parser,
 int err_no
)
{
  parser->err_no = err_no;
}


static inline int parse_cpu_index
(
 cpuset_parser_t* parser,
 unsigned int* index
)
{
  char* end_pos;

  *index = (unsigned int)strtoul(parser->str_pos, &end_pos, 10);

  if (end_pos == parser->str_pos)
    {
      set_parser_error(parser, EINVAL);
      return -1;
    }

  if (*index >= parser->sys_ncpus)
    {
      set_parser_error(parser, E2BIG);
      return -1;
    }

  parser->str_pos = end_pos;

  return 0;
}


static inline int is_digit(char c)
{
  return (c >= '0') && (c <= '9');
}


static inline int is_list_delim(char c)
{
  return c == ':';
}


static inline int is_eol(char c)
{
  /* end of list */
  return (c == ',') || (c == 0);
}


static int parse_cpu_list
(
 cpuset_parser_t* parser,
 unsigned int* index_low,
 unsigned int* index_high
)
{
  /* 1 token look ahead */
  if (is_digit(*parser->str_pos))
    {
      if (parse_cpu_index(parser, index_low))
	return -1;
    }
  else
    {
      *index_low = 0;
    }

  *index_high = *index_low;

  if (is_list_delim(*parser->str_pos))
    {
      ++parser->str_pos;

      /* 1 token look ahead */
      if (is_eol(*parser->str_pos))
	{
	  *index_high = parser->total_ncpus - 1;
	}
      else if (parse_cpu_index(parser, index_high))
	{
	  return -1;
	}
    }

  /* swap indices if needed */
  if (*index_low > *index_high)
    {
      const unsigned int tmp_index = *index_high;
      *index_high = *index_low;
      *index_low = tmp_index;
    }

  return 0;
}


static int parse_cpu_expr(cpuset_parser_t* parser)
{
  unsigned int is_enabled = 1;
  unsigned int index_low;
  unsigned int index_high;

  for (; *parser->str_pos == '!'; ++parser->str_pos)
    is_enabled ^= 1;

  if (parse_cpu_list(parser, &index_low, &index_high))
    return -1;

  for (; index_low <= index_high; ++index_low)
    {
      if (is_enabled == 0)
	parser->cpu_bits[index_low] |= CPU_BIT_IS_DISABLED;

      if (parser->cpu_bits[index_low] & CPU_BIT_IS_DISABLED)
	{
	  /* this cpu is disabled. if it has been previously
	     mark usable, discard it and decrement the cpu cuont
	  */

	  if (parser->cpu_bits[index_low] & CPU_BIT_IS_USABLE)
	    {
	      parser->cpu_bits[index_low] &= ~CPU_BIT_IS_USABLE;
	      --parser->used_ncpus;
	    }
	}
      else if (!(parser->cpu_bits[index_low] & CPU_BIT_IS_USABLE))
	{
	  /* cpu not previously usable */

	  parser->cpu_bits[index_low] |= CPU_BIT_IS_USABLE;

	  if (++parser->used_ncpus == parser->total_ncpus)
	    break;
	}
    }

  return 0;
}


static int parse_cpu_set(cpuset_parser_t* parser)
{
  unsigned int i;
  unsigned int j;

  /* build the cpu set from string */

  while (1)
    {
      if (parse_cpu_expr(parser))
	return -1;

      if (*parser->str_pos != ',')
	{
	  if (!*parser->str_pos)
	    break;

	  set_parser_error(parser, EINVAL);

	  return -1;
	}

      ++parser->str_pos;
    }

  /* bind kid to available cpus */

  for (j = 0, i = 0; i < parser->used_ncpus; ++i, ++j)
    {
      for (; !(parser->cpu_bits[j] & CPU_BIT_IS_USABLE); ++j)
	;

      parser->kid_map[i] = j;
    }

  return 0;
}


#if 0
static void __attribute__((unused))
print_parser(const cpuset_parser_t* parser)
{
  unsigned int i;

  for (i = 0; i < parser->used_ncpus; ++i)
    printf("%u ~ %u\n", i, parser->kid_map[i]);
}
#endif


static void fill_identity_kid_map
    (
     unsigned int* kid_map,
     unsigned int ncpus
    )
{
  unsigned int icpu;

  for (icpu = 0; icpu < ncpus; ++icpu)
    kid_map[icpu] = icpu;
}


static int str_to_kid_map
    (
     unsigned int* kid_map,
     const char* cpuset_str,
     unsigned int sys_ncpus,
     unsigned int* total_ncpus
    )
{
  cpuset_parser_t parser;

  if (cpuset_str == NULL)
    return 0;

  init_parser(&parser, kid_map, sys_ncpus,
	      *total_ncpus, cpuset_str);

  if (parse_cpu_set(&parser))
    return parser.err_no;

  if (parser.used_ncpus < *total_ncpus)
    *total_ncpus = parser.used_ncpus;

  return 0;
}


/** \ingroup WS
    Initialize from xkaapi runtime parameters from command line
    \param argc [IN] command line argument count
    \param argv [IN] command line argument vector
    \retval 0 in case of success 
    \retval EINVAL because of error when parsing then KAAPI_CPUSET string
    \retval E2BIG because of a cpu index too high in KAAPI_CPUSET
    
*/
int kaapi_setup_param( int argc, char** argv )
{
  /* compute the number of cpu of the system */
#if defined(KAAPI_USE_LINUX)
  default_param.syscpucount = sysconf(_SC_NPROCESSORS_CONF);
#elif defined(KAAPI_USE_APPLE)
  {
    int mib[2];
    size_t len;
    mib[0] = CTL_HW;
    mib[1] = HW_NCPU;
    len = sizeof(default_param.syscpucount);
    sysctl(mib, 2, &default_param.syscpucount, &len, 0, 0);
  }
#else
  #warning "Could not compute number of physical cpu of the system. Default value==1"
  default_param.syscpucount = 1;
#endif
  /* adjust system limit, if library is compiled with greather number of processors that available */
  if (default_param.syscpucount > KAAPI_MAX_PROCESSOR)
    default_param.syscpucount = KAAPI_MAX_PROCESSOR;

  default_param.use_affinity = 0;
    
  /* Set default values */
  default_param.cpucount  = default_param.syscpucount;
  default_param.stacksize = 8*4096;
  
  /* Get values from environment variable */
  if (getenv("KAAPI_DISPLAY_PERF") !=0)
  {
    default_param.display_perfcounter = 1;
  }
  else
  {
    default_param.display_perfcounter = 0;
  }

  /* Get values from environment variable */
  if (getenv("KAAPI_STACKSIZE") !=0)
  {
    default_param.stacksize = atoi(getenv("KAAPI_STACKSIZE"));
  }
  if (getenv("KAAPI_CPUCOUNT") !=0)
  {
    default_param.cpucount = atoi(getenv("KAAPI_CPUCOUNT"));
    if (default_param.cpucount > KAAPI_MAX_PROCESSOR/* default_param.syscpucount*/)
      default_param.cpucount = default_param.syscpucount;
  }

  if (getenv("KAAPI_CPUSET") != 0)
  {
    int err_no =
      str_to_kid_map(default_param.kid_to_cpu,
		     getenv("KAAPI_CPUSET"),
		     default_param.syscpucount,
		     &default_param.cpucount);

    if (err_no)
      return err_no;

    default_param.use_affinity = 1;
  }
  else
  {
    fill_identity_kid_map(default_param.kid_to_cpu,
			  default_param.cpucount);
  }

  /* workstealing selection function */
  {
    default_param.wsselect = &kaapi_sched_select_victim_rand;
    const char* const wsselect = getenv("KAAPI_WSSELECT");
    if ((wsselect != NULL) && !strcmp(wsselect, "workload"))
      default_param.wsselect = &kaapi_sched_select_victim_workload_rand;
  }

#if defined(KAAPI_VERY_COMPACT_TASK)
  /* init default task body */
  kaapi_bodies[kaapi_nop_body]          = _kaapi_nop_body;
  kaapi_bodies[kaapi_taskstartup_body]  = _kaapi_taskstartup_body;
  kaapi_bodies[kaapi_retn_body]         = _kaapi_retn_body;
  kaapi_bodies[kaapi_suspend_body]      = _kaapi_suspend_body;
  kaapi_bodies[kaapi_tasksig_body]      = _kaapi_tasksig_body;
  kaapi_bodies[kaapi_taskfinalize_body] = _kaapi_taskfinalize_body;
  kaapi_bodies[kaapi_tasksteal_body]    = _kaapi_tasksteal_body;
  kaapi_bodies[kaapi_taskwrite_body]    = _kaapi_taskwrite_body;
  kaapi_bodies[kaapi_aftersteal_body]   = _kaapi_aftersteal_body;
#endif
  
  return 0;
}



/**
*/
#define KAAPI_REGISTER_BASICTYPEFORMAT( formatobject, type, fmt ) \
  static void formatobject##_cstor(void* dest)  { *(type*)dest = 0; }\
  static void formatobject##_dstor(void* dest) { *(type*)dest = 0; }\
  static void formatobject##_cstorcopy( void* dest, const void* src) { *(type*)dest = *(type*)src; } \
  static void formatobject##_copy( void* dest, const void* src) { *(type*)dest = *(type*)src; } \
  static void formatobject##_assign( void* dest, const void* src) { *(type*)dest = *(type*)src; } \
  static void formatobject##_print( FILE* file, const void* src) { fprintf(file, fmt, *(type*)src); } \
  static inline kaapi_format_t* fnc_##formatobject(void) \
  {\
    return &formatobject;\
  }\
  static inline void __attribute__ ((constructor)) __kaapi_register_format_##formatobject (void)\
  { \
    static int isinit = 0;\
    if (isinit) return;\
    isinit = 1;\
    kaapi_format_structregister( &fnc_##formatobject, \
                                 #type, sizeof(type), \
                                 &formatobject##_cstor, &formatobject##_dstor, &formatobject##_cstorcopy, \
                                 &formatobject##_copy, &formatobject##_assign, &formatobject##_print ); \
  }


KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_char_format, char, "%hhi")
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_short_format, short, "%hi")
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_int_format, int, "%i")
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_long_format, long, "%li")
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_longlong_format, long long, "%lli")


KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_uchar_format, unsigned char, "%hhu")
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_ushort_format, unsigned short, "%hu")
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_uint_format, unsigned int, "%u")
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_ulong_format, unsigned long, "%lu")
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_ulonglong_format, unsigned long long, "%llu")

KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_float_format, float, "%e")
KAAPI_REGISTER_BASICTYPEFORMAT(kaapi_double_format, double, "%e")


