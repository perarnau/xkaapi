/*
 ** xkaapi
 ** 
 ** Copyright 2009,2010,2011,2012 INRIA.
 **
 ** Contributors :
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
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
#include "kaapic_impl.h"

int kaapic_foreach_attr_init(kaapic_foreach_attr_t* attr)
{
  attr->rep.li.s_grain  = 1;
  attr->rep.li.p_grain  = 1;
  attr->datadist.type   = KAAPIC_DATADIST_VOID; /* means no distribution */
  attr->nthreads = -1;
  attr->policy   = 0;
  kaapi_cpuset_full(&attr->threadset);
  return 0;
}

int kaapic_foreach_attr_set_grains(
  kaapic_foreach_attr_t* attr, 
  long s_grain,
  long p_grain
)
{
  attr->rep.li.s_grain = s_grain;
  attr->rep.li.p_grain = p_grain;
  return 0;
}

int kaapic_foreach_attr_set_grains_ull(
  kaapic_foreach_attr_t* attr, 
  unsigned long long s_grain,
  unsigned long long p_grain
)
{
  attr->rep.ull.s_grain = s_grain;
  attr->rep.ull.p_grain = p_grain;
  return 0;
}

int kaapic_foreach_attr_set_threads(
  kaapic_foreach_attr_t* attr, 
  unsigned int nthreads
)
{
  attr->nthreads = nthreads;
  return 0;
}

int kaapic_foreach_attr_set_bloccyclic_datadistribution(
  kaapic_foreach_attr_t* attr, 
  unsigned long long blocsize,
  unsigned int cyclelength
)
{
  if ((blocsize == 0) || (cyclelength == 0))
    return EINVAL;

  attr->datadist.type                   = KAAPIC_DATADIST_BLOCCYCLIC; /* means no distribution */    
  attr->datadist.dist.bloccyclic.size   = blocsize;
  attr->datadist.dist.bloccyclic.length = cyclelength;
  return 0;
}
