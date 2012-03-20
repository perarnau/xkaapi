/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** Thierry Gautier, thierry.gautier@inrialpes.fr
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
#ifndef KAAPI_TRACE_READER_FNC_H
#define KAAPI_TRACE_READER_FNC_H

#include "kaapi_event.h"
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* Set of files 
*/
struct FileSet;

/* Open a set of event files
*/
extern struct FileSet* OpenFiles( int count, const char** filenames );

/* Return the [min,max] date value. Return 0 in case of success.
*/
extern int GetInterval(struct FileSet* fdset, uint64_t* tmin, uint64_t* tmax );

/* Return the number of kprocessor. -1 in case of error
*/
extern int GetProcessorCount(struct FileSet* fdset );

/* Read and call callback on each event, ordered by date (in nanosecond)
*/
extern int ReadFiles(struct FileSet* fdset, void (*callback)( char* name, const kaapi_event_t* event) );

/* Return true iff no more event
*/
extern int EmptyEvent(struct FileSet* fdset);

/* Return current event
*/
extern const kaapi_event_t* TopEvent(struct FileSet* fdset);

/* Pass to the next
*/
extern void NextEvent(struct FileSet* fdset);

/* Read and call callback on each event, ordered by date
*/
extern int CloseFiles(struct FileSet* fdset );

#if defined(__cplusplus)
}
#endif

#endif