/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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
#define __STDC_LIMIT_MACROS 1
#include <stdint.h>

#include <stdlib.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include <queue>
#include <string>

#include "kaapi_impl.h"
//#include <float.h>


#include "kaapi_trace_reader.h"

/* Reader for one file */
struct file_event {
  int                     fd;
  char                    name[128]; /* container name */
  kaapi_eventfile_header* header;    /* pointer to the header of the file */
  kaapi_event_t*          base;      /* base for event */
  size_t                  rpos;      /* next position to read event */
  size_t                  end;       /* past the last position to read event */
  void*                   addr;      /* memory mapped file */
  size_t                  size;      /* file size */
};

/* Compare (less) for priority queue
*/
struct next_event_t {
  next_event_t( uint64_t d=0, int f=0 )
   : date(d), fds(f) 
  {}
  uint64_t date;
  int      fds;  /* index in file_event set */
  
};
struct compare_event {
  bool operator()( const next_event_t& e1, const next_event_t& e2)
  { return e1.date > e2.date; }
};


/* Set of files 
*/
struct FileSet {
  uint64_t tmax;
  uint64_t tmin;
  std::vector<std::string> filenames;
  std::vector<file_event>  fds;
  std::priority_queue<next_event_t,std::vector<next_event_t>,compare_event> eventqueue;
};


extern "C" {

/*
*/
FileSet* OpenFiles( int count, const char** filenames )
{
  int err;
  struct stat fd_stat;
  FileSet*    fdset;
    
  fdset = new FileSet; 
  fdset->filenames.resize( count );
  fdset->fds.resize( count );
  fdset->tmin = UINT64_MAX;
  fdset->tmax = 0;

  /* open all files */
  int c = 0;
  for (int i=0; i<count; ++i)
  {
    fdset->fds[c].fd = open(filenames[i], O_RDONLY);
    if (fdset->fds[c].fd == -1) 
    {
      fprintf(stderr, "*** cannot open file '%s'\n", filenames[i]);
      exit(1);
    }
    fprintf(stdout, "*** file '%s'\n", filenames[i]);
    fdset->filenames[i] = std::string(filenames[i]);
  
    /* memory map the file */
    err = fstat(fdset->fds[c].fd, &fd_stat);
    if (err !=0)
    {
      fprintf(stderr, "*** cannot read information about file '%s'\n", 
          filenames[i]);
      return 0;
    }

    if (fd_stat.st_size ==0) 
      continue;

    fdset->fds[c].base = 0;
    fdset->fds[c].rpos = 0;
    fdset->fds[c].size = fd_stat.st_size;
    fdset->fds[c].addr = (void*)mmap(
          0, 
          fdset->fds[c].size, 
          PROT_READ|PROT_WRITE, 
          MAP_PRIVATE,
          fdset->fds[c].fd,
          0
    );
    if (fdset->fds[c].addr == (void*)-1)
    {
      fprintf(stderr, "*** cannot map file '%s', error=%i, msg=%s\n", 
          filenames[i],
          errno,
          strerror(errno)
      );
      return 0;
    }
    fdset->fds[c].header = (kaapi_eventfile_header*)fdset->fds[c].addr;
    fdset->fds[c].base   = (kaapi_event_t*)(fdset->fds[c].header+1);
    fdset->fds[c].rpos   = 0;
    fdset->fds[c].end    = (fdset->fds[c].size-sizeof(kaapi_eventfile_header)) / sizeof(kaapi_event_t);
    fdset->filenames[c]  = filenames[i];
    
    /* update min/max */
    if (fdset->fds[c].base[0].date < fdset->tmin)
      fdset->tmin = fdset->fds[c].base[0].date;
    if (fdset->tmax < fdset->fds[c].base[fdset->fds[c].end-1].date)
      fdset->tmax = fdset->fds[c].base[fdset->fds[c].end-1].date;

    /* insert date of first event in queue */
    fdset->eventqueue.push( next_event_t(fdset->fds[c].base->date, c) );

    /* */
    ++c;
  }
  
  fdset->fds.resize(c);
  fdset->filenames.resize(c);
  return fdset;
}
  


/* Read and call callback on each event, ordered by date
*/
int ReadFiles(FileSet* fdset, void (*callback)( char* name, const kaapi_event_t* event) )
{
  if (callback ==0) return EINVAL;
  
  /* sort loop ! */
  while (!fdset->eventqueue.empty())
  {
    next_event_t ne = fdset->eventqueue.top();
    fdset->eventqueue.pop();
    file_event* fe = &fdset->fds[ne.fds];
    

    /* The container name is passed in/out: first event can initialize them */
    callback( fe->name, &fe->base[fe->rpos++] );
    
    if (fe->rpos < fe->end)
      fdset->eventqueue.push( next_event_t(fe->base[fe->rpos].date, ne.fds) );
  }
  return 0;
}


const kaapi_event_t* TopEvent(FileSet* fdset)
{
  if (fdset->eventqueue.empty()) 
    return 0;

  next_event_t ne = fdset->eventqueue.top();
  file_event* fe = &fdset->fds[ne.fds];
  const kaapi_event_t* retval = &fe->base[fe->rpos];

  return retval;
}

int EmptyEvent(FileSet* fdset)
{
  if (fdset->eventqueue.empty()) 
    return 1;
  return 0;
}

void NextEvent(FileSet* fdset)
{
  if (fdset->eventqueue.empty()) 
    return;

  next_event_t ne = fdset->eventqueue.top();
  fdset->eventqueue.pop();
  file_event* fe = &fdset->fds[ne.fds];
  fe->rpos++;
    
  if (fe->rpos < fe->end)
    fdset->eventqueue.push( next_event_t(fe->base[fe->rpos].date, ne.fds) );
}

/* Return the [min,max] date value
*/
int GetProcessorCount(struct FileSet* fdset )
{
  if (fdset == 0) 
    return -1;
  return (int)fdset->fds.size();
}


/* Return the header information of the ith file of the set fdset.
   Return 0 in case of success.
*/
int GetHeader(struct FileSet* fdset, int ith, kaapi_eventfile_header* header )
{
  if (fdset == 0) 
    return EINVAL;
  if ((ith <0) || (ith >= (int)fdset->fds.size()))
    return EINVAL;
  memcpy(header, fdset->fds[ith].header, sizeof(kaapi_eventfile_header));
  return 0;
}

/* Return the [min,max] date value
*/
int GetInterval(struct FileSet* fdset, uint64_t* tmin, uint64_t* tmax )
{
  if (fdset == 0) 
    return EINVAL;
  if (fdset->fds.size() ==0)
    return EINVAL;
  *tmin = fdset->tmin;
  *tmax = fdset->tmax;
  return 0;
}


/* Read and call callback on each event, ordered by date
*/
int CloseFiles(FileSet* fdset )
{
  if (fdset ==0) return EINVAL;

  int count = (int)fdset->fds.size();  
  for (int i=0; i<count; ++i)
  {
    close(fdset->fds[i].fd);
    munmap(fdset->fds[i].addr, fdset->fds[i].size );
  }
  delete fdset;
  return 0;
}

};
