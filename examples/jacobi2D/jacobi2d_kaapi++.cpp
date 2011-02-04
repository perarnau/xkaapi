/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
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
#include "kaapi++"
#include <iostream>

/**
*/
enum Direction {
  NODIR   = 0x0,
  LEFT    = 0x1,
  RIGHT   = 0x2,
  BOTTOM  = 0x4,
  UP      = 0x8
};

/* return the index of the direction in 0..3 */
int Direction2Index( Direction dir)
{
  switch (dir) {
    case LEFT:
      return 0;
    case RIGHT:
      return 1;
    case BOTTOM:
      return 2;
    case UP:
      return 3;
    default:
      break;
  }
  return -1;
}

/* return the index of the i-direction in -1 or 1*/
int Direction2i( Direction dir)
{
  switch (dir) {
    case BOTTOM:
      return -1;
    case UP:
      return 1;
    default:
      break;
  }
  return 0;
}

/* return the index of the direction in 0..3 */
int Direction2j( Direction dir)
{
  switch (dir) {
    case LEFT:
      return -1;
    case RIGHT:
      return 1;
    default:
      break;
  }
  return 0;
}

/* return the index of the i-direction in -1 or 1*/
int IndexDirection2i( int dir)
{
  switch (dir) {
    case 2:
      return -1;
    case 3:
      return 1;
    default:
      break;
  }
  return 0;
}

/* return the index of the direction in 0..3 */
int IndexDirection2j( int dir)
{
  switch (dir) {
    case 0:
      return -1;
    case 1:
      return 1;
    default:
      break;
  }
  return 0;
}

/* return the index of the direction in 0..3 */
static Direction table_i2d[] = {
  LEFT,
  RIGHT,
  BOTTOM,
  UP
};
inline Direction Index2Direction( int index )
{
  return table_i2d[index];
}

static const char* table_i2dn[] = {
  "LEFT",
  "RIGHT",
  "BOTTOM",
  "UP"
};

inline const char* Direction2Name( Direction dir )
{
  switch (dir) {
    case LEFT:
      return "LEFT";
    case RIGHT:
      return "RIGHT";
    case BOTTOM:
      return "BOTTOM";
    case UP:
      return "UP";
    default:
      break;
  }
  return "<bad name>";
}

static Direction table_i2od[] = {
  RIGHT,
  LEFT,
  UP,
  BOTTOM
};
inline int Index2OppositeIndex(int dir)
{
  switch (dir) {
    case 0:
      return 1;
    case 1:
      return 0;
    case 2:
      return 3;
    case 3:
      return 2;
    default:
      break;
  }
  return -10;
}


// --------------------------------------------------------------------
struct TaskInit: public ka::Task<4>::Signature<int, int, int, ka::W<ka::range2d<double> > > {};
template<>
struct TaskBodyCPU<TaskInit> {
  void operator() ( int bi, int bj, int dir, ka::range2d_w<double> D )
  {
    D = 0.0;

    /* limit condition: 1 if the box is a limit box */
    if (dir & LEFT)
    {
      unsigned di = D.dim(0);
      for (unsigned i=0; i < di; ++i)
        D(i,0U) = 1.0;
    }
    if (dir & RIGHT)
    {
      unsigned di = D.dim(0);
      unsigned last = D.dim(1)-1;
      for (unsigned i=0; i < di; ++i)
        D(i,last) = 1.0;
    }
    if (dir & BOTTOM)
    {
      unsigned dj = D.dim(1);
      unsigned last = D.dim(0)-1;
      for (unsigned j=0; j < dj; ++j)
        D(last,j) = 1.0;
    }
    if (dir & UP)
    {
      unsigned dj = D.dim(1);
      for (unsigned j=0; j < dj; ++j)
        D(0U,j) = 1.0;
    }
//    std::cout << ka::System::local_gid << "::[TaskInit] pos:" << pos << ", D: 1.0" << std::endl;
  }
};


// --------------------------------------------------------------------
struct TaskPrint: public ka::Task<3>::Signature<int, int, ka::R<ka::range2d<double> > > {};
template<>
struct TaskBodyCPU<TaskPrint> {
  void operator() ( int bi, int bj, ka::range2d_r<double> D )
  {
    std::cout << ka::System::local_gid << "::[TaskPrint] pos: (" << bi << ',' << bj << "):\n";
    for (unsigned i=0; i<D.dim(0); ++i)
    {
      for (unsigned j=0; j<D.dim(1); ++j)
      {
        printf("%3.9e   ", D(i,j) );
      }
      printf("\n");
    }
    printf("\n\n");
  }
};


// --------------------------------------------------------------------
struct TaskUpdateInternal: public ka::Task<3>::Signature<
             int, int, 
             ka::RW<ka::range2d<double> > 
> {};
template<>
struct TaskBodyCPU<TaskUpdateInternal> {
  void operator() ( int bi, int bj, ka::range2d_rw<double> D )
  {
#if 0
    std::cout << ka::System::local_gid << "::TaskUpdateInternal (" << bi << "," << bj << ")" << std::endl;
#endif
    unsigned i,j;
    unsigned di = D.dim(0);
    unsigned dj = D.dim(1);
    double dcoef = 1/8.0;

    /* up row */
    for (j=1; j < dj-1; ++j)
      D(0U,j) = dcoef*(4.0*D(0U,j) + D(1U,j) + D(0U,j-1) + D(0U,j+1));

    for (i=1; i < di-1; ++i)
    {
      /* left colum */
      D(i,0U) = dcoef*(4.0*D(i,0U) + D(i-1,0U) + D(i+1,0U) + D(i,1U));

      /* internal */
      for (j=1; j < dj-1; ++j)
        D(i,j) = dcoef*(4.0*D(i,j) + D(i-1,j) + D(i+1,j) + D(i,j-1) + D(i,j+1));

      /* right colum */
      D(i,j) = dcoef*(4.0*D(i,j) + D(i-1,j) + D(i+1,j) + + D(i,j-1));
    }

    /* bottom row */
    for (j=1; j < dj-1; ++j)
      D(di-1,j) = dcoef*(4.0*D(di-1,j) + D(di-2,j) + D(di-1,j-1) + D(di-1,j+1));
    
    /* corners */
    D(0U,0U) = dcoef*(4.0*D(0U,0U)+D(1U,0U)+D(0U,1U));
    D(0U,dj-1) = dcoef*(4.0*D(0U,dj-1)+D(1U,dj-1)+D(0U,dj-2));
    D(di-1,0U) = dcoef*(4.0*D(di-1,0U)+D(di-2,0U)+D(di-1,1U));
    D(di-1,dj-1) = dcoef*(4.0*D(di-1,dj-1)+D(di-2,dj-1)+D(di-1,dj-2));
  }
};


// --------------------------------------------------------------------
struct TaskUpdateExternal: public ka::Task<5>::Signature<
             int, int, 
             ka::RW<ka::range2d<double> >,
             Direction, /* position of the fontier with respect to the domain */
             ka::R<ka::range1d<double> >
> {};
template<>
struct TaskBodyCPU<TaskUpdateExternal> {
  void operator() ( int bi, int bj, ka::range2d_rw<double> D, Direction dir, ka::range1d_r<double> F )
  {
    unsigned i,j;
    unsigned di = D.dim(0);
    unsigned dj = D.dim(1);
    double dcoef = 1/8.0;

#if 0
    std::cout << ka::System::local_gid << "::TaskUpdateExternal [ " << bi << "," << bj << " ] " 
              << Direction2Name(dir) 
              << "  Fontier| @F=" << *(void**)(void*)&F << " =[ ";
    for ( i=0; i<F.size(); ++i)
      std::cout << F(i) << "  ";
    std::cout << "]" << std::endl;
#endif

    switch (dir) 
    {
      case LEFT:
      {
        for (i=0; i < di; ++i)
          D(i,0U) += dcoef * F(i);
      } break;

      case RIGHT:
      {
        for (i=0; i < di; ++i)
          D(i,dj-1) += dcoef * F(i);
      } break;

      case BOTTOM:
      {
        for (j=0; j < dj; ++j)
          D(di-1,j) += dcoef*F(j);
      } break;

      case UP:
      {
        for (j=0; j < dj; ++j)
          D(0U,j) += dcoef*F(j);
      } break;

      default:
        break;
    }
  }
};


// --------------------------------------------------------------------
struct TaskExtractF: public ka::Task<5>::Signature<
            int, int, 
            ka::W<ka::range1d<double> >, 
            Direction, /* position of the fontier to extract with respect to the domain */
            ka::R<ka::range2d<double> > 
> {};
template<>
struct TaskBodyCPU<TaskExtractF> {
  void operator() ( int bi, int bj, ka::range1d_w<double> F, Direction dir, ka::range2d_r<double> D )
  {
    unsigned i,j;
    unsigned di = D.dim(0);
    unsigned dj = D.dim(1);
    switch (dir) 
    {
      case LEFT:
      {
        for (i=0; i < di; ++i)
          F(i) = D(i,0U);
      } break;

      case RIGHT:
      {
        for (i=0; i < di; ++i)
          F(i) = D(i,dj-1U);
      } break;

      case BOTTOM:
      {
        for (j=0; j < dj; ++j)
          F(j) = D(di-1U,j);
      } break;

      case UP:
      {
        for (j=0; j < dj; ++j)
          F(j) = D(0U,j);
      } break;

      default:
        break;
    }
#if 0
    std::cout << ka::System::local_gid << "::TaskExtractF[ " << bi << "," << bj << " ] " << Direction2Name(dir) 
              << ", @F=" << *(void**)(void*)&F << "  Fontier| F=[ ";
    for ( i=0; i<F.size(); ++i)
      std::cout << F(i) << "  ";
    std::cout << "]" << std::endl;
#endif
  }
};


/* set partition functor */
struct SetPartition {
  SetPartition(int ni, int nj)
   : _ni(ni), _nj(nj)
  {}
  int operator()(int i, int j)
  { return _nj*i+j; }
private:
  int _ni, _nj;
};


/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    std::cout << "My pid=" << getpid() << " -> my rank:" << ka::System::local_gid << std::endl;
    unsigned dn = 4;    /* size of the subdomain */
    unsigned nbloc = 2; /* number of the bloc in each direction */
    unsigned iter = 10;
#if 1
    if (argc >1)
      dn = atoi(argv[1]);
    if (argc >2)
      nbloc = atoi(argv[2]);
    if (argc >3)
      iter = atoi(argv[3]);
#endif
    unsigned n = nbloc*dn;

    SetPartition setpart(nbloc,nbloc);
    
    ka::ThreadGroup threadgroup( nbloc*nbloc );
    
    std::vector<double> domain(n*n);
    /* 2D view of the domain, lda = n */
    ka::array<2, double> D(&domain[0], n, n, n);          

    std::vector<double> fontier(4*nbloc*nbloc*n);
    /* 1D view of all the fontiers, fontier dir for bloc i, j 
       begins at index SetPartition(i,j)+n*IndexDirection(dir) and has size n
    */
    ka::array<1, double> F(&fontier[0], 4*nbloc*nbloc*n); 
    
    /* range */
    typedef ka::array<1,double>::range range;

    /* use default  mapping */
    threadgroup.begin_partition( );

    int dir = NODIR;
    for (unsigned i=0; i<nbloc; ++i)
    {
      range ri(i*dn, (i+1)*dn);
      for (unsigned j=0; j<nbloc; ++j)
      {
        dir = NODIR;
        if (i ==0) 
          dir |= UP;
        if (i == nbloc-1) 
          dir |= BOTTOM;
        if (j ==0) 
          dir |= LEFT;
        if (j == nbloc-1) 
          dir |= RIGHT;
        range rj(j*dn, (j+1)*dn);
        ka::Spawn<TaskInit> (ka::SetPartition( setpart(i,j) ))  
                            ( i, j, dir, D(ri,rj) );
#if 0
        ka::Spawn<TaskPrint> (ka::SetPartition( setpart(i,j) ))  
                             ( i, j, D(ri,rj) );
#endif
      }
    }
    threadgroup.end_partition();

    threadgroup.execute();
    
#if 0
    /* test synchronization: only for multicore... */
    threadgroup.synchronize( );
    printf("\n\n*********** After initialisation\n");
    for (unsigned i=0; i<n; ++i)
    {
      for (unsigned j=0; j<n; ++j)
      {
        printf("%3.9e ", domain[i*n+j]);
      }
      printf("\n");
    }
    printf("\n\n");
#endif    

    threadgroup.begin_partition( KAAPI_THGRP_SAVE_FLAG );

#if 0
    for (unsigned i=0; i<nbloc; ++i)
    {
      range ri(i*dn, (i+1)*dn);
      for (unsigned j=0; j<nbloc; ++j)
      {
        range rj(j*dn, (j+1)*dn);
        ka::Spawn<TaskPrint> (ka::SetPartition( setpart(i,j) ))  
                             ( i, j, D(ri,rj) );
      }
    }
#endif

    for (unsigned i=0; i<nbloc; ++i)
    {
      range ri(i*dn, (i+1)*dn);
      for (unsigned j=0; j<nbloc; ++j)
      {
        range rj(j*dn, (j+1)*dn);
        for (unsigned dir = 0; dir <4; ++dir)
        {
          int dir_i = IndexDirection2i(dir);
          int dir_j = IndexDirection2j(dir);
          if ( ((i+dir_i >=0) && (i+dir_i < nbloc))
            && ((j+dir_j >=0) && (j+dir_j < nbloc)) )
          {
            range r(setpart(i,j)*4*nbloc + dir*dn, setpart(i,j)*4*nbloc + (dir+1)*dn);
            ka::Spawn<TaskExtractF> (ka::SetPartition(setpart(i,j)))  
                                    ( i, j, F(r), Index2Direction(dir), D(ri,rj) );
          }
        }
      }
    }
    for (unsigned i=0; i<nbloc; ++i)
    {
      range ri(i*dn, (i+1)*dn);
      for (unsigned j=0; j<nbloc; ++j)
      {
        range rj(j*dn, (j+1)*dn);
        ka::Spawn<TaskUpdateInternal> (ka::SetPartition(setpart(i,j)))  
                                      ( i, j, D(ri,rj) );
      }
    }

    for (unsigned i=0; i<nbloc; ++i)
    {
      range ri(i*dn, (i+1)*dn);
      for (unsigned j=0; j<nbloc; ++j)
      {
        range rj(j*dn, (j+1)*dn);
        for (unsigned dir = 0; dir <4; ++dir)
        {
          int dir_i = IndexDirection2i(dir);
          int dir_j = IndexDirection2j(dir);
          int opdir = Index2OppositeIndex(dir);
          if ( ((i+dir_i >=0) && (i+dir_i < nbloc))
            && ((j+dir_j >=0) && (j+dir_j < nbloc)) )
          {
            range r(setpart(i+dir_i,j+dir_j)*4*nbloc + opdir*dn, 
                    setpart(i+dir_i,j+dir_j)*4*nbloc + (opdir+1)*dn );
            ka::Spawn<TaskUpdateExternal> (ka::SetPartition(setpart(i,j)))  
                                          ( i, j, D(ri,rj), Index2Direction(opdir), F(r) );
          }
        }
      }
    }

    threadgroup.end_partition();

    threadgroup.set_iteration_step( iter );
    
    for (unsigned k=0; k<iter; ++k)
    {
      printf("\n\n*********** Step %i\n", k);
      threadgroup.execute();
      
#if 0
      if (k == iter-1)
      {
        /* memory synchronize : only D !!! */
        threadgroup.synchronize( );

        printf("\n\n%i::*********** Value at step %i\n", ka::System::local_gid, k);
        for (unsigned i=0; i<n; ++i)
        {
          for (unsigned j=0; j<n; ++j)
          {
            printf("%3.9e ", domain[i*n+j]);
          }
          printf("\n");
        }
        printf("\n\n");
      }
#endif
    }

  }
};


/*
*/
int main( int argc, char** argv ) 
{
  try {
    ka::Community com = ka::System::join_community( argc, argv );
    
    ka::SpawnMain<doit>()(argc, argv); 
          
    com.leave();

    ka::System::terminate();
  }
  catch (const ka::Exception& E) {
    ka::logfile() << "Catch : "; E.print(std::cout); std::cout << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  return 0;    
}
