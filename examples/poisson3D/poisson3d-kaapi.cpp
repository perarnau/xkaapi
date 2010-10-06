#include "kaapi++"

#include <math.h>
#include <iomanip>

#define USE_THGRP

#if defined(USE_THGRP)
#define MySetPartition(a) ka::SetPartition(a) 
#else
#define MySetPartition(a)
#endif

#include "poisson3d.h"
#include "poisson3diter.h"

// --------------------------------------------------------------------
ka::OStream& operator<< (ka::OStream& s_out, const Poisson3D::Direction& dir )
{
  s_out << (unsigned int)dir;
  return s_out;
}

ka::IStream& operator>> ( ka::IStream& s_in, Poisson3D::Direction& dir )
{
  unsigned int value;
  s_in >> value;
  dir = (Poisson3D::Direction) value;
  return s_in;
}

// --------------------------------------------------------------------
ka::OStream& operator<< (ka::OStream& s_out, const Poisson3D::Index& index )
{
  s_out << index.get_i() << index.get_j() << index.get_k();
  return s_out;
}

ka::IStream& operator>> ( ka::IStream& s_in, Poisson3D::Index& index )
{
  unsigned int i, j, k;
  s_in >> i >> j >> k;
  index = Poisson3D::Index(i,j,k);
  return s_in;
}


// --------------------------------------------------------------------
class KaSubDomain : public SubDomain
{
  friend ka::OStream& ::operator<< (ka::OStream&, const KaSubDomain& );
  friend ka::IStream& ::operator>> (ka::IStream&, KaSubDomain& );
};

ka::OStream& operator<< (ka::OStream& s_out, const KaSubDomain& sd )
{
  s_out << sd._nx << sd._ny << sd._nz;
  if (sd._data != 0 && sd._nx * sd._ny * sd._nz > 0){
    s_out << (ka::ka_uint8_t)1;
    s_out.write( &ka::WrapperFormat<double>::format, ka::OStream::DA, sd._data, (sd._nx+2) * (sd._ny+2) * (sd._nz+2) ) ;
  }
  else 
  {
    s_out << (ka::ka_uint8_t)0;
  }
  return s_out;
}

ka::IStream& operator>> ( ka::IStream& s_in, KaSubDomain& sd )
{
  unsigned int nx, ny, nz;
  s_in >> nx >> ny >> nz;
  sd.resize( nx, ny, nz );

  ka::ka_uint8_t flag;
  s_in >> flag;
  if (flag == 1){
    s_in.read( &ka::WrapperFormat<double>::format, ka::OStream::DA, sd._data, (nx+2)*(ny+2)*(nz+2) );
  }
  else
  {
    sd._data = 0;
  }
  return s_in;
}


// --------------------------------------------------------------------
class KaSubDomainInterface : public SubDomainInterface
{
  friend ka::OStream& ::operator<< (ka::OStream&, const KaSubDomainInterface& );
  friend ka::IStream& ::operator>> (ka::IStream&, KaSubDomainInterface& );
};

ka::OStream& operator<< (ka::OStream& s_out, const KaSubDomainInterface& sdi )
{
//   std::cout << "[KaSubDomainInterface::operator<<] @ = " << &sdi << " - nx = " << sdi._nx <<  " - ny = " << sdi._ny << std::endl;
  s_out << sdi._nx << sdi._ny;
  if (sdi._data != 0 && sdi._nx * sdi._ny > 0){
    s_out << (ka::ka_uint8_t)1;
    s_out.write(  &ka::WrapperFormat<double>::format, ka::OStream::DA, sdi._data, sdi._nx * sdi._ny ) ;
  }
  else 
  {
    s_out << (ka::ka_uint8_t)0;
  }
  return s_out;
}

ka::IStream& operator>> ( ka::IStream& s_in, KaSubDomainInterface& sdi )
{
  unsigned int nx, ny;
  s_in >> nx >> ny;
 sdi.resize( nx, ny );

  ka::ka_uint8_t flag;
  s_in >> flag;
  if (flag == 1){
    s_in.read(  &ka::WrapperFormat<double>::format, ka::OStream::DA, sdi._data, nx*ny );
  }
  else
  {
    sdi._data = 0;
  }
//   std::cout << "[KaSubDomainInterface::operator>>] @ = " << &sdi << " - nx = " << sdi._nx <<  " - ny = " << sdi._ny << std::endl;
  return s_in;
}


// --------------------------------------------------------------------
class SiteGenerator
{
public:  
  SiteGenerator() : base_site(0), sub_x(1), sub_y(1), sub_z(1), dim_x(1), dim_y(1), dim_z(1) {}
    
  SiteGenerator(unsigned int bs, unsigned int sx, unsigned int sy, unsigned int sz, unsigned int dx, unsigned int dy, unsigned int dz)
    : base_site(bs), sub_x(sx), sub_y(sy), sub_z(sz), dim_x(dx), dim_y(dy), dim_z(dz)
  {}
  
  unsigned int get_site( unsigned int i, unsigned int j, unsigned int k) const
  {
    return base_site + ( ( ( ((i)/sub_x*dim_y/sub_y) + ((j)/sub_y) ) * dim_z/sub_z) + ((k)/sub_z) );
  };

  unsigned int get_base_site() const
  {
    return base_site;
  };

  unsigned int get_size() const
  {
    return ( (dim_x/sub_x) * (dim_y/sub_y) * (dim_z/sub_z) );
  };
  
  unsigned int get_max() const
  {
    return base_site - 1 + ( (dim_x/sub_x) * (dim_y/sub_y) * (dim_z/sub_z) );
  };

protected:
  unsigned int base_site;
  unsigned int sub_x;
  unsigned int sub_y; 
  unsigned int sub_z;
  unsigned int dim_x;
  unsigned int dim_y;
  unsigned int dim_z;

  friend ka::OStream& ::operator<< (ka::OStream&, const SiteGenerator& );
  friend ka::IStream& ::operator>> (ka::IStream&, SiteGenerator& );
};

ka::OStream& operator<< (ka::OStream& s_out, const SiteGenerator& sg )
{
  s_out << sg.base_site;
  s_out << sg.sub_x << sg.sub_y << sg.sub_z;
  s_out << sg.dim_x << sg.dim_y << sg.dim_z;
  return s_out;
}

ka::IStream& operator>> ( ka::IStream& s_in, SiteGenerator& sg )
{
  s_in >> sg.base_site;
  s_in >> sg.sub_x >> sg.sub_y >> sg.sub_z;
  s_in >> sg.dim_x >> sg.dim_y >> sg.dim_z;
  return s_in;
}


// --------------------------------------------------------------------
struct ExtractSubDomainInterface: public ka::Task<3>::Signature< 
        ka::R<KaSubDomain>, 
        Poisson3D::Direction,
        ka::W<KaSubDomainInterface>
> {};

template<>
struct TaskBodyCPU<ExtractSubDomainInterface> {
  void operator() ( ka::pointer_r<KaSubDomain> shared_subdomain, 
                    Poisson3D::Direction dir, 
                    ka::pointer_w<KaSubDomainInterface> shared_sdi )
  {
    shared_subdomain->extract_interface( dir, *shared_sdi );
//     std::cout << "[ExtractSubDomainInterface] @ = " << sdi << " - nx = " << sdi->get_nx() <<  " - ny = " << sdi->get_ny() << std::endl;
  }
};
static ka::RegisterBodyCPU<ExtractSubDomainInterface> dummy_object_TaskExtractSubDomainInterface;


// --------------------------------------------------------------------
struct UpdateInternal: public ka::Task<2>::Signature< 
        ka::W<KaSubDomain>,
        ka::R<KaSubDomain> 
> {};

template<>
struct TaskBodyCPU<UpdateInternal> {
  void operator() ( ka::pointer_w<KaSubDomain> shared_new_subdomain, 
                    ka::pointer_r<KaSubDomain> shared_old_subdomain )
  {
    shared_new_subdomain->update_internal( *shared_old_subdomain );
  }
};
static ka::RegisterBodyCPU<UpdateInternal> dummy_object_TaskUpdateInternal;


// --------------------------------------------------------------------
struct UpdateExternal: public ka::Task<3>::Signature< 
        ka::RW<KaSubDomain>,
        const Poisson3D::Direction&,
        ka::R<KaSubDomainInterface>
> {};

template<>
struct TaskBodyCPU<UpdateExternal> {
  void operator() ( ka::pointer_rw<KaSubDomain> shared_subdomain, 
                    const Poisson3D::Direction& dir, 
                    ka::pointer_r<KaSubDomainInterface> shared_sdi )
  {
//     std::cout << "[UpdateExternal] @ = " << &sdi << " - nx = " << sdi.get_nx() <<  " - ny = " << sdi.get_ny() << std::endl;
    shared_subdomain->update_external( dir, *shared_sdi );
  }
};
static ka::RegisterBodyCPU<UpdateExternal> dummy_object_TaskUpdateExternal;


// --------------------------------------------------------------------
struct UpdateExternalVal: public ka::Task<3>::Signature< 
        ka::RW<KaSubDomain>,
        Poisson3D::Direction,
        double
> {};

template<>
struct TaskBodyCPU<UpdateExternalVal> {
  void operator() ( ka::pointer_rw<KaSubDomain> shared_subdomain, 
                    Poisson3D::Direction dir, 
                    double value )
  {
    shared_subdomain->update_external( dir, value );
  }
};
static ka::RegisterBodyCPU<UpdateExternalVal> dummy_object_TaskUpdateExternalVal;


// --------------------------------------------------------------------
struct ComputeResidueAndSwap: public ka::Task<4>::Signature< 
        ka::RW<KaSubDomain>,
        ka::R<KaSubDomain>,
        ka::R<KaSubDomain>,
        ka::W<double>
> {};

template<>
struct TaskBodyCPU<ComputeResidueAndSwap> {
  void operator() ( ka::pointer_rw<KaSubDomain> shared_old_subdomain, 
                    ka::pointer_r<KaSubDomain> shared_new_subdomain,
                    ka::pointer_r<KaSubDomain> shared_frhs,
                    ka::pointer_w<double> shared_res2 )
  {
    *shared_res2 = shared_old_subdomain->compute_residue_and_swap( *shared_new_subdomain, *shared_frhs );
  }
};
static ka::RegisterBodyCPU<ComputeResidueAndSwap> dummy_object_TaskComputeResidueAndSwap;


// --------------------------------------------------------------------
struct ResidueSum {
  void operator()( const std::vector<double>& res2,
                   double& residue )
  {
    double sum_res2 = 0.0;
    for( unsigned int i = 0 ; i < res2.size(); ++i )
      sum_res2 += res2[i];
    residue = sqrt( sum_res2 );
    ka::logfile() << "[Poisson3D-kaapi] Residue = " << residue << std::endl; 
  }
};


// --------------------------------------------------------------------
//#define USE_Z_CURVE 1
struct Kernel {
  void operator() ( ka::ThreadGroup& threadgroup, 
                    MeshIndex3D& mesh3d,
                    std::vector<KaSubDomain>& old_domain,
                    std::vector<KaSubDomain>& new_domain,
                    const std::vector<KaSubDomain>& frhs,
                    double& residue,
                    const SiteGenerator sg,
                    std::vector<double>& res2,
                    std::vector< KaSubDomainInterface >& sdi              
                  )
  {

#if defined(USE_THGRP)
    threadgroup.begin_partition();
#else
    kaapi_frame_t frame;
    kaapi_thread_save_frame(kaapi_self_thread(), &frame);
#endif

#if defined(USE_Z_CURVE)
    MeshIndex3D::const_z_iterator ibeg;
    MeshIndex3D::const_z_iterator iend;
#else
    MeshIndex3D::const_iterator ibeg;
    MeshIndex3D::const_iterator iend;
#endif
    
#if defined(USE_Z_CURVE)
    ibeg = mesh3d.begin_z();
    iend = mesh3d.end_z();
#else
    ibeg = mesh3d.begin();
    iend = mesh3d.end();
#endif
    while (ibeg != iend)
    {
      Poisson3D::Index curr_index = *ibeg;
      int curr_site = sg.get_site(curr_index.get_i(), curr_index.get_j(), curr_index.get_k());
//std::cout << "Process subdomain:" << curr_index.get_i() << ", " << curr_index.get_j() << ", " << curr_index.get_k() << std::endl;
      // Extract interfaces from neighbors
      for( unsigned int d = 0 ; d < Poisson3D::NB_DIRECTIONS ; d++ )
      {
        Poisson3D::Direction dir = Poisson3D::ALL_DIRECTIONS[d];
        if ( curr_index.has_neighbor( dir ) )
        {
          Poisson3D::Index neighbor = curr_index.get_neighbor( dir );
          int neighbor_site = sg.get_site( neighbor.get_i(), neighbor.get_j() ,neighbor.get_k() );
#if defined(USE_THGRP)
          threadgroup.
#else
          ka::
#endif          
          Spawn<ExtractSubDomainInterface>(MySetPartition(curr_site))( &old_domain[neighbor()], dir,  &sdi[ curr_index()*Poisson3D::NB_DIRECTIONS + d ] );
        }
      }
      ++ibeg;
    }
    
#if defined(USE_Z_CURVE)
    ibeg = mesh3d.begin_z();
    iend = mesh3d.end_z();
#else
    ibeg = mesh3d.begin();
    iend = mesh3d.end();
#endif
    while (ibeg != iend)
    {
      Poisson3D::Index curr_index = *ibeg;
      int curr_site = sg.get_site(curr_index.get_i(), curr_index.get_j(), curr_index.get_k());

      // Internal computation
#if defined(USE_THGRP)
      threadgroup.
#else
      ka::
#endif          
      Spawn<UpdateInternal>( MySetPartition(curr_site) )( &new_domain[curr_index()], &old_domain[curr_index()] );

#if 0
      ++ibeg;    
    }

        
#if defined(USE_Z_CURVE)
    ibeg = mesh3d.begin_z();
    iend = mesh3d.end_z();
#else
    ibeg = mesh3d.begin();
    iend = mesh3d.end();
#endif
    while (ibeg != iend)
    {
      Poisson3D::Index curr_index = *ibeg;
      int curr_site = sg.get_site(curr_index.get_i(), curr_index.get_j(), curr_index.get_k());
#endif

      // SubDomainInterface contributions
      for( unsigned int d = 0 ; d < Poisson3D::NB_DIRECTIONS ; d++ )
      {
        Poisson3D::Direction dir = Poisson3D::ALL_DIRECTIONS[d];
        if ( curr_index.has_neighbor( dir ) )
        {
#if defined(USE_THGRP)
          threadgroup.
#else
          ka::
#endif          
          Spawn<UpdateExternal>(MySetPartition(curr_site))( &new_domain[curr_index()], dir, &sdi[curr_index()*Poisson3D::NB_DIRECTIONS + d] );
        }
        else
        {
#if defined(USE_THGRP)
          threadgroup.
#else
          ka::
#endif          
          Spawn<UpdateExternalVal>(MySetPartition(curr_site))( &new_domain[curr_index()], dir, Poisson3D::DIR_CONSTRAINTS[d] );
        }
      }

#if defined(USE_THGRP)
      threadgroup.
#else
        ka::
#endif          
        Spawn<ComputeResidueAndSwap>(MySetPartition(curr_site))( &old_domain[curr_index()], &new_domain[curr_index()], &frhs[curr_index()], &res2[curr_index()] );
      ++ibeg;
    }

#if defined(USE_THGRP)
//std::cout << "Kernel:" << std::endl << std::flush;
//    threadgroup.print();    
    threadgroup.end_partition();
    threadgroup.execute();
#else
    ka::Sync();
    kaapi_thread_restore_frame(kaapi_self_thread(), &frame);
#endif

    // Compute residue sum
    ResidueSum()(res2, residue);
  }
};

// --------------------------------------------------------------------
struct InitializeSubDomain: public ka::Task<4>::Signature< 
        Poisson3D::Index,
        ka::W<KaSubDomain>,
        ka::W<KaSubDomain>,
        ka::W<KaSubDomain>
> {};

template<>
struct TaskBodyCPU<InitializeSubDomain> {
  void operator() ( Poisson3D::Index index, 
                    ka::pointer_w<KaSubDomain> shared_subdom, 
                    ka::pointer_w<KaSubDomain> shared_frhs,
                    ka::pointer_w<KaSubDomain> shared_sol )
  {
    Poisson3D::initialize_subdomains( index, *shared_subdom, *shared_frhs, *shared_sol );
  }
};
static ka::RegisterBodyCPU<InitializeSubDomain> dummy_object_TaskInitializeSubDomain;


// --------------------------------------------------------------------
struct Initialize {
  void operator() ( ka::ThreadGroup& threadgroup, 
                    std::vector<KaSubDomain>& domain,
                    std::vector<KaSubDomain>& frhs,
                    std::vector<KaSubDomain>& solution,
                    const SiteGenerator sg )
  {
#if defined(USE_THGRP)
    threadgroup.begin_partition();
#endif
    for( unsigned int i = 0 ; i < Poisson3D::nb_subdomX ; ++i )
      for( unsigned int j = 0 ; j < Poisson3D::nb_subdomY; ++j )
        for( unsigned int k = 0 ; k < Poisson3D::nb_subdomZ ; ++k )
        {
          int site = sg.get_site(i,j,k);
          Poisson3D::Index curr_index(i,j,k);
#if defined(USE_THGRP)
          threadgroup.
#else
          ka::
#endif          
          Spawn<InitializeSubDomain> ( MySetPartition(site) ) ( 
              curr_index, 
              ka::pointer<KaSubDomain>(&domain[curr_index()]), 
              ka::pointer<KaSubDomain>(&frhs[curr_index()]),
              ka::pointer<KaSubDomain>(&solution[curr_index()]) 
          );
        }
#if defined(USE_THGRP)
//std::cout << "Initialize:" << std::endl << std::flush;
//    threadgroup.print();    
    threadgroup.end_partition();
    threadgroup.execute();    
#else
    ka::Sync();
#endif
  }
};

// --------------------------------------------------------------------
struct ComputeError: public ka::Task<3>::Signature< 
        ka::R<KaSubDomain>,
        ka::R<KaSubDomain>,
        ka::W<double>
> {};

template<>
struct TaskBodyCPU<ComputeError> {
  void operator() ( ka::pointer_r<KaSubDomain> shared_subdomain, 
                    ka::pointer_r<KaSubDomain> shared_solution, 
                    ka::pointer_w<double> shared_error )
  {
    *shared_error = shared_subdomain->compute_error( *shared_solution );
  }
};
static ka::RegisterBodyCPU<ComputeError> dummy_object_TaskComputeError;


// --------------------------------------------------------------------
struct ErrorMax {
  void operator()( const std::vector<double>& shared_error )
  {
    double error = 0.0;
    for( unsigned int i = 0 ; i < shared_error.size() ; ++i )
      error = std::max( error, shared_error[i] );
    ka::logfile() << "[Poisson3D-kaapi] Error with solution = " << error << std::endl; 
  }
};

// --------------------------------------------------------------------
struct Verification {
  void operator() ( ka::ThreadGroup& threadgroup, 
                    std::vector<KaSubDomain>& domain,
                    std::vector<KaSubDomain>& solution,
                    SiteGenerator sg )
  {
#if defined(USE_THGRP)
    threadgroup.begin_partition();
#endif

    std::vector<double> error(Poisson3D::nb_subdomX * Poisson3D::nb_subdomY * Poisson3D::nb_subdomZ );

    // Compute error on all subdomains
    for( unsigned int i = 0 ; i < Poisson3D::nb_subdomX ; ++i )
      for( unsigned int j = 0 ; j < Poisson3D::nb_subdomY; ++j )
        for( unsigned int k = 0 ; k < Poisson3D::nb_subdomZ ; ++k ){
          int site = sg.get_site(i,j,k);
          Poisson3D::Index curr_index(i,j,k);
#if defined(USE_THGRP)
          threadgroup.
#else
          ka::
#endif          
          Spawn<ComputeError>( MySetPartition(site) ) ( 
            ka::pointer<KaSubDomain>(&domain[curr_index()]),
            ka::pointer<KaSubDomain>(&solution[curr_index()]), 
            ka::pointer<double>(&error[curr_index()]) 
          );
        }
#if defined(USE_THGRP)
//std::cout << "Verification:" << std::endl << std::flush;
//    threadgroup.print();    
    threadgroup.end_partition();
    threadgroup.execute();    
#else
    ka::Sync();
#endif
    // Compute error max
    ErrorMax() ( error );
  }
};


// --------------------------------------------------------------------
/** Main Task, only executed on one process.
*/
struct doit {
  void operator()(int argc, char** argv )
  {   
    ka::logfile() << "Starting Poisson3D/Kaapi" << std::endl;  
    Poisson3D::print_info();

    // number of partitions = X*Y*Z
    ka::ThreadGroup threadgroup( Poisson3D::nb_subdomX * Poisson3D::nb_subdomY * Poisson3D::nb_subdomZ );

    // One subdomain per site
    SiteGenerator sg(0, 1, 1, 1, Poisson3D::nb_subdomX, Poisson3D::nb_subdomY, Poisson3D::nb_subdomZ);

    // Computation domain
    std::vector< KaSubDomain > domain( Poisson3D::nb_subdomX * Poisson3D::nb_subdomY * Poisson3D::nb_subdomZ );
    std::vector< KaSubDomain > new_domain( Poisson3D::nb_subdomX * Poisson3D::nb_subdomY * Poisson3D::nb_subdomZ );

    // Righ-hand side
    std::vector< KaSubDomain > frhs( Poisson3D::nb_subdomX * Poisson3D::nb_subdomY * Poisson3D::nb_subdomZ );

    // Solution
    std::vector< KaSubDomain > solution( Poisson3D::nb_subdomX * Poisson3D::nb_subdomY * Poisson3D::nb_subdomZ );

    // Error, allocate in the stack
    double residue;

    // Initialize subdomains
    Initialize()( threadgroup, domain, frhs, solution, sg );

    // Kernel loop
    std::vector<double> res2( domain.size() );
    std::vector< KaSubDomainInterface > sdi( domain.size() * Poisson3D::NB_DIRECTIONS );
    MeshIndex3D mesh3d( Poisson3D::nb_subdomX, Poisson3D::nb_subdomY, Poisson3D::nb_subdomZ );

    double t0, t1, total = 0;
    for (unsigned int i=0; i<Poisson3D::max_iter; ++i)
    {
      t0 = kaapi_get_elapsedtime();
      Kernel()( threadgroup, mesh3d, domain, new_domain, frhs, residue, sg, res2, sdi );
      t1 = kaapi_get_elapsedtime();
      if (i>0) total += t1-t0;
      std::cout << "Time: " << t1 - t0 << std::endl;
    }
    std::cout << "Avrg Time: " << total/(Poisson3D::max_iter-1) << std::endl;

    // Check the result
    Verification()( threadgroup, domain, solution, sg );
  }
};


// --------------------------------------------------------------------
/** Main of the program
*/
int main( int argc, char** argv ) 
{
  try {
    ka::Community com = ka::System::join_community( argc, argv );
//     ka::logfile() << "[Poisson3Dbis::main] pid = " << getpid() << " - gid = " << Util::Init::local_gid 
//         << " - nb_total_nodes = " << WS::Init::world->size() << " - nb_local_threads = " << Util::KaapiComponentManager::prop["community.thread.poolsize"] << std::endl;

    if (!Poisson3D::parse_args( argc, argv )) exit(1);  
    Poisson3D::initialize();

    ka::SpawnMain<doit>()(argc, argv); 
    com.leave();
    ka::System::terminate();
  }
  catch (ka::Exception& ex) {
    std::cerr << "[main] catch exception: " << ex.what() << std::endl;
  }
  catch (...) {
    std::cerr << "[main] catch unknown exception" << std::endl;
  }
  return 0;
}

