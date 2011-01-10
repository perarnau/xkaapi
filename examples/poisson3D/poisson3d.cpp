#include "poisson3d.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define DEBUG_COPY false

//------------------------------------------------------------
unsigned int Poisson3D::max_iter;

unsigned int Poisson3D::domain_sizeX;
unsigned int Poisson3D::domain_sizeY;
unsigned int Poisson3D::domain_sizeZ;
unsigned int Poisson3D::subdom_sizeX;
unsigned int Poisson3D::subdom_sizeY;
unsigned int Poisson3D::subdom_sizeZ;
unsigned int Poisson3D::nb_subdomX;
unsigned int Poisson3D::nb_subdomY;
unsigned int Poisson3D::nb_subdomZ;

double Poisson3D::hx;
double Poisson3D::hy;
double Poisson3D::hz;
double Poisson3D::rhx2;
double Poisson3D::rhy2;
double Poisson3D::rhz2;
double Poisson3D::dcoef;

const Poisson3D::Direction Poisson3D::ALL_DIRECTIONS[] = { Poisson3D::LEFT, Poisson3D::RIGHT, Poisson3D::BOTTOM, Poisson3D::TOP, Poisson3D::FRONT, Poisson3D::BACK };
const double Poisson3D::DIR_CONSTRAINTS[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

//------------------------------------------------------------
bool Poisson3D::parse_args( int argc, char** argv ) 
{
  if(argc == 4) 
  {
    subdom_sizeX = subdom_sizeY = subdom_sizeZ = atoi(argv[2]);
    nb_subdomX = nb_subdomY = nb_subdomZ = atoi(argv[3]); 
  }
  else if (argc == 8)  
  {
    subdom_sizeX = atoi(argv[2]);
    subdom_sizeY = atoi(argv[3]);
    subdom_sizeZ = atoi(argv[4]);
    nb_subdomX = atoi(argv[5]); 
    nb_subdomY = atoi(argv[6]); 
    nb_subdomZ = atoi(argv[7]);
  }
  else
  {
    std::cout << " Usage:" << std::endl;
    std::cout << "    " << argv[0] << " [nb_iter] [subdom_size] [nb_subdom]" << std::endl;
    std::cout << " OR " << argv[0] << " [nb_iter] [subdom_sizeX] [subdom_sizeY] [subdom_sizeZ] [nb_subdomX] [nb_subdomY] [nb_subdomZ]" << std::endl;
    return false;
  }
  max_iter = atoi(argv[1]);

  domain_sizeX = subdom_sizeX * nb_subdomX;
  domain_sizeY = subdom_sizeY * nb_subdomY;
  domain_sizeZ = subdom_sizeZ * nb_subdomZ;

  return true;
}

void Poisson3D::initialize() 
{
  hx = 1.0/double( domain_sizeX+1 );
  hy = 1.0/double( domain_sizeY+1 );
  hz = 1.0/double( domain_sizeZ+1 );
  rhx2 = 1/ (hx * hx);
  rhy2 = 1/ (hy * hy);
  rhz2 = 1/ (hz * hz);
  dcoef = 2.0*( rhx2 + rhy2 + rhz2 );
}

void Poisson3D::print_info() 
{
  // print info
  std::cout << "Array Dimensions: " << domain_sizeX << " " << domain_sizeY << " " << domain_sizeZ << std::endl;
  std::cout << "Block Dimensions: " << subdom_sizeX << " " << subdom_sizeY << " " << subdom_sizeZ << std::endl;
  std::cout << "Nb of SubDomains: " << nb_subdomX << " " << nb_subdomY << " " << nb_subdomZ << std::endl;
}

//------------------------------------------------------------
void Poisson3D::initialize_subdomains( Poisson3D::Index index, SubDomain& subdom, SubDomain& frhs, SubDomain& solution )
{
    subdom.resize( Poisson3D::subdom_sizeX, Poisson3D::subdom_sizeY, Poisson3D::subdom_sizeZ );
    frhs.resize( Poisson3D::subdom_sizeX, Poisson3D::subdom_sizeY, Poisson3D::subdom_sizeZ );
    solution.resize( Poisson3D::subdom_sizeX, Poisson3D::subdom_sizeY, Poisson3D::subdom_sizeZ );

    unsigned int nx = Poisson3D::subdom_sizeX;
    unsigned int ny = Poisson3D::subdom_sizeY;
    unsigned int nz = Poisson3D::subdom_sizeZ;

    for ( unsigned int x = 0 ; x < nx ; x++ )
      for ( unsigned int y = 0 ; y < ny ; y++ )
        for ( unsigned int z = 0 ; z < nz ; z++ )
        {
          unsigned int xg = 1 + x + index.get_i() * nx;
          unsigned int yg = 1 + y + index.get_j() * ny;
          unsigned int zg = 1 + z + index.get_k() * nz;

          double xx = (double)xg * Poisson3D::hx;
          double yy = (double)yg * Poisson3D::hy;
          double zz = (double)zg * Poisson3D::hz;

          frhs(x,y,z) = -2.*((xx*(xx-1.)+yy*(yy-1.))*zz*(zz-1.)+xx*(xx-1.)*yy*(yy-1.));          
          subdom(x,y,z) = 0.0;
          solution(x,y,z) = xx*yy*zz*(xx-1.)*(yy-1.)*(zz-1.);
        }
    subdom.print( std::cout << "Subdomain" );
    solution.print( std::cout << "Solution" );
//     printf( "[Poisson3D::initialize_subdomains] [%d,%d,%d] - (%d,%d,%d) - frhs(0,0,0) = %f\n", nx, ny, nz, index.get_i(), index.get_j(), index.get_k(), frhs(0,0,0) );
}

//------------------------------------------------------------
bool Poisson3D::Index::has_neighbor( Direction dir )
{
  switch( dir )
  {
    case LEFT:
      return _i > 0;
    case RIGHT:
      return _i < nb_subdomX - 1;
    case BOTTOM:
      return _j > 0;
    case TOP:
      return _j < nb_subdomY - 1;
    case FRONT:
      return _k > 0;
    case BACK:
      return _k < nb_subdomZ - 1;
    default:
      assert( false );
      return false;
  }
}

Poisson3D::Index Poisson3D::Index::get_neighbor( Direction dir )
{
  switch( dir )
  {
    case LEFT:
      return Index(_i-1,_j,_k);
    case RIGHT:
      return Index(_i+1,_j,_k);
    case BOTTOM:
      return Index(_i,_j-1,_k);
    case TOP:
      return Index(_i,_j+1,_k);
    case FRONT:
      return Index(_i,_j,_k-1);
    case BACK:
      return Index(_i,_j,_k+1);
    default:
      assert( false );
      return Index();
  }
}


//------------------------------------------------------------
SubDomain::SubDomain() : _nx(0), _ny(0), _nz(0), _data(0) 
{
  assert( _nx*_ny*_nz != 0 || _data == 0 );
  assert( _nx*_ny*_nz == 0 || _data != 0 );
}

SubDomain::SubDomain( const SubDomain& sd ) 
  : _nx(0), _ny(0), _nz(0), _data(0)
{
  if (DEBUG_COPY) std::cout << "[" << __PRETTY_FUNCTION__ << "] Copy from " << &sd << " to " << this << std::endl;
  resize( sd._nx, sd._ny, sd._nz );
  if ( sd._data != 0 )
    memcpy( _data, sd._data, (_nx+2) * (_ny+2) * (_nz+2) * sizeof(double) );
  else
    _data = 0;
  assert( _nx*_ny*_nz != 0 || _data == 0 );
  assert( _nx*_ny*_nz == 0 || _data != 0 );
}

SubDomain::~SubDomain()
{ 
  _nx = 0;
  _ny = 0;
  _nz = 0;
  if ( _data != 0 ) delete[] _data;
  _data = 0;
}

SubDomain& SubDomain::operator=( const SubDomain& sd ) 
{
  if (DEBUG_COPY) std::cout << "[" << __PRETTY_FUNCTION__ << "] Copy from " << &sd << " to " << this << std::endl;
  resize( sd._nx, sd._ny, sd._nz );
  if ( sd._data != 0 )
    memcpy( _data, sd._data, (_nx+2) * (_ny+2) * (_nz+2) * sizeof(double) );
  else
    _data = 0;
  assert( _nx*_ny*_nz != 0 || _data == 0 );
  assert( _nx*_ny*_nz == 0 || _data != 0 );
  return *this;
}

void SubDomain::resize( unsigned int nx, unsigned int ny, unsigned int nz )
{ 
  assert( _nx*_ny*_nz != 0 || _data == 0 );
  assert( _nx*_ny*_nz == 0 || _data != 0 );
  if ( (nx+2) * (ny+2) * (nz+2) != (_nx+2) * (_ny+2) * (_nz+2) ) 
  {
    if ( _data != 0 )  delete[] _data;
    if ( nx * ny * nz != 0 ) 
    {
      _data = new double[ (nx+2) * (ny+2) * (nz+2) ];
      memset( _data, 0, (nx+2) * (ny+2) * (nz+2) * sizeof(double) );
    }
    else
      _data = 0;
  }
  _nx = nx;
  _ny = ny;
  _nz = nz;
  assert( _nx*_ny*_nz != 0 || _data == 0 );
  assert( _nx*_ny*_nz == 0 || _data != 0 );
}

SubDomain& SubDomain::operator= (double value)
{ 
  for ( unsigned int x = 0 ; x < _nx ; x++ )
    for ( unsigned int y = 0 ; y < _ny ; y++ )
      for ( unsigned int z = 0 ; z < _nz ; z++ )
        (*this)(x,y,z) = value;
  return *this;
}


void SubDomain::extract_interface( Poisson3D::Direction dir, SubDomainInterface& sdi ) const
{
  switch( dir )
  {
    case Poisson3D::BOTTOM:
    {
      sdi.resize( _nx, _nz );
      for ( unsigned int x = 0 ; x < _nx ; x++ )
        for ( unsigned int z = 0 ; z < _nz ; z++ )
          sdi(x,z) = (*this)(x,0,z);
      break;
    }
    case Poisson3D::TOP:
    {
      sdi.resize( _nx, _nz );
      for ( unsigned int x = 0 ; x < _nx ; x++ )
        for ( unsigned int z = 0 ; z < _nz ; z++ )
          sdi(x,z) = (*this)(x,_ny-1,z);
      break;
    }
    case Poisson3D::LEFT:
    {
      sdi.resize( _ny, _nz );
      for ( unsigned int y = 0 ; y < _ny ; y++ )
        for ( unsigned int z = 0 ; z < _nz ; z++ )
          sdi(y,z) = (*this)(0,y,z);
      break;
    }
    case Poisson3D::RIGHT:
    {
      sdi.resize( _ny, _nz );
      for ( unsigned int y = 0 ; y < _ny ; y++ )
        for ( unsigned int z = 0 ; z < _nz ; z++ )
          sdi(y,z) = (*this)(_nx-1,y,z);
      break;
    }
    case Poisson3D::FRONT:
    {
      sdi.resize( _nx, _ny );
      for ( unsigned int x = 0 ; x < _nx ; x++ )
        for ( unsigned int y = 0 ; y < _ny ; y++ )
          sdi(x,y) = (*this)(x,y,0);
      break;
    }
    case Poisson3D::BACK:
    {
      sdi.resize( _nx, _ny );
      for ( unsigned int x = 0 ; x < _nx ; x++ )
        for ( unsigned int y = 0 ; y < _ny ; y++ )
          sdi(x,y) = (*this)(x,y,_nz-1);
      break;
    }
    default:
      assert( false );
  }
}

void SubDomain::update_internal( const SubDomain& old )
{
  resize( old._nx, old._ny, old._nz );
  for ( unsigned int x = 0 ; x < _nx ; x++ )
    for ( unsigned int y = 0 ; y < _ny ; y++ )
      for ( unsigned int z = 0 ; z < _nz ; z++ )
        (*this)(x,y,z) = Poisson3D::dcoef * old(x,y,z)
          - Poisson3D::rhx2*(old(x-1,y,z) + old(x+1,y,z))
          - Poisson3D::rhy2*(old(x,y-1,z) + old(x,y+1,z))
          - Poisson3D::rhz2*(old(x,y,z-1) + old(x,y,z+1));
}

void SubDomain::update_external( Poisson3D::Direction dir, const SubDomainInterface& sdi )
{
  switch( dir )
  {
    case Poisson3D::BOTTOM:
    {
      assert( _nx*_nz == sdi._nx * sdi._ny );
      for ( unsigned int x = 0 ; x < _nx ; x++ )
        for ( unsigned int z = 0 ; z < _nz ; z++ )
          (*this)(x,0,z) += -Poisson3D::rhy2 * sdi(x,z);
      break;
    }
    case Poisson3D::TOP:
    {
      assert( _nx*_nz == sdi._nx * sdi._ny );
      for ( unsigned int x = 0 ; x < _nx ; x++ )
        for ( unsigned int z = 0 ; z < _nz ; z++ )
          (*this)(x,_ny-1,z) += -Poisson3D::rhy2 * sdi(x,z);
      break;
    }
    case Poisson3D::LEFT:
    {
      assert( _ny*_nz == sdi._nx * sdi._ny );
      for ( unsigned int y = 0 ; y < _ny ; y++ )
        for ( unsigned int z = 0 ; z < _nz ; z++ )
          (*this)(0,y,z) += -Poisson3D::rhx2 * sdi(y,z);
      break;
    }
    case Poisson3D::RIGHT:
    {
      assert( _ny*_nz == sdi._nx * sdi._ny );
      for ( unsigned int y = 0 ; y < _ny ; y++ )
        for ( unsigned int z = 0 ; z < _nz ; z++ )
          (*this)(_nx-1,y,z) += -Poisson3D::rhx2 * sdi(y,z);
      break;
    }
    case Poisson3D::FRONT:
    {
      assert( _nx*_ny == sdi._nx * sdi._ny );
      for ( unsigned int x = 0 ; x < _nx ; x++ )
        for ( unsigned int y = 0 ; y < _ny ; y++ )
          (*this)(x,y,0) += -Poisson3D::rhz2 * sdi(x,y);
      break;
    }
    case Poisson3D::BACK:
    {
      assert( _nx*_ny == sdi._nx * sdi._ny );
      for ( unsigned int x = 0 ; x < _nx ; x++ )
        for ( unsigned int y = 0 ; y < _ny ; y++ )
          (*this)(x,y,_nz-1) += -Poisson3D::rhz2 * sdi(x,y);
      break;
    }
    default:
      assert( false );
  }
}

void SubDomain::update_external( Poisson3D::Direction dir, double value )
{
  switch( dir )
  {
    case Poisson3D::BOTTOM:
    {
      for ( unsigned int x = 0 ; x < _nx ; x++ )
        for ( unsigned int z = 0 ; z < _nz ; z++ )
          (*this)(x,0,z) += -Poisson3D::rhy2 * value;
      break;
    }
    case Poisson3D::TOP:
    {
      for ( unsigned int x = 0 ; x < _nx ; x++ )
        for ( unsigned int z = 0 ; z < _nz ; z++ )
          (*this)(x,_ny-1,z) += -Poisson3D::rhy2 * value;
      break;
    }
    case Poisson3D::LEFT:
    {
      for ( unsigned int y = 0 ; y < _ny ; y++ )
        for ( unsigned int z = 0 ; z < _nz ; z++ )
          (*this)(0,y,z) += -Poisson3D::rhx2 * value;
      break;
    }
    case Poisson3D::RIGHT:
    {
      for ( unsigned int y = 0 ; y < _ny ; y++ )
        for ( unsigned int z = 0 ; z < _nz ; z++ )
          (*this)(_nx-1,y,z) += -Poisson3D::rhx2 * value;
      break;
    }
    case Poisson3D::FRONT:
    {
      for ( unsigned int x = 0 ; x < _nx ; x++ )
        for ( unsigned int y = 0 ; y < _ny ; y++ )
          (*this)(x,y,0) += -Poisson3D::rhz2 * value;
      break;
    }
    case Poisson3D::BACK:
    {
      for ( unsigned int x = 0 ; x < _nx ; x++ )
        for ( unsigned int y = 0 ; y < _ny ; y++ )
          (*this)(x,y,_nz-1) += -Poisson3D::rhz2 * value;
      break;
    }
    default:
      assert( false );
  }
}


void SubDomain::copy( const SubDomain& sd )
{
  resize( sd._nx, sd._ny, sd._nz );
  for ( unsigned int x = 0 ; x < _nx ; x++ )
    for ( unsigned int y = 0 ; y < _ny ; y++ )
      for ( unsigned int z = 0 ; z < _nz ; z++ )
        (*this)(x,y,z) = sd(x,y,z);
}

void SubDomain::swap( SubDomain& sd )
{
  unsigned int tmp_nx = sd._nx;
  unsigned int tmp_ny = sd._ny;
  unsigned int tmp_nz = sd._nz;
  double* tmp_data = sd._data;
  sd._nx = _nx;
  sd._ny = _ny;
  sd._nz = _nz;
  sd._data = _data;  
  _nx = tmp_nx;
  _ny = tmp_ny;
  _nz = tmp_nz;
  _data = tmp_data;
}

double SubDomain::compute_residue_and_swap( const SubDomain& new_sd, const SubDomain& frhs )
{
  SubDomain& old_sd = *this;
  double res2 = 0.0;
  double invdcoef = 1.0 / Poisson3D::dcoef;
  for ( unsigned int x = 0 ; x < _nx ; ++x )
    for ( unsigned int y = 0 ; y < _ny ; ++y )
      for ( unsigned int z = 0 ; z < _nz ; ++z )
      {
        double d = ( frhs(x,y,z) - new_sd(x,y,z) );
        res2 += d*d;
        old_sd(x,y,z) = old_sd(x,y,z) + d * invdcoef;
      }
  print( std::cout << "Sub Domain: res=" << res2 << "\n" );
  return res2;
}

double SubDomain::compute_error( const SubDomain& solution ) const
{
  double error = 0.0;
  for ( unsigned int x = 0 ; x < _nx ; ++x )
    for ( unsigned int y = 0 ; y < _ny ; ++y )
      for ( unsigned int z = 0 ; z < _nz ; ++z )
        error = std::max( error, fabs( (*this)(x,y,z) - solution(x,y,z) ) );
  return error;
}

void SubDomain::print(std::ostream& cout ) const
{
  for ( unsigned int z = 0 ; z < _nz ; ++z )
  {
    for ( unsigned int x = 0 ; x < _nx ; ++x )
    {
      for ( unsigned int y = 0 ; y < _ny ; ++y )
        cout << (*this)(x,y,z) << '\t';
      cout << std::endl;
    }
  }
}

//------------------------------------------------------------
SubDomainInterface::SubDomainInterface() : _nx(0), _ny(0), _data(0) 
{
  assert( _nx*_ny != 0 || _data == 0 );
  assert( _nx*_ny == 0 || _data != 0 );
}

// --------------------------------------------------------------------
SubDomainInterface::SubDomainInterface( const SubDomainInterface& sdi ) 
 : _nx(0), _ny(0), _data(0)
{
  if (DEBUG_COPY) std::cout << "[" << __PRETTY_FUNCTION__ << "] Copy from " << &sdi << " to " << this << std::endl;
  resize( sdi._nx, sdi._ny );
  if ( sdi._data != 0 )
    memcpy( _data, sdi._data, _nx * _ny * sizeof(double) );
  else
    _data = 0;
  assert( _nx*_ny != 0 || _data == 0 );
  assert( _nx*_ny == 0 || _data != 0 );
}

// --------------------------------------------------------------------
SubDomainInterface::~SubDomainInterface()
{ 
  _nx = 0;
  _ny = 0;
  if ( _data != 0 ) delete[] _data;
  _data = 0;
}

// --------------------------------------------------------------------
SubDomainInterface& SubDomainInterface::operator=(const SubDomainInterface& sdi)
{
  if (DEBUG_COPY) std::cout << "[" << __PRETTY_FUNCTION__ << "] Copy from " << &sdi << " to " << this << std::endl;
  resize( sdi._nx, sdi._ny );
  if ( sdi._data != 0 )
    memcpy( _data, sdi._data, _nx * _ny * sizeof(double) );
  else
    _data = 0;
  assert( _nx*_ny != 0 || _data == 0 );
  assert( _nx*_ny == 0 || _data != 0 );
  return *this; 
}

// --------------------------------------------------------------------
void SubDomainInterface::resize( unsigned int nx, unsigned int ny )
{ 
  assert( _nx*_ny != 0 || _data == 0 );
  assert( _nx*_ny == 0 || _data != 0 );
  if ( nx * ny != _nx * _ny ) 
  {
    if ( _data != 0 )  delete[] _data;
    if ( nx * ny != 0 ) 
    {
      _data = new double[ nx * ny ];
      memset( _data, 0, nx * ny * sizeof(double) );
    }
    else
      _data = 0;
  }
  _nx = nx;
  _ny = ny;
  assert( _nx*_ny != 0 || _data == 0 );
  assert( _nx*_ny == 0 || _data != 0 );
}
