#ifndef _POISSON3D_H_
#define _POISSON3D_H_
class SubDomain;
class SubDomainInterface;

//------------------------------------------------------------
class Poisson3D
{
public:

  enum Direction {
    LEFT,    /* X-1,Y,Z */
    RIGHT,   /* X+1,Y,Z */
    BOTTOM,  /* X,Y-1,Z */
    TOP,     /* X,Y+1,Z */
    FRONT,   /* X,Y,Z-1 */
    BACK     /* X,Y,Z+1 */
  };

  static const unsigned int NB_DIRECTIONS = 6;
  static const Direction ALL_DIRECTIONS[NB_DIRECTIONS];  
  static const double DIR_CONSTRAINTS[NB_DIRECTIONS];

  class Index
  {
    public:
      Index() : _i(-1), _j(-1), _k(-1) {}
      Index( unsigned int i, unsigned int j, unsigned int k ) : _i(i), _j(j), _k(k) {}
      bool has_neighbor( Direction dir );
      Index get_neighbor( Direction dir );
      inline unsigned int operator()() const
      { 
        return ( ( ((_i) * Poisson3D::nb_subdomY) + (_j) ) * Poisson3D::nb_subdomZ + (_k) ); 
      }


      unsigned int get_i() const { return _i; }
      unsigned int get_j() const { return _j; }
      unsigned int get_k() const { return _k; }

    protected:
      unsigned int _i;
      unsigned int _j;
      unsigned int _k;
  };

  static unsigned int max_iter;

  static unsigned int domain_sizeX;
  static unsigned int domain_sizeY;
  static unsigned int domain_sizeZ;
  static unsigned int subdom_sizeX;
  static unsigned int subdom_sizeY;
  static unsigned int subdom_sizeZ;
  static unsigned int nb_subdomX;
  static unsigned int nb_subdomY;
  static unsigned int nb_subdomZ;
  
  static double hx;
  static double hy;
  static double hz;
  static double rhx2;
  static double rhy2;
  static double rhz2;
  static double dcoef;

  static bool parse_args( int argc, char** argv );
  static void initialize();
  static void print_info();
  static void initialize_subdomains( Poisson3D::Index index, SubDomain& subdom, SubDomain& frhs, SubDomain& solution );
};

//------------------------------------------------------------
class SubDomain 
{
public:

  SubDomain();
  SubDomain( const SubDomain& sd );
  ~SubDomain();

  SubDomain& operator=( const SubDomain& sd );

  void resize( unsigned int nx, unsigned int ny, unsigned int nz );

  SubDomain& operator= (double value);

  inline const double& operator() (unsigned int i, unsigned int j, unsigned int k) const
  {
    return _data[ ((i+1)*(_ny+2) + (j+1)) * (_nz+2) + (k+1) ];
  }

  inline double& operator() (unsigned int i, unsigned int j, unsigned int k)
  {
    return _data[ ((i+1)*(_ny+2) + (j+1)) * (_nz+2) + (k+1) ];
  }

  void extract_interface( Poisson3D::Direction dir, SubDomainInterface& sdi ) const;
  void update_internal( const SubDomain& old );
  void update_external( Poisson3D::Direction dir, const SubDomainInterface& sdi );
  void update_external( Poisson3D::Direction dir, double value );
  void copy( const SubDomain& sd );
  void swap( SubDomain& sd );
  double compute_residue_and_swap( const SubDomain& new_sd, const SubDomain& frhs );
  double compute_error( const SubDomain& solution ) const;  

protected:
  unsigned int _nx;
  unsigned int _ny;
  unsigned int _nz;
  double* _data;

  friend class Poisson;
};

//------------------------------------------------------------
class SubDomainInterface
{
public:

  SubDomainInterface();  
  SubDomainInterface( const SubDomainInterface& sdi );
  ~SubDomainInterface();

  SubDomainInterface& operator=(const SubDomainInterface& sdi);

  void resize(unsigned int nx, unsigned int ny);

  inline const double& operator() (unsigned int i, unsigned int j) const
  { return _data[ (i*_ny) + j ]; }
  
  inline double& operator() (unsigned int i, unsigned int j)
  { return _data[ (i*_ny) + j ]; }

  unsigned int get_nx() const
  { return _nx; }

  unsigned int get_ny() const
  { return _ny; }

protected:
  unsigned int _nx;
  unsigned int _ny;
  double* _data;
  friend class SubDomain;
};
#endif
