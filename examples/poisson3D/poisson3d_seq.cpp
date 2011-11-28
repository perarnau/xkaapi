#include "poisson3d.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

double get_WallTime()
{
   struct timeval tmp ;
   gettimeofday (&tmp, 0) ;
   return (double) tmp.tv_sec + ((double) tmp.tv_usec) * 1e-6;
}

int main( int argc, char** argv ) 
{
  std::cout << "Starting Poisson3D/Sequential with domain decomposition" << std::endl; 
  if (!Poisson3D::parse_args( argc, argv )) exit(1);  
  Poisson3D::initialize();
  Poisson3D::print_info();

  // Create SubDomains
  std::vector<SubDomain> old_domain( Poisson3D::nb_subdomX * Poisson3D::nb_subdomY * Poisson3D::nb_subdomZ );
  std::vector<SubDomain> new_domain( Poisson3D::nb_subdomX * Poisson3D::nb_subdomY * Poisson3D::nb_subdomZ );
  std::vector<SubDomain> frhs( Poisson3D::nb_subdomX * Poisson3D::nb_subdomY * Poisson3D::nb_subdomZ );
  std::vector<SubDomain> solution( Poisson3D::nb_subdomX * Poisson3D::nb_subdomY * Poisson3D::nb_subdomZ );

  // Initialize SubDomains
  for (unsigned int i = 0 ; i < Poisson3D::nb_subdomX ; i++)
    for (unsigned int j = 0 ; j < Poisson3D::nb_subdomY ; j++)
      for (unsigned int k = 0 ; k < Poisson3D::nb_subdomZ ; k++)
      {
        Poisson3D::Index curr_index = Poisson3D::Index(i,j,k);
        Poisson3D::initialize_subdomains( curr_index, old_domain[curr_index()], frhs[curr_index()], solution[curr_index()] );
//         std::cout << "[Poisson3D-seqdec] (" << i << "," << j << "," << k << ") Poisson3D() - fhrs value = " << frhs[curr_index()](0,0,0) << std::endl;    
      }

  // Iterate
  for ( unsigned int curr_iter = 0 ; curr_iter < Poisson3D::max_iter ; curr_iter++ )
  {
    double iter_start_time = get_WallTime();

    for (unsigned int i = 0 ; i < Poisson3D::nb_subdomX ; i++)
      for (unsigned int j = 0 ; j < Poisson3D::nb_subdomY ; j++)
        for (unsigned int k = 0 ; k < Poisson3D::nb_subdomZ ; k++)
        {
          Poisson3D::Index curr_index = Poisson3D::Index(i,j,k);
          SubDomain& old_subdomain = old_domain[curr_index()];
          SubDomain& new_subdomain = new_domain[curr_index()];
//           std::cout << "[Poisson3D-seqdec] (" << i << "," << j << "," << k << ") begin_iteration() - value = " << old_subdomain(0,0,0) << std::endl;

          // Update internal
          new_subdomain.update_internal( old_subdomain );
//           std::cout << "[Poisson3D-seqdec] (" << i << "," << j << "," << k << ") after update_internal() - value = " << old_subdomain(0,0,0) << std::endl;
          
#if 0
          // Extract interfaces and Update external
          for( unsigned int d = 0 ; d < Poisson3D::NB_DIRECTIONS ; d++ )
          {
            Poisson3D::Direction dir = Poisson3D::ALL_DIRECTIONS[d];
            if ( curr_index.has_neighbor( dir ) )
            {
              SubDomainInterface sdi;
              Poisson3D::Index neighbor = curr_index.get_neighbor( dir );
              old_domain[neighbor()].extract_interface( dir, sdi );
              new_subdomain.update_external( dir, sdi );
            }
            else
            {
              new_subdomain.update_external( dir, Poisson3D::DIR_CONSTRAINTS[d] );
            }
          }
#endif
        }

    double sum_res2 = 0.0;
    for (unsigned int i = 0 ; i < Poisson3D::nb_subdomX ; i++)
      for (unsigned int j = 0 ; j < Poisson3D::nb_subdomY ; j++)
        for (unsigned int k = 0 ; k < Poisson3D::nb_subdomZ ; k++)
        {
          Poisson3D::Index curr_index = Poisson3D::Index(i,j,k);

          // Compute residue and swap old/new subdomains
          SubDomain& old_subdomain = old_domain[curr_index()];
          SubDomain& new_subdomain = new_domain[curr_index()];
          SubDomain& frhs_subdomain = frhs[curr_index()];
//           std::cout << "[Poisson3D-seqdec] (" << i << "," << j << "," << k << ") before compute_residue_and_swap() - value = " << old_subdomain(0,0,0) << std::endl;
          double res2 = old_subdomain.compute_residue_and_swap( new_subdomain, frhs_subdomain );
//           std::cout << "[Poisson3D-seqdec] (" << i << "," << j << "," << k << ") end_iteration() - value = " << old_subdomain(0,0,0) << std::endl;
          sum_res2 += res2;
//           std::cout << "[Poisson3D-seqdec] (" << i << "," << j << "," << k << ") Residue^2 = " << std::setprecision(6) << res2 << std::endl;
        }
    double residue = sqrt( sum_res2 );

    double iter_end_time = get_WallTime();
    std::cout << "[Poisson3D-seqdec] Iter " << curr_iter << " - Residue = " << residue << " - Time = " << iter_end_time - iter_start_time << std::endl;
  }

  double error = 0.0;
  for (unsigned int i = 0 ; i < Poisson3D::nb_subdomX ; i++)
    for (unsigned int j = 0 ; j < Poisson3D::nb_subdomY ; j++)
      for (unsigned int k = 0 ; k < Poisson3D::nb_subdomZ ; k++)
      {
        Poisson3D::Index curr_index = Poisson3D::Index(i,j,k);
        double partial_error = old_domain[curr_index()].compute_error( solution[curr_index()] );
        error = std::max( error, partial_error );
      }
  std::cout << "[Poisson3D-seqdec] Error with solution = " << error << std::endl;

  return 0;
}

