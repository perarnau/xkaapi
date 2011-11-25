#include "poisson3diter.h"


unsigned char MeshIndex3D::const_z_iterator::table0[] =
{/* table of size 64 for the dimension 0 of the Morton code*/
    0, 1, 0, 1, 0, 1, 0, 1, 
    2, 3, 2, 3, 2, 3, 2, 3, 
    0, 1, 0, 1, 0, 1, 0, 1, 
    2, 3, 2, 3, 2, 3, 2, 3, 
    0, 1, 0, 1, 0, 1, 0, 1, 
    2, 3, 2, 3, 2, 3, 2, 3, 
    0, 1, 0, 1, 0, 1, 0, 1, 
    2, 3, 2, 3, 2, 3, 2, 3
};

unsigned char MeshIndex3D::const_z_iterator::table1[] =
{/* table of size 64 for the dimension 1 of the Morton code*/
    0, 0, 1, 1, 0, 0, 1, 1, 
    0, 0, 1, 1, 0, 0, 1, 1, 
    2, 2, 3, 3, 2, 2, 3, 3, 
    2, 2, 3, 3, 2, 2, 3, 3, 
    0, 0, 1, 1, 0, 0, 1, 1, 
    0, 0, 1, 1, 0, 0, 1, 1, 
    2, 2, 3, 3, 2, 2, 3, 3, 
    2, 2, 3, 3, 2, 2, 3, 3
};

unsigned char MeshIndex3D::const_z_iterator::table2[] =
{/* table of size 64 for the dimension 2 of the Morton code*/
    0, 0, 0, 0, 1, 1, 1, 1, 
    0, 0, 0, 0, 1, 1, 1, 1, 
    0, 0, 0, 0, 1, 1, 1, 1, 
    0, 0, 0, 0, 1, 1, 1, 1, 
    2, 2, 2, 2, 3, 3, 3, 3, 
    2, 2, 2, 2, 3, 3, 3, 3, 
    2, 2, 2, 2, 3, 3, 3, 3, 
    2, 2, 2, 2, 3, 3, 3, 3
};

const MeshIndex3D::point MeshIndex3D::directions[6]
 = {
  point(-1,0,0),
  point(1,0,0),
  point(0,-1,0),
  point(0,1,0),
  point(0,0,-1),
  point(0,0,1)
};



