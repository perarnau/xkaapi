#ifndef _POISSON3DITER_H_
#define _POISSON3DITER_H_

/* Iteration over 3D mesh index
*/
class MeshIndex3D {
public:
  /**
  */
  MeshIndex3D()
  { dim[0] = 0; dim[1] = 0; dim[2] = 0; }

  /**
  */
  MeshIndex3D(unsigned short i, unsigned short j, unsigned short k )
  { dim[0] = i; dim[1] = j; dim[2] = k; }
  
  /**
  */
  void resize( unsigned short i, unsigned short j, unsigned short k )
  { dim[0] = i; dim[1] = j; dim[2] = k; }

  /** Point of a mesh 
  */
  class point {
  public:
    point() : _i(0), _j(0), _k(0) {}
    point( int i, int j, int k ) : _i(i), _j(j), _k(k) {}

    int get_i() const { return _i; }
    int get_j() const { return _j; }
    int get_k() const { return _k; }

    int operator[](unsigned short i) const
    {
      const int* v = (const int *)this;
      return v[i];
    }
    bool operator==(const point& p) const
    { return (_i == p._i) && (_j == p._j) && (_k == p._k); }
    bool operator!=(const point& p) const
    { return (_i != p._i) || (_j != p._j) || (_k != p._k); }

    point operator+( const point& p ) const
    { return point(_i+p._i, _j+p._j, _k+p._k); }
  protected:
    int _i;
    int _j;
    int _k;
    friend class MeshIndex3D;
  };

  /* convert to unsigned int value in order to linearize mesh */
  unsigned int get_index( const point& p) const
  { 
    return ( (p._i * dim[1]) + p._j ) * dim[2] + p._k; 
  }

  /* return true iff the point is inside the mesh */
  bool is_inside( const point& p) const
  { 
    return (p[0] >=0) && (p[0] < dim[0]) 
        && (p[1] >=0) && (p[1] < dim[1]) 
        && (p[2] >=0) && (p[2] < dim[2]);
  }

  enum Direction {
    LEFT,    /* X-1,Y,Z */
    RIGHT,   /* X+1,Y,Z */
    BOTTOM,  /* X,Y-1,Z */
    TOP,     /* X,Y+1,Z */
    FRONT,   /* X,Y,Z-1 */
    BACK     /* X,Y,Z+1 */
  };
  
  static const point directions[6];

  /**
  */
  class const_neighbor_iterator {
  public:
    /* equality */
    bool operator==(const const_neighbor_iterator& it) const
    { return (p == it.p) && (dir == it.dir); }
    /* not equal */
    bool operator!=(const const_neighbor_iterator& it) const
    { return (p != it.p) && (dir != it.dir); }
    /* pre increment */
    const_neighbor_iterator& operator++()
    {
      ++dir;
      for ( ; dir < 6; ++dir)
        if (mesh->is_inside( p+MeshIndex3D::directions[dir])) return *this;
    }

    /* post increment */
    const_neighbor_iterator operator++(int)
    {
      const_neighbor_iterator retval = *this;
      ++(*this);
      return retval;
    }
    
    /* indirection return the point */
    point operator*()
    { return p+MeshIndex3D::directions[dir]; }

  protected:
    friend class MeshIndex3D;
  
  private:
    const MeshIndex3D* mesh;
    unsigned int       dir;
    point              p;
  };
  
  /**
  */
  class const_iterator {
  public:
    /* equality */
    bool operator==(const const_iterator& it) const
    { return (pos[0] == it.pos[0]) && (pos[1] == it.pos[1]) && (pos[2] == it.pos[2]); }
    /* not equal */
    bool operator!=(const const_iterator& it) const
    { return (pos[0] != it.pos[0]) || (pos[1] != it.pos[1]) || (pos[2] != it.pos[2]); }
    /* pre increment */
    const_iterator& operator++()
    {
      if (++pos[2] < mesh->dim[2]) return *this;
      pos[2] = 0;
      if (++pos[1] < mesh->dim[1]) return *this;
      pos[1] = 0;
      ++pos[0];
      return *this;
    }
    /* post increment */
    const_iterator operator++(int)
    {
      const_iterator retval = *this;
      ++*this;
      return retval;
    }
    /* indirection return the point */
    point operator*()
    { return point(pos[0], pos[1], pos[2]); }

    /**
    */
    const_neighbor_iterator begin() const
    { 
      const_neighbor_iterator retval;
      retval.mesh = mesh;
      retval.dir  = -1;
      retval.p    = point(pos[0], pos[1], pos[2]); 
      ++retval;
      return retval;
    }
    
    /**
    */
    const_neighbor_iterator end() const
    {
      const_neighbor_iterator retval;
      retval.mesh = mesh;
      retval.dir  = 6;
      retval.p    = point(pos[0], pos[1], pos[2]); 
      return retval;
    }

  protected:
    friend class MeshIndex3D;
  
  private:
    const MeshIndex3D* mesh;
    int pos[3];
  };

  /**
  */
  const_iterator begin() const
  { 
    const_iterator retval;
    retval.mesh = this;
    retval.pos[0] = 0;
    retval.pos[1] = 0;
    retval.pos[2] = 0;
    return retval;
  }
  
  /**
  */
  const_iterator end() const
  {
    const_iterator retval;
    retval.mesh = this;
    retval.pos[0] = dim[0];
    retval.pos[1] = 0;
    retval.pos[2] = 0;
    return retval;
  }


  /** Using Morton Z-filling curve. Only for integer less than 256*256*256
  */
  class const_z_iterator {
  public:
    /* equality */
    bool operator==(const const_z_iterator& it) const
    { return lpos == it.lpos; }

    /* not equal */
    bool operator!=(const const_z_iterator& it) const
    { return lpos != it.lpos; }

    /* pre increment */
    const_z_iterator& operator++()
    {
      ++lpos;
      return *this;
    }
    /* post increment */
    const_z_iterator operator++(int)
    {
      const_z_iterator retval = *this;
      ++lpos;
      return retval;
    }
    /* indirection */
    point operator*()
    { 
      /* decode the 3D Morton code */
      int i =       table2[ lpos & 0x3f ]               /* take bit b0, b1 */
            + 4U  * table2[ (lpos & (0x3f << 6))>>6 ]   /* take bit b2, b3 */
            + 16U * table2[ (lpos & (0x3f << 12))>>12 ] /* take bit b4, b5 */
            + 64U * table2[ (lpos & (0x3f << 18))>>18 ];/* take bit b6, b7 */

      int j =       table1[ lpos & 0x3f ]               /* take bit b0, b1 */
            + 4U  * table1[ (lpos & (0x3f << 6))>>6 ]   /* take bit b2, b3 */
            + 16U * table1[ (lpos & (0x3f << 12))>>12 ] /* take bit b4, b5 */
            + 64U * table1[ (lpos & (0x3f << 18))>>18 ];/* take bit b6, b7 */

      int k =       table0[ lpos & 0x3f ]               /* take bit b0, b1 */
            + 4U  * table0[ (lpos & (0x3f << 6))>>6 ]   /* take bit b2, b3 */
            + 16U * table0[ (lpos & (0x3f << 12))>>12 ] /* take bit b4, b5 */
            + 64U * table0[ (lpos & (0x3f << 18))>>18 ];/* take bit b6, b7 */
      return point(i,j,k); 
    }

  protected:
    friend class MeshIndex3D;
  
  private:
    unsigned int lpos; /* linear position */
    static unsigned char table0[];
    static unsigned char table1[];
    static unsigned char table2[];
  };
  
  /**
  */
  const_z_iterator begin_z() const
  { 
    const_z_iterator retval;
    retval.lpos = 0;
    return retval;
  }
  
  /**
  */
  const_z_iterator end_z() const
  {
    const_z_iterator retval;
    retval.lpos = dim[0]*dim[1]*dim[2];
    return retval;
  }

private:
  unsigned short dim[3];
};

#endif

