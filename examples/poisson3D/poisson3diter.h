#ifndef _POISSON3DITER_H_
#define _POISSON3DITER_H_
#include "poisson3d.h"

/* iteration over 3D mesh index */
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
      if (++pos[2] < dim[2]) return *this;
      pos[2] = 0;
      if (++pos[1] < dim[1]) return *this;
      pos[1] = 0;
      ++pos[0];
      return *this;
    }
    /* post increment */
    const_iterator operator++(int)
    {
      const_iterator retval = *this;
      if (++pos[2] < dim[2]) return *this;
      pos[2] = 0;
      if (++pos[1] < dim[1]) return *this;
      pos[1] = 0;
      ++pos[0];
      return retval;
    }
    /* indirection */
    Poisson3D::Index operator*()
    { return Poisson3D::Index(pos[0], pos[1], pos[2]); }

  protected:
    friend class MeshIndex3D;
  
  private:
    unsigned short dim[3];
    unsigned short pos[3];
  };

  /**
  */
  const_iterator begin() const
  { 
    const_iterator retval;
    retval.dim[0] = dim[0];
    retval.dim[1] = dim[1];
    retval.dim[2] = dim[2];
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
    retval.dim[0] = dim[0];
    retval.dim[1] = dim[1];
    retval.dim[2] = dim[2];
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
    Poisson3D::Index operator*()
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
      return Poisson3D::Index(i,j,k); 
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