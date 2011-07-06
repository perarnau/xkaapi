#include <GL/gl.h>

extern "C" void glBindBuffer(GLenum, GLuint);
extern "C" void glBufferData(GLenum, GLsizeiptr, const GLvoid *, GLenum);

#include <sofa/component/forcefield/HexahedronFEMForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

using namespace sofa::component::forcefield;
using namespace sofa::defaulttype;
using sofa::helper::vector;

#ifdef SOFA_NEW_HEXA
typedef sofa::core::topology::BaseMeshTopology::Hexa Element;
typedef sofa::core::topology::BaseMeshTopology::SeqHexahedra VecElement;
#else
typedef sofa::core::topology::BaseMeshTopology::Cube Element;
typedef sofa::core::topology::BaseMeshTopology::SeqCubes VecElement;
#endif

static void fu(unsigned int i, Element& elem)
{
  elem[0] = i * 42;
}

int main(int ac, char** av)
{
  VecElement* v;
  unsigned int i = 0;
  typename VecElement::iterator it;

  for (it = v->begin(); it != v->end(); ++it, ++i) fu(i, *it);

  return 0;
}
