import numpy as np

import gmsh

def remeshSTL(path, maxSize, minSize, name='object'):
  gmsh.initialize()
  gmsh.model.add(name)
  gmsh.model.mesh.setOrder(1)

  gmsh.option.setNumber("Mesh.CharacteristicLengthMax", maxSize )
  gmsh.option.setNumber("Mesh.CharacteristicLengthMin", minSize )
  gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 10 )
  gmsh.option.setNumber("Mesh.RecombineAll", 0)
  gmsh.option.setNumber("Mesh.Algorithm"  ,  5)

  gmsh.option.setNumber("General.Verbosity", 2)

  gmsh.merge(path)

  gmsh.model.mesh.classifySurfaces(
    angle=np.pi/10, 
    #boundary=True, 
    #forReparametrization=True, 
    #curveAngle=np.pi/10
    )

  gmsh.model.mesh.createGeometry()

  gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
  gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

  gmsh.model.mesh.generate(2)

  nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
  elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=2)

  gmsh.model.geo.synchronize()
  #gmsh.fltk.run()
  gmsh.finalize()

  return nodeTags, nodeCoords, elemTypes, elemTags, elemNodeTags

def poly_area(poly):
    '''
    poly: array of coordinates of vertices

    https://stackoverflow.com/a/68115011
    '''
    #shape (N, 3)
    if isinstance(poly, list):
        poly = np.array(poly)
    #all edges
    edges = poly[1:] - poly[0:1]
    # row wise cross product
    cross_product = np.cross(edges[:-1],edges[1:], axis=1)
    #area of all triangles
    area = np.linalg.norm(cross_product, axis=1)/2
    return sum(area)

if __name__ == "__main__":
  print('I do nothing, I am just a storage of functions.')
