import numpy as np

import vtk
import gmsh

def remeshSTL(path, maxSize, minSize, name='object', runGUI=False):
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

  gmsh.model.mesh.classifySurfaces(angle=np.pi/10)

  gmsh.model.mesh.createGeometry()

  gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
  gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

  gmsh.model.mesh.generate(2)

  nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
  elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=2)

  gmsh.model.geo.synchronize()
  if runGUI: gmsh.fltk.run()
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

def writeVTK(name,
             nodeCoords,
             particleAreas,
             particleNormals):
  
  writer = vtk.vtkPolyDataWriter()
  writer.SetFileName('data/'+name+'.vtk')

  vPoints = vtk.vtkPoints()
  vPoints.SetNumberOfPoints(len(particleAreas))
  for i in range(len(particleAreas)):
    vPoints.SetPoint(i, nodeCoords[i,:])

  vpoly = vtk.vtkPolyData()
  vpoly.SetPoints(vPoints)

  vNormal = vtk.vtkFloatArray()
  vNormal.SetNumberOfComponents(3)
  vNormal.SetName('Normals')
  vNormal.SetNumberOfTuples(len(particleAreas))

  for i in range(len(particleAreas)):
    vNormal.SetTuple3(i, particleNormals[i,0], particleNormals[i,1], particleNormals[i,2])

  vpoly.GetPointData().AddArray(vNormal)

  vArea = vtk.vtkFloatArray()
  vArea.SetNumberOfComponents(1)
  vArea.SetName('Areas')
  vArea.SetNumberOfTuples(len(particleAreas))

  for i in range(len(particleAreas)):
    vArea.SetTuple1(i, particleAreas[i])

  vpoly.GetPointData().AddArray(vArea)

  writer.SetInputData(vpoly)
  writer.SetFileTypeToBinary()
  writer.Update()
  writer.Write()

if __name__ == "__main__":
  print('I do nothing, I am just a storage of functions.')
