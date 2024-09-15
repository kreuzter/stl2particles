import numpy as np
import vtk
from scipy.sparse import dok_matrix as sparse_mat

import auxiliaryFunctions as aux

name = 'sifon'
path = 'data/sifon_dp0-005_geo.stl'

nodeTags, nodeCoords, elemTypes, elemTags, elemNodeTags = aux.remeshSTL(path,
                                                                        10e-3,
                                                                         1e-3,
                                                                        name)

numberOfNodesOnElement = np.ones(len(elemTags[0]))*(len(elemNodeTags[0])/len(elemTags[0]))

for i in range(1,len(elemNodeTags)):
  numberOfNodesOnElement = np.append(
    numberOfNodesOnElement,
    np.ones(len(elemTags[i]))*(len(elemNodeTags[i])/len(elemTags[i]))
    )
numberOfNodesOnElement = numberOfNodesOnElement.astype(int)

startOfElemNodesTags = np.cumsum(numberOfNodesOnElement)
startOfElemNodesTags = np.append(0,startOfElemNodesTags[:-1])
endOfElemNodesTags = startOfElemNodesTags+numberOfNodesOnElement

nodeCoords = nodeCoords.reshape((int(len(nodeCoords)/3),3))
globalElemTags = np.concatenate(elemTags).ravel()
globalElemNodeTags = np.concatenate(elemNodeTags).ravel()

globalElemAreas   = np.zeros( len(globalElemTags)    )
globalElemNormals = np.zeros((len(globalElemTags), 3))

connectionMatrix =  sparse_mat((len(nodeTags), len(globalElemTags)), dtype=bool)

# np.zeros((len(globalElemTags), len(nodeTags)), dtype=bool)
for elemIdx in range(len(globalElemTags)):
  first = startOfElemNodesTags[elemIdx]
  last = endOfElemNodesTags[elemIdx]
  tags = globalElemNodeTags[first:last]

  for nodeTag in tags:
    connectionMatrix[ np.where(nodeTags==nodeTag)[0][0], elemIdx] = True

  elemNodesIdxs = [ np.where(elemNode == nodeTags)[0][0] for elemNode in (tags)]
  elemNodesCoords = nodeCoords[elemNodesIdxs]
  normal = np.cross( elemNodesCoords[1] - elemNodesCoords[0],
                    elemNodesCoords[2] - elemNodesCoords[0] )  
  globalElemNormals[elemIdx] = normal/np.linalg.norm(normal)
  globalElemAreas[elemIdx] = aux.poly_area(elemNodesCoords)
# From element properties compute particle properties

particleAreas = connectionMatrix.dot(globalElemAreas/numberOfNodesOnElement)
#np.dot(globalElemAreas/numberOfNodesOnElement, connectionMatrix)

print(f'total area of elements:  {np.sum(globalElemAreas)}')
print(f'total area of particles: {np.sum(particleAreas)  }')

normals = np.zeros((len(particleAreas), 3))
for i in [0,1,2]:
  normals[:,i] = connectionMatrix.dot(np.multiply(globalElemAreas, globalElemNormals[:,i]))


for i in range(len(particleAreas)):
  normals[i,:] = normals[i,:]/np.linalg.norm(normals[i,:])

# Create and write to VTK

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
vNormal.SetName('normals')
vNormal.SetNumberOfTuples(len(particleAreas))

for i in range(len(particleAreas)):
  vNormal.SetTuple3(i, normals[i,0], normals[i,1], normals[i,2])

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
#writer.Write()


