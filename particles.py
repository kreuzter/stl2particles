import numpy as np
import vtk

import auxiliaryFunctions as aux

name = 'sifon'
path = 'data/sifon_dp0-005_geo.stl'

nodeTags, nodeCoords, elemTypes, elemTags, elemNodeTags = aux.remeshSTL(path,
                                                                        50e-3,
                                                                        1e-3,
                                                                        name)

nodeCoords = nodeCoords.reshape((int(len(nodeCoords)/3),3))
elemNums = np.empty(len(elemTypes), dtype=int)

globalElemTags = np.concatenate(elemTags).ravel()
globalElemAreas = np.zeros(len(globalElemTags))
globalElemNormals = np.zeros((len(globalElemTags), 3))

for i in range(len(elemTypes)):
  el_numNodes = len(elemNodeTags[i])/len(elemTags[i])
  if not el_numNodes.is_integer(): 
    print('Something went wrong.')
  else: elemNums[i] = int(el_numNodes)

globalElemNodeTags = np.ones((0, elemNums.max()), dtype=int)

for i in range(len(elemTypes)):
  temp_elemNodeTags = elemNodeTags[i].reshape((
    int(len(elemTags[i])),
     elemNums[i]
     ))
  
  if elemNums[i] < elemNums.max():
    temp_add = -1*np.ones((len(elemTags[i]), elemNums.max() - elemNums[i]))
    temp_elemNodeTags = np.hstack((temp_elemNodeTags, temp_add))
  
  globalElemNodeTags = np.vstack((globalElemNodeTags, temp_elemNodeTags))

connectionMatrix = np.zeros((len(globalElemTags), len(nodeTags)), dtype=int)
for elemIdx in range(len(globalElemTags)):
  for nodeTag in globalElemNodeTags[elemIdx]:
    if nodeTag == -1: pass
    else:
      connectionMatrix[elemIdx, np.where(nodeTags==nodeTag)[0][0]] = True

numberOfNodesOnElement = np.sum(connectionMatrix, axis=1)

for elemIdx in range(len(globalElemTags)):
  elemTag = globalElemTags[elemIdx]
  elemNodesTags = globalElemNodeTags[elemIdx, :]
  temp_index = np.argwhere(elemNodesTags==-1)
  elemNodesTags = np.delete(elemNodesTags, temp_index)
  elemNodesIdxs = [ np.where(elemNode == nodeTags)[0][0] for elemNode in (elemNodesTags)]
  elemNodesCoords = nodeCoords[elemNodesIdxs]
  normal = np.cross( elemNodesCoords[1] - elemNodesCoords[0],
                    elemNodesCoords[2] - elemNodesCoords[0] )  
  globalElemNormals[elemIdx] = normal/np.linalg.norm(normal)
  globalElemAreas[elemIdx] = aux.poly_area(elemNodesCoords)

particleAreas = np.dot(globalElemAreas/numberOfNodesOnElement, connectionMatrix)

print(f'total area of elements:  {np.sum(globalElemAreas)}')
print(f'total area of particles: {np.sum(particleAreas)  }')

normals = np.zeros((len(particleAreas), 3))
for i in [0,1,2]:
  normals[:,i] =  np.dot(
    np.multiply(globalElemAreas, globalElemNormals[:,i]), connectionMatrix
    )

for i in range(len(particleAreas)):
  normals[i,:] = normals[i,:]/np.linalg.norm(normals[i,:])

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
writer.Write()


