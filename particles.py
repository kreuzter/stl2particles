import numpy as np
from scipy.sparse import dok_matrix as sparse_mat

import auxiliaryFunctions as aux

name = 'sifon'
path = 'data/sifon_dp0-005_geo.stl'
writeOutput = False
maxMinSize = np.array([50e-3, 1e-3])
gmshGui=False

old_nodeTags, nodeCoords, _, old_elemTags, old_elemNodeTags = aux.remeshSTL(path,
                                                                maxMinSize.max(),
                                                                maxMinSize.min(),
                                                                name,
                                                                gmshGui)

numParticles = len(old_nodeTags)
numElements  = np.sum([len(el) for el in old_elemTags])

numberOfNodesOnElement = np.ones(len(old_elemTags[0]))*(len(old_elemNodeTags[0])/len(old_elemTags[0]))

for i in range(1,len(old_elemNodeTags)):
  numberOfNodesOnElement = np.append(
    numberOfNodesOnElement,
    np.ones(len(old_elemTags[i]))*(len(old_elemNodeTags[i])/len(old_elemTags[i]))
    )
numberOfNodesOnElement = numberOfNodesOnElement.astype(int)

startOfElemNodesTags = np.cumsum(numberOfNodesOnElement)
startOfElemNodesTags = np.append(0,startOfElemNodesTags[:-1])
endOfElemNodesTags = startOfElemNodesTags+numberOfNodesOnElement

nodeCoords = nodeCoords.reshape((int(len(nodeCoords)/3),3))

globalElemTags = np.concatenate(old_elemTags).ravel()
globalElemNodeTags = np.concatenate(old_elemNodeTags).ravel()

temp_nodeArray = np.zeros(int(old_nodeTags.max()+1), dtype=int)
for tagI in range(len(old_nodeTags)):
  temp_nodeArray[old_nodeTags[tagI]] = tagI

new_globalElemNodeTags = np.zeros(len(globalElemNodeTags), dtype=int)
for nodeTagI in range(len(globalElemNodeTags)):
  new_globalElemNodeTags[nodeTagI] = temp_nodeArray[globalElemNodeTags[nodeTagI]]

globalElemAreas   = np.zeros( numElements    )
globalElemNormals = np.zeros((numElements, 3))

connectionMatrix =  sparse_mat((numParticles, numElements), dtype=bool)

for elemIdx in range(numElements):
  first = startOfElemNodesTags[elemIdx]
  last = endOfElemNodesTags[elemIdx]
  tags = new_globalElemNodeTags[first:last]

  for nodeTag in tags:
    connectionMatrix[ nodeTag, elemIdx] = True

  elemNodesIdxs = tags
  elemNodesCoords = nodeCoords[elemNodesIdxs]
  normal = np.cross( elemNodesCoords[1] - elemNodesCoords[0],
                    elemNodesCoords[2] - elemNodesCoords[0] )  
  globalElemNormals[elemIdx] = normal/np.linalg.norm(normal)
  globalElemAreas[elemIdx] = aux.poly_area(elemNodesCoords)

# From element properties compute particle properties

particleAreas = connectionMatrix.dot(globalElemAreas/numberOfNodesOnElement)

print(f'total area of elements:  {np.sum(globalElemAreas)}')
print(f'total area of particles: {np.sum(particleAreas)  }')

particleNormals = np.zeros((len(particleAreas), 3))
for i in [0,1,2]:
  particleNormals[:,i] = connectionMatrix.dot(np.multiply(globalElemAreas, globalElemNormals[:,i]))

for i in range(len(particleAreas)):
  particleNormals[i,:] = particleNormals[i,:]/np.linalg.norm(particleNormals[i,:])

if writeOutput: aux.writeVTK(name, nodeCoords, particleAreas, particleNormals)
