import numpy as np
from .file_re_mesh import Matrix4x4,AABB,Sphere,CompressedSixWeightIndices

#MESH VERSIONS
VERSION_SF6 = 230110883
VERSION_MHWILDS_BETA = 240820143
VERSION_MHWILDS = 241111606
VERSION_MHS3 = 250604100
VERSION_ONIWOTS = 260209350
VERSION_PRAGDEMO = 250925211
VERSION_RE9 = 250925211

BLEND_SHAPE_MESH_VERSIONS = frozenset((
	VERSION_MHWILDS_BETA,
	VERSION_MHWILDS,
))


def _mesh_supports_blend_shapes(reMesh):
	return int(getattr(reMesh, "meshVersion", 0) or 0) in BLEND_SHAPE_MESH_VERSIONS


SIX_WEIGHT_MESH_VERSIONS = frozenset([
	VERSION_SF6,
	VERSION_MHWILDS_BETA,
	VERSION_MHWILDS,
	VERSION_MHS3,
	VERSION_ONIWOTS,
	#VERSION_PRAGDEMO,
	])

typeNameMapping = ["Position","NorTan","UV","UV2","Weight","Color","SF6UnknownVertexDataType","ExtraWeight"]
typeStrideDict = {
	"Position":12,
	"NorTan":8,
	"UV":4,
	"UV2":4,
	"Weight":16,
	"Color":4,
	"ExtraWeight":16,
	}

def ReadPosBuffer(vertexPosBuffer,tags):
	posList = np.frombuffer(vertexPosBuffer,dtype="<3f").tolist()
	return posList

def _DecodeMHWILDSNormalGroup(encodedValue):
	"""Translate the Wilds Normal.w byte to its editable logical group ID."""
	encodedValue = int(encodedValue) & 0xFF
	if encodedValue == 0:
		return 0
	if encodedValue <= 127:
		return encodedValue
	if encodedValue == 128:
		return 0
	return encodedValue - 1

def ReadNorTanBuffer(norTanBuffer,tags):
	usableSize = (len(norTanBuffer) // 8) * 8
	if usableSize == 0:
		return ([], [], []) if "MHWILDS" in tags else ([], [])
	rawSigned = np.frombuffer(norTanBuffer[:usableSize], dtype="<4b")
	norTanArray = np.divide(np.delete(rawSigned, 3, axis=1), 127)
	norTanList = norTanArray.tolist()
	normalList = norTanList[::2]
	tangentList = norTanList[1::2]
	rawUnsigned = np.frombuffer(
		norTanBuffer[:usableSize], dtype=np.uint8
	).reshape((-1, 4))
	if "MHWILDS" not in tags:
		# Preserve the original two-list return for every non-Wilds game.
		return (normalList, tangentList)
	normalGroupList = [
		_DecodeMHWILDSNormalGroup(value)
		for value in rawUnsigned[::2, 3]
	]
	return (normalList, tangentList, normalGroupList)

def ReadUVBuffer(uvBuffer,tags):
	#uvArray = np.frombuffer(uvBuffer,dtype="<2e",)
	uvArray = np.frombuffer(bytearray(uvBuffer),dtype="<e",)#Convert bytes to bytearray to make numpy array mutable
	#Do (1-x) to v value
	uvArray[1::2] *= -1
	uvArray[1::2] += 1
	uvArray = uvArray.reshape((-1,2))
	return uvArray.tolist()

def ReadWeightBuffer(weightBuffer,tags):
	#weightArray = np.frombuffer(weightBuffer,dtype=f"<{stride//2}B")
	weightArray = np.frombuffer(weightBuffer,dtype="<8B")
	boneIndicesList = weightArray[::2]
	if "SixWeightCompressed" in tags:
		#TODO make decompression faster, this can probably done entirely through numpy with pack and unpackbits
		#Spent too much time trying to get it to work through numpy only, gave up and used ctypes instead
		#Convert byte array to list of uint64 to be passed to bitfield
		
		
		bf = CompressedSixWeightIndices()
		uint64List = np.frombuffer(weightArray[::2].tobytes(),dtype="<Q")
		

		boneIndicesList = [(0,0,0,0,0,0,0,0)]*len(uint64List)
		#print(boneIndicesList)
		for index, value in enumerate(uint64List):
			bf.asUInt64 = value
			boneIndicesList[index] = (bf.weights.w0,bf.weights.w1,bf.weights.w2,bf.weights.w3,bf.weights.w4,bf.weights.w5,0,0)
		#print(boneIndicesArray)
	else:
		boneIndicesList = weightArray[::2].tolist()
	boneWeightsList = np.divide(weightArray[1::2],255).tolist()
	return (boneIndicesList,boneWeightsList)

def ReadColorBuffer(colorBuffer,tags):
	colorArray = np.frombuffer(colorBuffer,dtype="<4B",)
	colorArray = np.divide(colorArray,255)
	return colorArray.tolist()

def ReadFaceBuffer(faceBuffer):
	faceBuffer = np.frombuffer(faceBuffer,dtype="<3H",)
	return faceBuffer.tolist()
def ReadIntFaceBuffer(faceBuffer):
	faceBuffer = np.frombuffer(faceBuffer,dtype="<3I",)
	return faceBuffer.tolist()
def readPackedBitsVec3Array(packedIntArray, numBits):
	#for packedInt in packedIntArray:
		#print(packedInt)
	limit = 2**numBits-1
	vec3Array = np.zeros(len(packedIntArray),np.dtype("<3f"))
	vec3Array[:,0] = ((packedIntArray >> 0) & limit) / limit
	vec3Array[:,1] = ((packedIntArray >> (numBits*1)) & limit) / limit
	vec3Array[:,2] = ((packedIntArray >> (numBits*2)) & limit) / limit
	#for val in vec3Array:
		#print(val)
	return vec3Array

#MPLY

def ReadNorBuffer(norBuffer,tags):
	norArray = np.frombuffer(norBuffer,dtype="<4b")
	norArray = np.delete(norArray, 3, axis=1)
	return (norArray.tolist())

def ReadCompressedPosBuffer(vertexPosBuffer,bitFlag,center,relOffset):
	
	
	if bitFlag.flags.use24BitPos:
		#print("DEBUG 24 bit pos")
		byte3Array = np.frombuffer(vertexPosBuffer,dtype="<3b")
		posArray = np.zeros(len(byte3Array),np.dtype("<3f"))
		posArray = byte3Array * (1.0/255)
		
		#posArray[:,0] = byte3Array[:,0] * (1.0/255)
		#posArray[:,1] = byte3Array[:,1] * (1.0/255)
		#posArray[:,2] = byte3Array[:,2] * (1.0/255)
		posArray[:] = 0#TODO FIX 24 bit import
	elif bitFlag.flags.use32BitPos:
		#print("DEBUG 32 bit pos")
		packedIntArray = np.frombuffer(vertexPosBuffer,dtype="<I")
		posArray = np.zeros(len(packedIntArray),np.dtype("<3f"))
		
		#11-10-11 packing
		"""
		posArray[:,0] = (packedIntArray & 2047) / 2047
		posArray[:,1] = ((packedIntArray >> 11) & 1023) / 1023
		posArray[:,2] = ((packedIntArray >> 21) & 2047) / 2047
		"""
		#10-10-10 packing
		posArray[:,0] = (packedIntArray & 1023)
		posArray[:,1] = ((packedIntArray >> 10) & 1023)
		posArray[:,2] = ((packedIntArray >> 20) & 1023)
	else:
		posArray = np.frombuffer(vertexPosBuffer,dtype="<3H")
		posArray = posArray.astype(dtype="f")
		posArray /= 65535.0
		
	#Thank you shadowcookie for figuring this out
	num = bitFlag.asUInt32
	divByte = (num >> 24) & 0xFF
	multByte = (num >> 16) & 0xFF
	divShift = (divByte - 127)
	scale = (1 << divShift) if divShift >= 0 else (1.0 / (1 << -divShift))
	offset = 1 << (multByte - divByte)
	
	posArray = (posArray - 0.5 + relOffset * offset) * scale + center
	#posArray
	
	return posArray.tolist()

BufferReadDict = {
	"Position":ReadPosBuffer,
	"NorTan":ReadNorTanBuffer,
	"UV":ReadUVBuffer,
	"UV2":ReadUVBuffer,
	"Weight":ReadWeightBuffer,
	"Color":ReadColorBuffer,
	"SF6UnknownVertexDataType":ReadColorBuffer,#Read as color data for now until what it is can be determined
	"ExtraWeight":ReadWeightBuffer,
	}

def ReadBlendShapeType7Buffer(blendShapeBuffer,tags):
	"""Decode Wilds' center-biased packed 11/10/11 delta rows."""
	usableSize = (len(blendShapeBuffer) // 4) * 4
	packed = np.frombuffer(blendShapeBuffer[:usableSize], dtype="<I")
	result = np.zeros((len(packed), 3), dtype=np.float32)
	if len(packed) == 0:
		return result
	result[:, 0] = ((packed & 0x7FF).astype(np.float32) - 1023.0) / 1023.0
	result[:, 1] = (
		((packed >> 11) & 0x3FF).astype(np.float32) - 511.0
	) / 511.0
	result[:, 2] = (
		((packed >> 21) & 0x7FF).astype(np.float32) - 1023.0
	) / 1023.0
	result = np.clip(result, -1.0, 1.0)
	result[packed == 0x7FEFFBFF, :] = 0.0
	return result

def ApplyBlendShapeAABBToDecodedRows(
	decodedRows, aabbEntry, blendShapeTyping, useSymmetricScale=False
):
	decodedRows = np.asarray(decodedRows, dtype=np.float32)
	if decodedRows.size == 0:
		return decodedRows.reshape((-1, 3))
	decodedRows = decodedRows.reshape((-1, 3)).copy()
	if useSymmetricScale or blendShapeTyping == 7:
		scale = np.array((
			max(abs(float(aabbEntry.min.x)), abs(float(aabbEntry.max.x)), 0.000001),
			max(abs(float(aabbEntry.min.y)), abs(float(aabbEntry.max.y)), 0.000001),
			max(abs(float(aabbEntry.min.z)), abs(float(aabbEntry.max.z)), 0.000001),
		), dtype=np.float32)
		decodedRows *= scale
	else:
		minimum = np.array(
			(aabbEntry.min.x, aabbEntry.min.y, aabbEntry.min.z),
			dtype=np.float32,
		)
		maximum = np.array(
			(aabbEntry.max.x, aabbEntry.max.y, aabbEntry.max.z),
			dtype=np.float32,
		)
		decodedRows = minimum + decodedRows * (maximum - minimum)
	return decodedRows

def ReadVertexElementBuffers(vertexElementList,vertexBuffer,tagSet):
	vertexDict = {
		"Position":None,
		"NorTan":None,
		"UV":None,
		"UV2":None,
		"Weight":None,
		"Color":None,
		"SF6UnknownVertexDataType":None,
		"ExtraWeight":None,
		"SecondaryWeight":None,
		}
	lastIndex = len(vertexElementList)-1
	importedElementsSet = set()
	for index,vertexElement in enumerate(vertexElementList):
		if index == lastIndex:
			#bufferEnd = len(vertexBuffer)
			bufferEnd = vertexElementList[index].posStartOffset + (vertexElement.stride * len(vertexDict["Position"]))
		else:
			bufferEnd = vertexElementList[index+1].posStartOffset
		elementName = typeNameMapping[vertexElement.typing]
		#print(f"{elementName} start {str(vertexElement.posStartOffset)} end {str(bufferEnd)} size {str(bufferEnd-vertexElement.posStartOffset)}")
		#print(elementName)

		if elementName not in importedElementsSet:#Prevent importing of doubled vertex element entries present on some meshes
			#I suspect the doubled vertex elements are for when the shadow meshes have their own unique LODs
			#TODO Make shadowVertexDict
			if "shadowLOD" not in tagSet:#Skip reading the first vertex element entry of a type if reading unique shadow LOD
				vertexDict[elementName] = BufferReadDict[elementName](vertexBuffer[vertexElement.posStartOffset:bufferEnd],tagSet)
			importedElementsSet.add(elementName)
		elif "shadowLOD" in tagSet:
			vertexDict[elementName] = BufferReadDict[elementName](vertexBuffer[vertexElement.posStartOffset:bufferEnd],tagSet)
	return vertexDict

class VisconGroup:
	def __init__(self):
		self.visconGroupNum = 0
		self.subMeshList = []
	
class LODLevel:
	def __init__(self):
		self.visconGroupList = []
		self.lodDistance = 0.0

class SubMesh:
	def __init__(self):
		self.vertexPosList = []
		self.faceList = []
		self.normalList = []
		self.tangentList = []
		self.normalGroupList = []
		self.normalPivotGroupList = []
		self.normalPivot0List = []
		self.normalPivot255List = []
		self.uvList = []
		self.uv2List = []
		self.faceList = []
		self.weightList = []
		self.weightIndicesList = []
		#MH Wilds extra weights
		self.extraWeightIndicesList = []
		self.extraWeightList = []
		self.colorList = []
		self.materialIndex = 0
		self.meshVertexOffset = 0#Used for determining mesh reuse
		self.isReusedMesh = False
		self.linkedSubMesh = None
		self.subMeshIndex = 0
		self.blendShapeList = []
		#DD2 shape key weights
		self.secondaryWeightList = []
		self.secondaryWeightIndicesList = []
		#MPLY
		self.relPos = None
		self.boundingBox = None
		self.boundingBoxCenter = None
class ParsedBone:
	def __init__(self):
		self.boneName = "BONE"
		self.boneIndex = 0
		self.parentIndex = 0
		self.nextSiblingIndex = 0
		self.nextChildIndex = 0
		self.symmetryBoneIndex = 0
		self.useSecondaryWeight = 0
		self.worldMatrix = Matrix4x4()
		self.localMatrix = Matrix4x4()
		self.inverseMatrix = Matrix4x4()
		self.boundingBox = None#Bounding box of weighted vertices
class Skeleton:
	def __init__(self):
		self.weightedBones = []
		self.boneList = []

class BlendShape:
	def __init__(self):
		self.blendShapeName = "newBlendShape"
		self.deltas = []

def _parseLODStructureWithoutBlendShapes(
	reMesh,
	targetLODList,
	vertexDictList,
	faceBufferList,
	usedVertexOffsetDictList,
):
	"""Parse ordinary mesh geometry without invoking a game blendshape handler."""
	lodList = []
	for lodIndex, lodGroup in enumerate(targetLODList):
		lod = LODLevel()
		lod.lodDistance = lodGroup.distance
		for visconGroup in lodGroup.meshGroupList:
			group = VisconGroup()
			group.visconGroupNum = visconGroup.visconGroupID
			lastSubmeshIndex = len(visconGroup.vertexInfoList) - 1
			for index, meshInfo in enumerate(visconGroup.vertexInfoList):
				if index == lastSubmeshIndex:
					bufferEnd = (
						visconGroup.vertexInfoList[0].vertexStartIndex
						+ visconGroup.vertexCount
					)
				else:
					bufferEnd = visconGroup.vertexInfoList[index + 1].vertexStartIndex
				submesh = SubMesh()
				submesh.materialIndex = meshInfo.materialIndex
				submesh.subMeshIndex = index
				if (
					meshInfo.vertexStartIndex
					in usedVertexOffsetDictList[meshInfo.vertexBufferIndex]
				):
					submesh.isReusedMesh = True
					submesh.linkedSubMesh = usedVertexOffsetDictList[
						meshInfo.vertexBufferIndex
					][meshInfo.vertexStartIndex]
				else:
					usedVertexOffsetDictList[meshInfo.vertexBufferIndex][
						meshInfo.vertexStartIndex
					] = submesh
				submesh.meshVertexOffset = meshInfo.vertexStartIndex

				vertexDict = vertexDictList[meshInfo.vertexBufferIndex]
				start = meshInfo.vertexStartIndex
				if vertexDict["Position"] is not None:
					submesh.vertexPosList = vertexDict["Position"][start:bufferEnd]
				faceBytes = faceBufferList[meshInfo.vertexBufferIndex]
				if reMesh.lodHeader.has32BitIndexBuffer:
					submesh.faceList = ReadIntFaceBuffer(
						faceBytes[
							meshInfo.faceStartIndex * 4:
							meshInfo.faceStartIndex * 4 + meshInfo.faceCount * 4
						]
					)
				else:
					submesh.faceList = ReadFaceBuffer(
						faceBytes[
							meshInfo.faceStartIndex * 2:
							meshInfo.faceStartIndex * 2 + meshInfo.faceCount * 2
						]
					)
				if vertexDict["NorTan"] is not None:
					submesh.normalList = vertexDict["NorTan"][0][start:bufferEnd]
					submesh.tangentList = vertexDict["NorTan"][1][start:bufferEnd]
				if vertexDict["UV"] is not None:
					submesh.uvList = vertexDict["UV"][start:bufferEnd]
				if vertexDict["UV2"] is not None:
					submesh.uv2List = vertexDict["UV2"][start:bufferEnd]
				if vertexDict["Weight"] is not None:
					submesh.weightIndicesList = vertexDict["Weight"][0][start:bufferEnd]
					submesh.weightList = vertexDict["Weight"][1][start:bufferEnd]
				if vertexDict["ExtraWeight"] is not None:
					submesh.extraWeightIndicesList = vertexDict["ExtraWeight"][0][start:bufferEnd]
					submesh.extraWeightList = vertexDict["ExtraWeight"][1][start:bufferEnd]
				if vertexDict["Color"] is not None:
					submesh.colorList = vertexDict["Color"][start:bufferEnd]
				if vertexDict["SecondaryWeight"] is not None:
					submesh.secondaryWeightIndicesList = vertexDict["SecondaryWeight"][0][start:bufferEnd]
					submesh.secondaryWeightList = vertexDict["SecondaryWeight"][1][start:bufferEnd]
				group.subMeshList.append(submesh)
			lod.visconGroupList.append(group)
		lodList.append(lod)
	return lodList


def _parseLODStructureWithBlendShapes(
	reMesh,
	targetLODList,
	vertexDictList,
	faceBufferList,
	usedVertexOffsetDictList,
	blendShapeBufferList=None,
):
	"""Parse LOD-local blendshape resources without merging vertex namespaces."""
	if not _mesh_supports_blend_shapes(reMesh):
		return _parseLODStructureWithoutBlendShapes(
			reMesh,
			targetLODList,
			vertexDictList,
			faceBufferList,
			usedVertexOffsetDictList,
		)
	lodList = []
	useSymmetricScale = True
	for lodIndex, lodGroup in enumerate(targetLODList):
		blendShapeDict = {}
		usedBlendGroupIndices = set()
		blendData = None
		if (
			reMesh.blendShapeHeader is not None
			and lodIndex < len(reMesh.blendShapeHeader.blendShapeList)
		):
			blendData = reMesh.blendShapeHeader.blendShapeList[lodIndex]

		if blendData is not None:
			# Current blendshape parsing is enabled only for Wilds.
			reader = ReadBlendShapeType7Buffer
			vertexBufferIndex = 0
			targets = list(getattr(blendData, "blendTargetList", []) or [])
			if targets:
				entries = list(
					getattr(targets[0], "subMeshEntryList", []) or []
				)
				if entries:
					targetStart = entries[0].subMeshVertexStartIndex
					for meshGroup in lodGroup.meshGroupList:
						for meshInfo in meshGroup.vertexInfoList:
							if meshInfo.vertexStartIndex == targetStart:
								vertexBufferIndex = meshInfo.vertexBufferIndex
								break
			buffer = b""
			if (
				blendShapeBufferList is not None
				and vertexBufferIndex < len(blendShapeBufferList)
			):
				buffer = blendShapeBufferList[vertexBufferIndex]
			decoded = (
				reader(buffer, set())
				if reader is not None
				else np.zeros((0, 3), dtype=np.float32)
			)
			activeCount = min(
				int(getattr(blendData, "targetCount", 0) or 0),
				len(targets),
				len(getattr(blendData, "aabbList", []) or []),
			)
			def mergedIntervals(intervals):
				merged = []
				for start, end in sorted(intervals):
					if merged and start <= merged[-1][1]:
						merged[-1] = (
							merged[-1][0],
							max(merged[-1][1], end),
						)
					else:
						merged.append((start, end))
				return tuple(merged)

			groupIntervals = []
			for meshGroup in lodGroup.meshGroupList:
				meshInfos = list(meshGroup.vertexInfoList)
				intervals = []
				for infoIndex, meshInfo in enumerate(meshInfos):
					start = int(meshInfo.vertexStartIndex)
					end = (
						int(meshInfos[infoIndex + 1].vertexStartIndex)
						if infoIndex + 1 < len(meshInfos)
						else int(meshInfos[0].vertexStartIndex)
						+ int(meshGroup.vertexCount)
					)
					intervals.append((start, end))
				groupIntervals.append(mergedIntervals(intervals))
			targetGroupMap = {}
			claimedGroups = set()
			for targetRecordIndex, targetRecord in enumerate(targets):
				targetEntries = list(
					getattr(targetRecord, "subMeshEntryList", []) or []
				)
				targetIntervals = mergedIntervals([
					(
						int(entry.subMeshVertexStartIndex),
						int(entry.subMeshVertexStartIndex)
						+ int(entry.vertCount),
					)
					for entry in targetEntries
				])
				exactCandidates = [
					index
					for index, intervals in enumerate(groupIntervals)
					if intervals == targetIntervals
				]
				available = [
					index
					for index in exactCandidates
					if index not in claimedGroups
				]
				if available:
					targetGroupMap[targetRecordIndex] = available[0]
					claimedGroups.add(available[0])
			payloadCursor = 0
			for targetIndex, target in enumerate(targets[:activeCount]):
				entries = list(
					getattr(target, "subMeshEntryList", []) or []
				)
				targetGroupIndex = None
				if targetIndex in targetGroupMap:
					targetGroupIndex = targetGroupMap[targetIndex]
					usedBlendGroupIndices.add(targetGroupIndex)
				elif entries:
					targetStarts = {
						int(entry.subMeshVertexStartIndex)
						for entry in entries
					}
					candidates = []
					for groupIndex, meshGroup in enumerate(
						lodGroup.meshGroupList
					):
						groupStarts = {
							int(meshInfo.vertexStartIndex)
							for meshInfo in meshGroup.vertexInfoList
						}
						if targetStarts.issubset(groupStarts):
							candidates.append(groupIndex)
					unusedCandidates = [
						index
						for index in candidates
						if index not in usedBlendGroupIndices
					]
					if unusedCandidates:
						targetGroupIndex = unusedCandidates[0]
					elif candidates:
						targetGroupIndex = candidates[0]
					if targetGroupIndex is not None:
						usedBlendGroupIndices.add(targetGroupIndex)
				channelCount = int(
					getattr(target, "blendShapeNum", 0) or 0
				)
				if channelCount <= 0:
					continue
				if entries:
					targetRowCount = sum(
						int(entry.vertCount) for entry in entries
					)
				else:
					targetRowCount = int(target.vertCount)
				useExplicitOffsets = any(
					int(getattr(entry, "vertOffset", 0) or 0) != 0
					for entry in entries
				)
				for channelIndex in range(channelCount):
					remapIndex = (
						int(getattr(target, "blendSSIndex", 0) or 0)
						+ channelIndex
					)
					if remapIndex >= len(reMesh.blendShapeNameRemapList):
						continue
					rawNameIndex = reMesh.blendShapeNameRemapList[remapIndex]
					if rawNameIndex >= len(reMesh.rawNameList):
						continue
					shapeName = reMesh.rawNameList[rawNameIndex]
					if entries:
						running = 0
						for entry in entries:
							start = (
								int(entry.vertOffset)
								+ channelIndex * targetRowCount
								if useExplicitOffsets
								else payloadCursor
								+ channelIndex * targetRowCount
								+ running
							)
							count = int(entry.vertCount)
							deltas = decoded[start:start + count].copy()
							deltas = ApplyBlendShapeAABBToDecodedRows(
								deltas,
								blendData.aabbList[targetIndex],
								blendData.typing,
								useSymmetricScale,
							)
							shape = BlendShape()
							shape.blendShapeName = shapeName
							shape.deltas = deltas
							blendShapeDict.setdefault((
								targetGroupIndex,
								int(entry.subMeshVertexStartIndex),
							), []).append(shape)
							running += count
					else:
						start = payloadCursor + channelIndex * targetRowCount
						shape = BlendShape()
						shape.blendShapeName = shapeName
						shape.deltas = ApplyBlendShapeAABBToDecodedRows(
							decoded[start:start + targetRowCount].copy(),
							blendData.aabbList[targetIndex],
							blendData.typing,
							useSymmetricScale,
						)
						blendShapeDict.setdefault((
							targetGroupIndex,
							int(target.subMeshVertexStartIndex),
						), []).append(shape)
				payloadCursor += targetRowCount * channelCount

		lod = LODLevel()
		lod.lodDistance = lodGroup.distance
		for meshGroupIndex, meshGroup in enumerate(lodGroup.meshGroupList):
			group = VisconGroup()
			group.visconGroupNum = meshGroup.visconGroupID
			meshInfos = list(meshGroup.vertexInfoList)
			for index, meshInfo in enumerate(meshInfos):
				faceBytes = faceBufferList[meshInfo.vertexBufferIndex]
				if reMesh.lodHeader.has32BitIndexBuffer:
					parsedFaces = ReadIntFaceBuffer(
						faceBytes[
							meshInfo.faceStartIndex * 4:
							(meshInfo.faceStartIndex + meshInfo.faceCount) * 4
						]
					)
				else:
					parsedFaces = ReadFaceBuffer(
						faceBytes[
							meshInfo.faceStartIndex * 2:
							(meshInfo.faceStartIndex + meshInfo.faceCount) * 2
						]
					)
				inferredVertexCount = (
					int(np.max(parsedFaces)) + 1
					if len(parsedFaces)
					else 0
				)
				bufferEnd = (
					int(meshInfo.vertexStartIndex) + inferredVertexCount
				)
				submesh = SubMesh()
				submesh.materialIndex = meshInfo.materialIndex
				submesh.subMeshIndex = index
				offsetMap = usedVertexOffsetDictList[
					meshInfo.vertexBufferIndex
				]
				if meshInfo.vertexStartIndex in offsetMap:
					submesh.isReusedMesh = True
					submesh.linkedSubMesh = offsetMap[
						meshInfo.vertexStartIndex
					]
				else:
					offsetMap[meshInfo.vertexStartIndex] = submesh
				submesh.meshVertexOffset = meshInfo.vertexStartIndex

				vertexDict = vertexDictList[meshInfo.vertexBufferIndex]
				start = meshInfo.vertexStartIndex
				if vertexDict["Position"] is not None:
					submesh.vertexPosList = vertexDict["Position"][
						start:bufferEnd
					]
				submesh.faceList = parsedFaces
				if vertexDict["NorTan"] is not None:
					submesh.normalList = vertexDict["NorTan"][0][start:bufferEnd]
					submesh.tangentList = vertexDict["NorTan"][1][start:bufferEnd]
					if len(vertexDict["NorTan"]) > 2:
						submesh.normalGroupList = vertexDict["NorTan"][2][
							start:bufferEnd
						]
						count = len(submesh.normalGroupList)
						submesh.normalPivotGroupList = [0] * count
						submesh.normalPivot0List = [0] * count
						submesh.normalPivot255List = [0] * count
				for field, destination in (
					("UV", "uvList"),
					("UV2", "uv2List"),
					("Color", "colorList"),
				):
					if vertexDict[field] is not None:
						setattr(
							submesh,
							destination,
							vertexDict[field][start:bufferEnd],
						)
				for field, indexDestination, weightDestination in (
					("Weight", "weightIndicesList", "weightList"),
					(
						"ExtraWeight",
						"extraWeightIndicesList",
						"extraWeightList",
					),
					(
						"SecondaryWeight",
						"secondaryWeightIndicesList",
						"secondaryWeightList",
					),
				):
					if vertexDict[field] is not None:
						setattr(
							submesh,
							indexDestination,
							vertexDict[field][0][start:bufferEnd],
						)
						setattr(
							submesh,
							weightDestination,
							vertexDict[field][1][start:bufferEnd],
						)
				submesh.blendShapeList.extend(blendShapeDict.get(
					(meshGroupIndex, int(meshInfo.vertexStartIndex)),
					blendShapeDict.get(
						(None, int(meshInfo.vertexStartIndex)), []
					),
				))
				group.subMeshList.append(submesh)
			lod.visconGroupList.append(group)
		lodList.append(lod)
	return lodList


# Current blendshape import dispatch is exclusive to Monster Hunter Wilds.
def parseLODStructure(
	reMesh,
	targetLODList,
	vertexDictList,
	faceBufferList,
	usedVertexOffsetDictList,
	blendShapeBufferList=None,
):
	if (
		blendShapeBufferList is None
		or not _mesh_supports_blend_shapes(reMesh)
	):
		return _parseLODStructureWithoutBlendShapes(
			reMesh,
			targetLODList,
			vertexDictList,
			faceBufferList,
			usedVertexOffsetDictList,
		)
	return _parseLODStructureWithBlendShapes(
		reMesh,
		targetLODList,
		vertexDictList,
		faceBufferList,
		usedVertexOffsetDictList,
		blendShapeBufferList,
	)


def debug_Generate010StreamingTemplate(templateLODList):
	#Yes this is driving me insane to the point to where I'm generating an 010 template to check if the buffers are being read correctly
	print("//Auto generated streaming mesh template")
	print("""
typedef struct
{
    float x;
    float y;
    float z;
}Position<bgcolor=0x0000FF>;

typedef struct
{
    ubyte normal[4];
    ubyte tangent[4];
}NorTan<bgcolor=0x00FF00>;

typedef struct
{
    hfloat u;
    hfloat v;
}UV<bgcolor=0xFF0000>;

typedef struct
{
    hfloat u;
    hfloat v;
}UV2<bgcolor=0xCC0000>;
typedef struct
{
    uint64 w0:10;;
    uint64 w1:10;
    uint64 w2:10;
    uint64 pad0:2;
    uint64 w3:10;
    uint64 w4:10;
    uint64 w5:10;
    uint64 pad1:2;
	ubyte indices[8];
}Weight<bgcolor=0x00FFFF>;

typedef struct
{
    ubyte r;
    ubyte g;
    ubyte b;
    ubyte a;
}Color<bgcolor=0xFFFF00>;

	""")
	for index, elementList in enumerate(templateLODList):
		print("struct")
		print("{")
		for element in elementList:
			print("\tFSeek("+str(element["start"])+");\n\t struct\n\t{")
			print("\t\t"+element["type"] + " entry["+str((element["end"]-element["start"])//element["stride"])+"];")
			print("\t}element;")
		print("}LOD"+str(index)+";")
		
	print("//EOF")


def _ResolveMHWILDSLOD0NormalPivotIdentities(reMesh, lodList):
	"""Map the 256 binary pivot positions onto editable LOD0 point metadata."""
	if reMesh.meshVersion not in (VERSION_MHWILDS_BETA, VERSION_MHWILDS):
		return None
	if not lodList or reMesh.floatsHeader is None:
		return None
	pivotValues = reMesh.floatsHeader.unknDataList
	pivotList = [] if pivotValues is None else list(pivotValues)
	if len(pivotList) < 256:
		return None

	groupCandidates = {}
	allCandidates = []
	globalRow = 0
	for viscon in lodList[0].visconGroupList:
		for submesh in viscon.subMeshList:
			if bool(getattr(submesh, "isReusedMesh", False)):
				continue
			groupValues = submesh.normalGroupList
			positionValues = submesh.vertexPosList
			groups = [] if groupValues is None else list(groupValues)
			positions = (
				[] if positionValues is None else list(positionValues)
			)
			if len(groups) != len(positions):
				globalRow += len(positions)
				continue
			count = len(groups)
			submesh.normalPivotGroupList = [0] * count
			submesh.normalPivot0List = [0] * count
			submesh.normalPivot255List = [0] * count
			for localIndex, (group, position) in enumerate(
				zip(groups, positions)
			):
				point = np.asarray(position, dtype=np.float32).reshape(3)
				candidate = (
					submesh,
					localIndex,
					point,
					int(group),
					globalRow + localIndex,
				)
				allCandidates.append(candidate)
				if 1 <= int(group) <= 254:
					groupCandidates.setdefault(int(group), []).append(
						candidate
					)
			globalRow += count
	if not allCandidates:
		return None

	def selectCandidate(pivot, candidates):
		pivot = np.asarray(pivot, dtype=np.float32).reshape(3)
		packed = pivot.tobytes()
		for candidate in candidates:
			if candidate[2].tobytes() == packed:
				return candidate, True
		points = np.asarray(
			[candidate[2] for candidate in candidates], dtype=np.float64
		)
		distanceSquared = np.sum(
			(points - pivot.astype(np.float64)[None, :]) ** 2,
			axis=1,
		)
		return candidates[int(np.argmin(distanceSquared))], False

	exactMatches = 0
	for group, candidates in sorted(groupCandidates.items()):
		entry = pivotList[group]
		selected, exact = selectCandidate(
			(entry.x, entry.y, entry.z), candidates
		)
		selected[0].normalPivotGroupList[selected[1]] = group
		exactMatches += int(exact)

	special = {}
	for slot, attributeName in (
		(0, "normalPivot0List"),
		(255, "normalPivot255List"),
	):
		entry = pivotList[slot]
		selected, exact = selectCandidate(
			(entry.x, entry.y, entry.z), allCandidates
		)
		getattr(selected[0], attributeName)[selected[1]] = 1
		special[slot] = {
			"globalRow": int(selected[4]),
			"exact": bool(exact),
		}
	return {
		"usedGroups": len(groupCandidates),
		"exactPivotMatches": exactMatches,
		"specialSlots": special,
	}


class ParsedREMesh:
	def __init__(self):
		self.isMPLY = False
		self.skeleton = None
		self.mainMeshLODList = []
		#self.shadowMeshLODList = []#Commented out because shadow meshes can only reuse lods from main mesh
		self.shadowMeshLinkedLODList = []#
		self.occlusionMeshLODList = []
		self.nameList = []
		self.boneNameRemapList = []
		self.materialNameList = []
		self.boundingSphere = Sphere()
		self.boundingBox = AABB()
		self.bufferHasPosition = False
		self.bufferHasNorTan = False
		self.bufferHasUV = False
		self.bufferHasUV2 = False
		self.bufferHasWeight = False
		self.bufferHasColor = False
		self.bufferHasIntFaces = False
		self.bufferHasExtraWeight = False#Doubled weight buffer, used in MH Wilds
		self.bufferHasSecondaryWeight = False#DD2 shapekeys
		self.mhwildsCanonicalTableProfile = []
	
	def ParseREMesh(self,reMesh,importOptions = {"importAllLOD":True,"importShadowMesh":True,"importOcclusionMesh":True,"importBlendShapes":True}):
		
		blendShapeImportEnabled = (
			_mesh_supports_blend_shapes(reMesh)
			and bool(importOptions.get("importBlendShapes", True))
		)
		self.isMPLY = reMesh.isMPLY
		usedVertexOffsetDictList = []
		lodOffsetDict = dict()#Used for linking shadow mesh lods to main mesh lods
		self.nameList = reMesh.rawNameList
		self.boneNameRemapList = reMesh.boneNameRemapList
		self.materialRemapList = reMesh.materialNameRemapList
		#Parse Skeleton
		for remapIndex in reMesh.materialNameRemapList:
			self.materialNameList.append(reMesh.rawNameList[remapIndex])
		if reMesh.skeletonHeader != None:
			self.skeleton = Skeleton()
			self.skeleton.weightedBones = []
			
			
			
			
			for remapIndex in reMesh.skeletonHeader.boneRemapList:
				self.skeleton.weightedBones.append(reMesh.rawNameList[reMesh.boneNameRemapList[remapIndex]])
				
			#I hope this doesn't cause weird issues somewhere, the root bone isn't counted in the remap table but it is used by some meshes and that causes issues
			#EX F:\RE2RT_EXTRACT\re_chunk_000\natives\STM\ObjectRoot\SetModel\sm4x_Gimmick\sm42\sm42_253_Switch01A\sm42_253_Switch01A_00md.mesh.2109108288
			#Check if the root bone is weighted and add it to the weighted bone list
			
			#Update: turns out this was right but the root bone is supposed to be the last bone in the list, not the first
			if reMesh.boneBoundingBoxHeader != None and reMesh.skeletonHeader.remapCount != reMesh.boneBoundingBoxHeader.count:
				self.skeleton.weightedBones.append(reMesh.rawNameList[reMesh.boneNameRemapList[0]])
			weightedBoneIndex = 0
			for i in range(reMesh.skeletonHeader.boneCount):
				#print(i)
				bone = ParsedBone()
				bone.boneName = reMesh.rawNameList[reMesh.boneNameRemapList[i]]
				bone.boneIndex = i
				bone.parentIndex = reMesh.skeletonHeader.boneInfoList[i].boneParent
				bone.nextSiblingIndex = reMesh.skeletonHeader.boneInfoList[i].boneSibling
				bone.nextChildIndex = reMesh.skeletonHeader.boneInfoList[i].boneChild
				bone.symmetryBoneIndex = reMesh.skeletonHeader.boneInfoList[i].boneSymmetric
				bone.useSecondaryWeight = reMesh.skeletonHeader.boneInfoList[i].useSecondaryWeight
				bone.worldMatrix = reMesh.skeletonHeader.worldMatList[i]
				bone.localMatrix = reMesh.skeletonHeader.localMatList[i]
				bone.inverseMatrix = reMesh.skeletonHeader.inverseMatList[i]
				
				if bone.boneName in self.skeleton.weightedBones:
					try:
						bone.boundingBox = reMesh.boneBoundingBoxHeader.bboxList[weightedBoneIndex]
						weightedBoneIndex += 1
					except:
						print("WARNING: Missing bone bounding box, likely incorrectly exported mesh mod")
				self.skeleton.boneList.append(bone)
			
		#Parse Vertex Buffer
		if reMesh.meshBufferHeader != None:
			tags = set()
			if reMesh.meshVersion in SIX_WEIGHT_MESH_VERSIONS or reMesh.fileHeader.version == 250707828:#Street Fighter 6 mesh version + MH Wilds, #Pragmata internal mesh version uses 6 weight but RE9 uses 8
				tags.add("SixWeightCompressed")#Add tag to parse compressed weights
			if reMesh.meshVersion in BLEND_SHAPE_MESH_VERSIONS:
				tags.add("MHWILDS")
			#if duplicate in vertexelementlist, add shadowLOD tag
			
			
			
			vertexDictList = []
			faceBufferList = []
			blendShapeBufferList = [] if blendShapeImportEnabled else None
			
			vertexDictList.append(ReadVertexElementBuffers(reMesh.meshBufferHeader.vertexElementList,  reMesh.meshBufferHeader.vertexBuffer,tags))
			faceBufferList.append(reMesh.meshBufferHeader.faceBuffer)
			if blendShapeImportEnabled:
				if (
					len(vertexDictList[-1]["Position"]) > 0
					and len(reMesh.meshBufferHeader.vertexElementList) > 0
				):
					vertexCount = len(vertexDictList[-1]["Position"])
					lastElement = reMesh.meshBufferHeader.vertexElementList[-1]
					fallbackStart = (
						lastElement.posStartOffset
						+ vertexCount * lastElement.stride
					)
					headerStart = (
						int(reMesh.meshBufferHeader.sunbreakSecondUnknown) >> 32
					) & 0xFFFFFFFF
					blendShapeStartPos = (
						headerStart
						if fallbackStart <= headerStart
						< len(reMesh.meshBufferHeader.vertexBuffer)
						else fallbackStart
					)
					blendShapeBufferList.append(
						reMesh.meshBufferHeader.vertexBuffer[blendShapeStartPos:]
					)
				else:
					blendShapeBufferList.append(b"")
			
			if reMesh.meshBufferHeader.secondaryWeightBuffer != None:
				self.bufferHasSecondaryWeight = True
				vertexDictList[-1]["SecondaryWeight"] = ReadWeightBuffer(reMesh.meshBufferHeader.secondaryWeightBuffer, tags = set())
			
			if reMesh.streamingInfoHeader != None and reMesh.streamingInfoHeader.entryCount != 0 and reMesh.streamingBuffer != None:
				for entry in reMesh.meshBufferHeader.streamingBufferHeaderList:
					vertexDictList.append(ReadVertexElementBuffers(entry.vertexElementList,entry.vertexBuffer,tags))
					faceBufferList.append(entry.faceBuffer)
					if blendShapeImportEnabled:
						blendShapeStartPos = int(getattr(entry, "unkn9", 0) or 0)
						if (
							blendShapeStartPos <= 0
							or blendShapeStartPos > len(entry.vertexBuffer)
						):
							vertexCount = len(vertexDictList[-1]["Position"])
							lastElement = entry.vertexElementList[-1]
							blendShapeStartPos = (
								lastElement.posStartOffset
								+ vertexCount * lastElement.stride
							)
						blendShapeBufferList.append(
							entry.vertexBuffer[blendShapeStartPos:]
						)
					usedVertexOffsetDictList.append(dict())
			
			usedVertexOffsetDictList.append(dict())
			while len(usedVertexOffsetDictList) < len(vertexDictList):
				usedVertexOffsetDictList.append(dict())
			#TODO
			#tags.add("shadowLOD")
			#shadowVertexDict = ReadVertexElementBuffers(reMesh.meshBufferHeader.vertexElementList, reMesh.meshBufferHeader.vertexBuffer,tags)
				
		#Parse Main Meshes
		if reMesh.lodHeader != None and len(vertexDictList) != 0:
			if reMesh.lodHeader.has32BitIndexBuffer:
				self.bufferHasIntFaces = True
			self.boundingSphere = reMesh.lodHeader.sphere
			self.boundingBox = reMesh.lodHeader.bbox
			self.mainMeshLODList = parseLODStructure(reMesh,reMesh.lodHeader.lodGroupList,vertexDictList,faceBufferList,usedVertexOffsetDictList,blendShapeBufferList)
			if reMesh.meshVersion in BLEND_SHAPE_MESH_VERSIONS:
				self.mhwildsNormalPivotImportReport = (
					_ResolveMHWILDSLOD0NormalPivotIdentities(
						reMesh, self.mainMeshLODList
					)
				)
				self.mhwildsCanonicalTableProfile = []
				for lodGroup in reMesh.lodHeader.lodGroupList:
					vertexBufferIndex = 0
					if (
						lodGroup.meshGroupList
						and lodGroup.meshGroupList[0].vertexInfoList
					):
						vertexBufferIndex = int(
							lodGroup.meshGroupList[0]
							.vertexInfoList[0]
							.vertexBufferIndex
						)
					if (
						vertexBufferIndex > 0
						and vertexBufferIndex - 1
						< len(
							reMesh.meshBufferHeader
							.streamingBufferHeaderList
						)
					):
						entry = (
							reMesh.meshBufferHeader
							.streamingBufferHeaderList[
								vertexBufferIndex - 1
							]
						)
						usesTables = not (
							int(entry.unkn7)
							== int(entry.unkn8)
							== int(entry.unkn9)
						)
					else:
						usesTables = False
					self.mhwildsCanonicalTableProfile.append(usesTables)
			for i in range(len(self.mainMeshLODList)):
				lodOffsetDict[reMesh.lodHeader.lodGroupOffsetList[i]] = self.mainMeshLODList[i]
		if reMesh.shadowHeader != None and len(vertexDictList) != 0:
			for offset in reMesh.shadowHeader.lodGroupOffsetList:
				if offset in lodOffsetDict:
					self.shadowMeshLinkedLODList.append(lodOffsetDict[offset])
				else:#This shouldn't happen
					#Update: it does :/
					#RE3_EXTRACT\re_chunk_000\natives\stm\escape\character\enemy\em9200\mesh\em9200.mesh.2109108288
					#TODO Add unique shadow mesh LOD importing
					print("ERROR: Shadow mesh has unique lod offsets, cannot import")
			#self.shadowMeshLODList = parseLODStructure(reMesh,reMesh.shadowHeader.lodGroupList,vertexDict,usedVertexOffsetDict)
		
		#TODO Add occlusion mesh
		
		if self.isMPLY:
			
			minAABB = reMesh.meshletLayout.meshletHeader.minAABB
			maxAABB = reMesh.meshletLayout.meshletHeader.maxAABB
			self.boundingBox.min.x = minAABB[0]
			self.boundingBox.min.y = minAABB[1]
			self.boundingBox.min.z = minAABB[2]
			self.boundingBox.max.x = maxAABB[0]
			self.boundingBox.max.y = maxAABB[1]
			self.boundingBox.max.z = maxAABB[2]
			
			
			AABBCenter = (np.array(minAABB) + np.array(maxAABB))/2
			AABBOffset = np.array(reMesh.meshletBVH.offset)
			AABBScale = reMesh.meshletBVH.scale
			print("Parsing MPLY.")
			self.mainMeshLODList = []
			self.materialNameList = reMesh.rawNameList
			for lodIndex in range(0,reMesh.meshletLayout.gpuMeshletHeader.lodNum):
				#print(lodIndex)
				lod = LODLevel()
				lod.lodDistance = reMesh.meshletLayout.gpuMeshletHeader.lodFactor
				group = VisconGroup()
				group.visconGroupNum = 0
				
				for submeshIndex, clusterHeader in enumerate(reMesh.meshletBVH.clusterHeaderLODList[lodIndex]):
					submesh = SubMesh()
					submesh.materialIndex = clusterHeader.bitfield.fields.materialId
					submesh.subMeshIndex = submeshIndex
					#print(f"{submeshIndex} - {submesh.materialIndex}")
					meshEntry = reMesh.clusterInfoLayout.lodList[lodIndex].entryList[submeshIndex]
					
					#TEMP
					tags = set()
					
					#Calculate bounding box
					submesh.boundingBox = AABB()
					
					center = np.array(meshEntry.bboxAABBCenter)
					extent = np.array(meshEntry.bboxExtent)
					
					subAABBMin = center - extent
					subAABBMin *= AABBScale
					subAABBMin += AABBOffset
					
					
					subAABBMax = center + extent
					subAABBMax *= AABBScale
					subAABBMax += AABBOffset
					#submesh.relPos = (center * AABBScale) + AABBOffset
					submesh.relPos = (0.0,0.0,0.0)
					
					submesh.boundingBox.min.x = subAABBMin[0]
					submesh.boundingBox.min.y = subAABBMin[1]
					submesh.boundingBox.min.z = subAABBMin[2]
					
					submesh.boundingBox.max.x = subAABBMax[0]
					submesh.boundingBox.max.y = subAABBMax[1]
					submesh.boundingBox.max.z = subAABBMax[2]
					
					
					#print(f"Transform: {positionModifier}")
					#print(f"Scaling: {scaleModifier}")
					submesh.vertexPosList = ReadCompressedPosBuffer(meshEntry.posBuffer,meshEntry.bitFlag,np.array(meshEntry.partAABBCenter),center - np.array((0.5,0.5,0.5)))
					if meshEntry.bitFlag.flags.isMeshletNoTangent:
						submesh.normalList = ReadNorBuffer(meshEntry.normalBuffer,tags)
					else:
						normalList,tangentList = ReadNorTanBuffer(meshEntry.normalBuffer,tags)
						submesh.normalList = normalList
						submesh.tangentList = tangentList
					
					submesh.uvList = ReadUVBuffer(meshEntry.uvBuffer, tags)
					if meshEntry.uv2Buffer != None:
						submesh.uv2List = ReadUVBuffer(meshEntry.uv2Buffer, tags)
					if meshEntry.uv3Buffer != None:
						submesh.uv3List = ReadUVBuffer(meshEntry.uv3Buffer, tags)
					if meshEntry.colorBuffer != None:
						submesh.colorList = ReadColorBuffer(meshEntry.colorBuffer, tags)
					if reMesh.streamingBuffer != None:
						
						faceStartOffset = clusterHeader.indexOffsetBytes
						faceEndOffset = clusterHeader.indexOffsetBytes + (clusterHeader.bitfield.fields.indexCount*2)
						#print(f"LOD {lodIndex} sub {submeshIndex} - face start: {faceStartOffset} end: {faceEndOffset}")
						submesh.faceList = ReadFaceBuffer(reMesh.streamingBuffer[faceStartOffset:faceEndOffset])
					else:
						submesh.faceList = ReadFaceBuffer(meshEntry.faceBuffer)
					
					
					group.subMeshList.append(submesh)
				lod.visconGroupList.append(group)
				self.mainMeshLODList.append(lod)
			pass#TODO Parse MPLY