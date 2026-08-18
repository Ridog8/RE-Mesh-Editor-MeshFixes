"""Monster Hunter Wilds blend-shape support for the unified RE Mesh exporter."""

import copy
import math
import os
import struct
import zlib
from collections import Counter
from io import BytesIO

import numpy as np

from . import file_re_mesh as mesh_format


EDIT_DELTA_THRESHOLD = 0.00001
DUMMY_SHAPEKEY_PREFIX = "DUMMY_"
NORMAL_GROUP_ATTRIBUTE = "MHWILDS_NormalGroup"
NORMAL_PIVOT_ATTRIBUTE = "MHWILDS_NormalPivot"
NORMAL_PIVOT0_ATTRIBUTE = "MHWILDS_NormalPivot0"
NORMAL_PIVOT255_ATTRIBUTE = "MHWILDS_NormalPivot255"

MHWILDS_BLENDSHAPE_MODE_DEFAULT = 0
MHWILDS_BLENDSHAPE_MODES = (0, 1)
MHWILDS_BLENDSHAPE_MODE_ATTRIBUTE = "_mhwildsBlendShapeMode"


def set_export_mode(parsed_mesh, mode):
	mode = int(mode)
	if mode not in MHWILDS_BLENDSHAPE_MODES:
		raise ValueError(f"MHWILDS blendshape mode must be 0 or 1, got {mode}")
	setattr(parsed_mesh, MHWILDS_BLENDSHAPE_MODE_ATTRIBUTE, mode)
	return mode


def _export_mode(parsed_mesh=None, re_mesh=None):
	"""Resolve the per-export mode, defaulting to Wilds mode 0."""
	for owner in (re_mesh, parsed_mesh):
		if owner is None:
			continue
		if hasattr(owner, MHWILDS_BLENDSHAPE_MODE_ATTRIBUTE):
			mode = int(getattr(owner, MHWILDS_BLENDSHAPE_MODE_ATTRIBUTE))
			if mode not in MHWILDS_BLENDSHAPE_MODES:
				raise RuntimeError(
					f"Invalid MHWILDS blendshape mode on export object: {mode}"
				)
			return mode
	return MHWILDS_BLENDSHAPE_MODE_DEFAULT


def _align16(value):
	return mesh_format.getPaddedPos(int(value), 16)


def reserve_normal_recalc_marker(re_mesh, current_offset):
	"""Reserve vanilla's 16-byte normal marker directly before blendShapesOffset."""
	marker_offset = _align16(current_offset)
	re_mesh.fileHeader.normalRecalcOffset = marker_offset
	re_mesh._mhwildsNormalMarkerOffset = marker_offset
	return marker_offset + 16


def _list_or_empty(value):
	"""Convert sequences, including NumPy arrays, without testing truthiness."""
	return [] if value is None else list(value)


def _shape_entry(parsed_submesh, shape_name):
	for entry in _list_or_empty(
		getattr(parsed_submesh, "blendShapeList", None)
	):
		if getattr(entry, "blendShapeName", "") == shape_name:
			return entry
	return None


def _shape_names(parsed_submeshes):
	names = []
	for parsed_submesh, _mesh_info, _vertex_count in parsed_submeshes:
		for entry in _list_or_empty(
			getattr(parsed_submesh, "blendShapeList", None)
		):
			name = str(getattr(entry, "blendShapeName", "") or "")
			if name.startswith(DUMMY_SHAPEKEY_PREFIX):
				continue
			if name and name not in names:
				names.append(name)
	return names


def _normalized_deltas(parsed_submesh, shape_name, vertex_count):
	entry = _shape_entry(parsed_submesh, shape_name)
	if entry is None:
		return np.zeros((vertex_count, 3), dtype=np.float32)
	deltas = np.asarray(getattr(entry, "deltas", []), dtype=np.float32)
	if deltas.size == 0:
		return np.zeros((vertex_count, 3), dtype=np.float32)
	deltas = deltas.reshape((-1, 3))
	if len(deltas) < vertex_count:
		deltas = np.vstack((
			deltas,
			np.zeros((vertex_count - len(deltas), 3), dtype=np.float32),
		))
	elif len(deltas) > vertex_count:
		deltas = deltas[:vertex_count]
	return deltas


def _symmetric_aabb(delta_arrays):
	aabb = mesh_format.AABB()
	valid = [
		np.asarray(array, dtype=np.float32).reshape((-1, 3))
		for array in delta_arrays
		if np.asarray(array).size
	]
	if valid:
		maximum = np.maximum(
			np.max(np.abs(np.vstack(valid)), axis=0),
			np.array((0.000001, 0.000001, 0.000001), dtype=np.float32),
		)
	else:
		maximum = np.array(
			(0.000001, 0.000001, 0.000001), dtype=np.float32
		)
	aabb.min.x, aabb.min.y, aabb.min.z = (
		float(-maximum[0]),
		float(-maximum[1]),
		float(-maximum[2]),
	)
	aabb.max.x, aabb.max.y, aabb.max.z = (
		float(maximum[0]),
		float(maximum[1]),
		float(maximum[2]),
	)
	aabb.min.w = 0.0
	aabb.max.w = 0.0
	return aabb


def _pack_delta_rows(deltas, aabb):
	"""Pack center-biased Wilds 11/10/11 rows; zero is 0x7FEFFBFF."""
	deltas = np.asarray(deltas, dtype=np.float32)
	if deltas.size == 0:
		return b""
	deltas = deltas.reshape((-1, 3))
	scale = np.array((
		max(abs(float(aabb.max.x)), 0.000001),
		max(abs(float(aabb.max.y)), 0.000001),
		max(abs(float(aabb.max.z)), 0.000001),
	), dtype=np.float32)
	normalized = np.clip(deltas / scale, -1.0, 1.0)
	x = np.clip(
		np.rint(1023.0 + normalized[:, 0] * 1023.0),
		0,
		2047,
	).astype(np.uint32)
	y = np.clip(
		np.rint(511.0 + normalized[:, 1] * 511.0),
		0,
		1023,
	).astype(np.uint32)
	z = np.clip(
		np.rint(1023.0 + normalized[:, 2] * 1023.0),
		0,
		2047,
	).astype(np.uint32)
	packed = (
		(x & 0x7FF)
		| ((y & 0x3FF) << 11)
		| ((z & 0x7FF) << 21)
	)
	return packed.astype("<I").tobytes()


def _exported_groups(lod, parsed_to_mesh_info):
	"""Split each viscon group by the shape channels actually authored per submesh."""
	groups = []
	for order, viscon in enumerate(
		_list_or_empty(getattr(lod, "visconGroupList", None))
	):
		submeshes = []
		for parsed_submesh in _list_or_empty(
			getattr(viscon, "subMeshList", None)
		):
			# Reused LOD geometry still needs its own Wilds blend body.
			# The MaterialSubdivision keeps the authoritative global start index,
			# while the visual vertex bytes remain de-duplicated in VBI0.
			mesh_info = parsed_to_mesh_info.get(parsed_submesh)
			vertex_positions = getattr(
				parsed_submesh, "vertexPosList", None
			)
			vertex_count = (
				0 if vertex_positions is None else len(vertex_positions)
			)
			if mesh_info is not None and vertex_count:
				submeshes.append(
					(parsed_submesh, mesh_info, int(vertex_count))
				)

		if not submeshes:
			continue

		# A Wilds active target must cover only submeshes that actually author
		# those channels. The former whole-viscon union caused unedited submeshes
		# to be serialized as active zero-delta participants.
		partitions = []
		partition_by_signature = {}
		for submesh in submeshes:
			shape_signature = tuple(_shape_names([submesh]))
			partition_index = partition_by_signature.get(shape_signature)
			if partition_index is None:
				partition_index = len(partitions)
				partition_by_signature[shape_signature] = partition_index
				partitions.append({
					"shapeNames": list(shape_signature),
					"submeshes": [],
				})
			partitions[partition_index]["submeshes"].append(submesh)

		for partition_index, partition in enumerate(partitions):
			groups.append({
				"order": order,
				"partition": partition_index,
				"groupID": int(
					getattr(viscon, "visconGroupNum", order) or 0
				),
				"submeshes": partition["submeshes"],
				"shapeNames": partition["shapeNames"],
			})
	return groups


def build_blend_shape_plan(parsed_mesh, parsed_to_mesh_info):
	"""Build active and inactive Wilds targets from Blender viscon groups."""
	lods = _list_or_empty(
		getattr(parsed_mesh, "mainMeshLODList", None)
	)
	if not lods:
		return None
	header = mesh_format.BlendShapeHeader()
	header.zero = 0
	header.hash = 0
	remap_names = []
	payloads = []
	remap_base = 0
	found_blend_lod = False
	aliases = _resolve_alias_map(
		getattr(parsed_mesh, "_blendShapeSharedLODMap", {}),
		len(lods),
	)
	body_remap_bases = []
	body_shape_name_sets = []

	for lod_index, lod in enumerate(lods):
		groups = _exported_groups(lod, parsed_to_mesh_info)
		if not groups:
			if found_blend_lod:
				break
			continue
		active_groups = [group for group in groups if group["shapeNames"]]
		inactive_groups = [
			group for group in groups if not group["shapeNames"]
		]
		if not active_groups:
			if any(
				any(group["shapeNames"] for group in _exported_groups(
					later_lod, parsed_to_mesh_info
				))
				for later_lod in lods[lod_index + 1:]
			):
				raise RuntimeError(
					"Wilds blend LODs must be a contiguous prefix from LOD0"
				)
			break
		found_blend_lod = True
		all_submeshes = [
			entry for group in groups for entry in group["submeshes"]
		]
		lod_vertex_base = min(
			int(mesh_info.vertexStartIndex)
			for _parsed, mesh_info, _count in all_submeshes
		)
		alias_owner = aliases.get(lod_index)
		if alias_owner is None:
			body_remap_base = remap_base
		else:
			alias_owner = int(alias_owner)
			if alias_owner >= len(body_remap_bases):
				raise RuntimeError(
					"Wilds blend alias owner has not been built: "
					f"LOD{lod_index}->LOD{alias_owner}"
				)
			body_remap_base = int(body_remap_bases[alias_owner])

		body = mesh_format.BlendShapeData()
		body.padding1 = 0
		body.padding2 = 0
		body.blendS = [0, 0, 0]
		body.aabbList = []
		body.blendSSList = []
		active_targets = []
		inactive_targets = []
		body_payload = bytearray()
		body_channel_count = 0

		for group in active_groups:
			shape_names = list(group["shapeNames"])
			target = mesh_format.BlendTarget()
			target.blendSSIndex = body_remap_base + body_channel_count
			target.blendShapeNum = len(shape_names)
			target.unkn0 = 0
			target.unkn2 = 1
			target.subMeshEntryList = []
			for _parsed, mesh_info, vertex_count in group["submeshes"]:
				entry = mesh_format.BlendSubMesh()
				# VBI0 is one global vertex namespace. Blend targets must use
				# the same global starts as the MaterialSubdivision records.
				entry.subMeshVertexStartIndex = int(mesh_info.vertexStartIndex)
				entry.vertOffset = 0
				entry.vertCount = int(vertex_count)
				entry.paramUnkn3 = 0
				target.subMeshEntryList.append(entry)
			target.subMeshEntryCount = len(target.subMeshEntryList)

			channel_deltas = []
			aabb_sources = []
			for shape_name in shape_names:
				per_submesh = []
				for parsed_submesh, _mesh_info, vertex_count in group[
					"submeshes"
				]:
					deltas = _normalized_deltas(
						parsed_submesh, shape_name, vertex_count
					)
					per_submesh.append(deltas)
					if (
						deltas.size
						and float(np.max(np.abs(deltas)))
						> EDIT_DELTA_THRESHOLD
					):
						aabb_sources.append(deltas)
				channel_deltas.append(per_submesh)
			aabb = _symmetric_aabb(aabb_sources)
			body.aabbList.append(aabb)
			for per_submesh in channel_deltas:
				for deltas in per_submesh:
					body_payload.extend(_pack_delta_rows(deltas, aabb))

			active_targets.append(target)
			if alias_owner is None:
				remap_names.extend(shape_names)
			body.blendSSList.extend(
				range(
					body_channel_count,
					body_channel_count + len(shape_names),
				)
			)
			body_channel_count += len(shape_names)

		for group in inactive_groups:
			target = mesh_format.BlendTarget()
			target.blendSSIndex = 0
			target.blendShapeNum = 0
			target.unkn0 = 0
			target.unkn2 = 0
			target.subMeshEntryList = []
			for _parsed, mesh_info, vertex_count in group["submeshes"]:
				entry = mesh_format.BlendSubMesh()
				entry.subMeshVertexStartIndex = int(mesh_info.vertexStartIndex)
				entry.vertOffset = 0
				entry.vertCount = int(vertex_count)
				entry.paramUnkn3 = 0
				if (
					target.subMeshEntryList
					and target.subMeshEntryList[-1].subMeshVertexStartIndex
					+ target.subMeshEntryList[-1].vertCount
					== entry.subMeshVertexStartIndex
				):
					target.subMeshEntryList[-1].vertCount += entry.vertCount
				else:
					target.subMeshEntryList.append(entry)
			target.subMeshEntryCount = len(target.subMeshEntryList)
			inactive_targets.append(target)

		body.targetCount = len(active_targets)
		body.typing = len(inactive_targets)
		body.blendTargetList = active_targets + inactive_targets
		body_shape_names = tuple(
			name
			for group in active_groups
			for name in group["shapeNames"]
		)
		if alias_owner is not None:
			owner_shape_names = body_shape_name_sets[alias_owner]
			if body_shape_names != owner_shape_names:
				raise RuntimeError(
					"Shared Wilds blend LOD has different shape channels: "
					f"LOD{lod_index}->LOD{alias_owner}"
				)
			owner_body = header.blendShapeList[alias_owner]
			owner_channel_count = int(
				getattr(owner_body, "_mhwildsChannelCount", 0)
			)
			if body_channel_count != owner_channel_count:
				raise RuntimeError(
					"Shared Wilds blend LOD channel count differs: "
					f"LOD{lod_index}={body_channel_count}, "
					f"LOD{alias_owner}={owner_channel_count}"
				)

		body.unknFlag = (
			((body_channel_count & 0xFFFF) << 16)
			| (body_remap_base & 0xFFFF)
		)
		body._mhwildsChannelCount = int(body_channel_count)
		header.blendShapeList.append(body)
		body_remap_bases.append(int(body_remap_base))
		body_shape_name_sets.append(body_shape_names)
		padding = mesh_format.getPaddingAmount(len(body_payload), 16)
		body_payload.extend(b"\x00" * padding)
		payloads.append(bytes(body_payload))
		if alias_owner is None:
			remap_base += body_channel_count

	if not header.blendShapeList:
		return None
	header.count = len(header.blendShapeList)
	blend_aliases = {
		int(target): int(source)
		for target, source in aliases.items()
		if int(target) < header.count
	}
	# Current-Blender source provenance for canonical-table generation.
	max_vertex_end = 0
	for lod in lods:
		for viscon in _list_or_empty(getattr(lod, "visconGroupList", None)):
			for parsed_submesh in _list_or_empty(getattr(viscon, "subMeshList", None)):
				mesh_info = parsed_to_mesh_info.get(parsed_submesh)
				positions = getattr(parsed_submesh, "vertexPosList", None)
				if mesh_info is not None and positions is not None:
					max_vertex_end = max(
						max_vertex_end,
						int(mesh_info.vertexStartIndex) + len(positions),
					)
	source_vertex_indices = [-1] * max_vertex_end
	source_identity_reliable = [False] * max_vertex_end
	for lod in lods:
		for viscon in _list_or_empty(getattr(lod, "visconGroupList", None)):
			for parsed_submesh in _list_or_empty(getattr(viscon, "subMeshList", None)):
				mesh_info = parsed_to_mesh_info.get(parsed_submesh)
				positions = getattr(parsed_submesh, "vertexPosList", None)
				if mesh_info is None or positions is None:
					continue
				count = len(positions)
				values = _list_or_empty(
					getattr(parsed_submesh, "blendShapeSourceVertexIndexList", None)
				)
				reliable = bool(
					getattr(parsed_submesh, "blendShapeSplitLoopVertices", False)
				) and len(values) == count
				start = int(mesh_info.vertexStartIndex)
				for local_index in range(count):
					row = start + local_index
					if 0 <= row < max_vertex_end:
						source_vertex_indices[row] = (
							int(values[local_index]) if len(values) == count else local_index
						)
						source_identity_reliable[row] = reliable

	return {
		"header": header,
		"shapeRemapNames": remap_names,
		"payloads": payloads,
		"aliases": blend_aliases,
		"bodyRemapBases": tuple(body_remap_bases),
		"sourceVertexIndices": tuple(source_vertex_indices),
		"sourceIdentityReliable": tuple(source_identity_reliable),
	}


def relocate_blend_shape_header(header, base_offset, version):
	"""Lay out each physical body and its child tables at absolute offsets."""
	header.count = len(header.blendShapeList)
	header.mainOffset = base_offset + 32 if header.count else 0
	current = _align16(base_offset + 32 + 8 * header.count)
	header.blendShapeOffsetList = []
	for body in header.blendShapeList:
		body_start = current
		header.blendShapeOffsetList.append(body_start)
		current = body_start + 48
		for target in body.blendTargetList:
			target.subMeshEntryCount = len(target.subMeshEntryList)
			if target.subMeshEntryCount:
				target.subMeshEntryOffset = current
				current += target.subMeshEntryCount * 16
			else:
				target.subMeshEntryOffset = 0
		current = _align16(current)
		body.dataOffset = current
		current += len(body.blendTargetList) * 16
		current = _align16(current)
		body.aabbOffset = current
		current += body.targetCount * 32
		current = _align16(current)
		body.blendSOffset = current
		current += len(body.blendS) * 4
		current = _align16(current)
		body.blendSSOffset = current
		current += len(body.blendSSList) * 4
		current = _align16(current)
	header._physicalBlendOffsets = list(
		header.blendShapeOffsetList
	)
	return current


def _face_intervals(lod_range):
	intervals = []
	for group in lod_range["lod"].meshGroupList:
		for entry in group.vertexInfoList:
			start = int(entry.faceStartIndex) - lod_range["baseFace"]
			count = int(entry.faceCount)
			vertex_start = (
				int(entry.vertexStartIndex) - lod_range["baseVertex"]
			)
			if count:
				intervals.append((start, start + count, vertex_start))
	intervals.sort(key=lambda value: value[0])
	return intervals


def _canonical_tables(
	re_mesh,
	lod_index,
	local_faces,
	local_vertices,
	elements,
	lod_range,
	source_vertex_indices=None,
	source_identity_reliable=None,
):
	vertex_count = lod_range["vertexCount"]
	intervals = _face_intervals(lod_range)
	if not intervals:
		raise RuntimeError(f"LOD{lod_index} has no canonical-table intervals")
	position = next(
		(element for element in elements if int(element.typing) == 0),
		None,
	)
	normal = next(
		(element for element in elements if int(element.typing) == 1),
		None,
	)
	if position is None or int(position.stride) < 12:
		raise RuntimeError(f"LOD{lod_index} has no usable position element")

	interval_for_vertex = [0] * vertex_count
	vertex_intervals = sorted(
		(vertex_start, interval_index)
		for interval_index, (_start, _end, vertex_start) in enumerate(
			intervals
		)
	)
	for index, (start, interval_index) in enumerate(vertex_intervals):
		end = (
			vertex_intervals[index + 1][0]
			if index + 1 < len(vertex_intervals)
			else vertex_count
		)
		for vertex_index in range(max(0, start), min(vertex_count, end)):
			interval_for_vertex[vertex_index] = interval_index

	def normal_vector(vertex_index):
		if normal is None or int(normal.stride) < 3:
			return None
		offset = int(normal.posStartOffset) + vertex_index * int(normal.stride)
		x, y, z = struct.unpack_from("<bbb", local_vertices, offset)
		length_squared = float(x * x + y * y + z * z)
		if length_squared == 0.0:
			return None
		inverse = 1.0 / math.sqrt(length_squared)
		return (x * inverse, y * inverse, z * inverse)

	candidates = {}
	canonical = []
	merge_dot = math.cos(math.radians(1.0))
	for vertex_index in range(vertex_count):
		offset = (
			int(position.posStartOffset)
			+ vertex_index * int(position.stride)
		)
		# If the current Blender exporter split one Blender vertex into
		# multiple game rows, that Blender vertex identity is the strongest
		# available logical/canonical provenance. For legacy 1:1 exports,
		# retain the validated geometric fallback instead of pretending a
		# unique source index proves that coincident imported rows are unrelated.
		use_source_identity = (
			source_vertex_indices is not None
			and source_identity_reliable is not None
			and vertex_index < len(source_vertex_indices)
			and vertex_index < len(source_identity_reliable)
			and bool(source_identity_reliable[vertex_index])
		)
		if use_source_identity:
			key = (
				interval_for_vertex[vertex_index],
				"BLENDER_VERTEX",
				int(source_vertex_indices[vertex_index]),
			)
		else:
			key = (
				interval_for_vertex[vertex_index],
				"GEOMETRIC_FALLBACK",
				bytes(local_vertices[offset:offset + 12]),
			)
		matches = candidates.setdefault(key, [])
		selected = None
		current_normal = normal_vector(vertex_index)
		for candidate in matches:
			candidate_normal = normal_vector(candidate)
			if current_normal is None:
				selected = candidate
				break
			if candidate_normal is not None:
				dot = sum(
					left * right
					for left, right in zip(
						current_normal, candidate_normal
					)
				)
				if dot >= merge_dot:
					selected = candidate
					break
		if selected is None:
			selected = vertex_index
			matches.append(vertex_index)
		canonical.append(int(selected))

	index_format = "<I" if re_mesh.lodHeader.has32BitIndexBuffer else "<H"
	index_stride = 4 if re_mesh.lodHeader.has32BitIndexBuffer else 2
	coordinate_count = max(end for _start, end, _vertex in intervals)
	face_values = [0] * coordinate_count
	canonical_face_vertices = [0] * coordinate_count
	for start, end, vertex_start in intervals:
		for coordinate in range(start, end):
			local_index = struct.unpack_from(
				index_format, local_faces, coordinate * index_stride
			)[0]
			combined = vertex_start + int(local_index)
			canonical_face_vertices[coordinate] = (
				canonical[combined]
				if 0 <= combined < len(canonical)
				else combined
			)
		for block_start in range(start, end, 1536):
			block_end = min(end, block_start + 1536)
			values = canonical_face_vertices[block_start:block_end]
			lane_by_vertex = {
				value: lane
				for lane, value in enumerate(sorted(set(values)))
			}
			for coordinate in range(block_start, block_end):
				value = canonical_face_vertices[coordinate]
				face_values[coordinate] = (
					(int(value) & 0x003FFFFF)
					| ((lane_by_vertex[value] & 0x3FF) << 22)
				)

	face_table = bytearray()
	for value in face_values:
		face_table.extend(struct.pack("<I", value & 0xFFFFFFFF))
	face_table.extend(
		b"\x00" * mesh_format.getPaddingAmount(len(face_table), 16)
	)

	boundary_vertices = set()
	for start, end, _vertex_start in intervals:
		edge_counts = Counter()
		triangle_end = start + ((end - start) // 3) * 3
		for coordinate in range(start, triangle_end, 3):
			triangle = canonical_face_vertices[
				coordinate:coordinate + 3
			]
			for left, right in (
				(triangle[0], triangle[1]),
				(triangle[1], triangle[2]),
				(triangle[2], triangle[0]),
			):
				edge_counts[tuple(sorted((left, right)))] += 1
		for (left, right), count in edge_counts.items():
			if count == 1:
				boundary_vertices.update((left, right))

	vertex_table = bytearray()
	for value in canonical:
		flag = 0x80000000 if value in boundary_vertices else 0
		vertex_table.extend(
			struct.pack("<I", (value & 0x7FFFFFFF) | flag)
		)
	vertex_table.extend(
		b"\x00" * mesh_format.getPaddingAmount(len(vertex_table), 16)
	)
	return bytes(face_table), bytes(vertex_table)


def _resolve_alias_map(raw_map, lod_count):
	result = {}
	for raw_target, raw_source in dict(raw_map or {}).items():
		target = int(raw_target)
		source = int(raw_source)
		if (
			target <= 0
			or target >= lod_count
			or source < 0
			or source >= target
		):
			raise RuntimeError(
				f"Invalid shared LOD mapping {target}->{source}"
			)
		while source in result:
			source = result[source]
		result[target] = source
	return result


def _stable_bucket(preferred_bytes, palette):
	"""Pick a deterministic starting group from a palette without Python hash()."""
	if not palette:
		return 0
	crc = zlib.crc32(preferred_bytes) & 0xFFFFFFFF
	return int(palette[crc % len(palette)])


def _color_blender_submesh(vertex_positions, face_list, source_indices, palette):
	"""Build deterministic adjacency-safe recalc buckets from current Blender topology."""
	count = len(vertex_positions)
	if count == 0 or not palette:
		return np.zeros(count, dtype=np.int16)

	adjacency = [set() for _ in range(count)]
	for face in _list_or_empty(face_list):
		if len(face) != 3:
			continue
		try:
			a, b, c = (int(face[0]), int(face[1]), int(face[2]))
		except Exception:
			continue
		if not all(0 <= value < count for value in (a, b, c)):
			continue
		for left, right in ((a, b), (b, c), (c, a)):
			if left != right:
				adjacency[left].add(right)
				adjacency[right].add(left)

	groups = np.zeros(count, dtype=np.int16)
	palette = tuple(int(value) for value in palette)
	palette_index = {value: index for index, value in enumerate(palette)}
	for vertex_index in range(count):
		source_index = (
			int(source_indices[vertex_index])
			if vertex_index < len(source_indices)
			else int(vertex_index)
		)
		position = np.asarray(vertex_positions[vertex_index], dtype=np.float32)
		seed = struct.pack(
			"<I3f",
			source_index & 0xFFFFFFFF,
			float(position[0]),
			float(position[1]),
			float(position[2]),
		)
		preferred = _stable_bucket(seed, palette)
		start = palette_index[preferred]
		for step in range(len(palette)):
			candidate = palette[(start + step) % len(palette)]
			if all(int(groups[neighbor]) != candidate for neighbor in adjacency[vertex_index]):
				groups[vertex_index] = candidate
				break
		if groups[vertex_index] == 0:
			raise RuntimeError(
				"Blender-authored normal topology exceeded the 254 Wilds recalc groups"
			)
	return groups


def _normal_contract(parsed_mesh):
	"""Generate Normal.w and all 256 pivot rows from CURRENT Blender topology.

	Imported MHWILDS_NormalGroup / pivot-marker attributes are deliberately not
	used to build this contract. If present, they are consulted only for a
	diagnostic agreement report.
	"""
	lods = _list_or_empty(getattr(parsed_mesh, "mainMeshLODList", None))
	if not lods:
		return None

	# First gather the physical Blender-authored submeshes in the same order used
	# by the parsed exporter. Reused logical LODs do not add another VBI0 range.
	lod_records = []
	global_vertex_count = 0
	for lod_index, lod in enumerate(lods):
		records = []
		for viscon in _list_or_empty(getattr(lod, "visconGroupList", None)):
			for submesh in _list_or_empty(getattr(viscon, "subMeshList", None)):
				if bool(getattr(submesh, "isReusedMesh", False)):
					continue
				positions = np.asarray(
					getattr(submesh, "vertexPosList", []), dtype=np.float32
				).reshape((-1, 3))
				count = len(positions)
				if count == 0:
					continue
				source_indices = [
					int(value) for value in _list_or_empty(
						getattr(submesh, "blendShapeSourceVertexIndexList", None)
					)
				]
				if len(source_indices) != count:
					source_indices = list(range(count))
				shape_names = _shape_names([(submesh, None, count)])
				records.append({
					"submesh": submesh,
					"positions": positions,
					"faces": _list_or_empty(getattr(submesh, "faceList", None)),
					"sourceIndices": source_indices,
					"shapeNames": tuple(shape_names),
					"globalStart": global_vertex_count,
					"count": count,
				})
				global_vertex_count += count
		lod_records.append(records)

	if not lod_records or not lod_records[0]:
		raise RuntimeError("Blender-authored Wilds normal contract has no physical LOD0 vertices")

	lod0_vertex_count = sum(record["count"] for record in lod_records[0])
	if lod0_vertex_count <= 0:
		raise RuntimeError("Blender-authored Wilds normal contract has empty LOD0")

	# Wilds exposes 254 nonzero logical group IDs. Use the complete palette when
	# LOD0 has enough blend-bearing vertices so the generated contract remains
	# deterministic and has valid pivots for every group higher LODs may use.
	lod0_active_count = sum(
		record["count"] for record in lod_records[0] if record["shapeNames"]
	)
	palette_size = min(254, max(1, lod0_active_count))
	lod0_palette = tuple(range(1, palette_size + 1))

	all_logical_groups = []
	lod_vertex_counts = []
	lod_used_groups = []
	lod0_positions = []
	lod0_global_groups = []
	lod0_reference_groups = []
	lod0_reference_valid = []

	def color_lod(records, palette, lod_index):
		logical_arrays = []
		current_count = 0
		for record in records:
			count = int(record["count"])
			if record["shapeNames"]:
				logical = _color_blender_submesh(
					record["positions"],
					record["faces"],
					record["sourceIndices"],
					palette,
				)
			else:
				logical = np.zeros(count, dtype=np.int16)
			logical_arrays.append(logical)
			current_count += count
		return logical_arrays, current_count

	# Color LOD0 first. Its actually used group IDs define the only palette
	# higher LODs are allowed to reference, guaranteeing every higher-LOD group
	# has a real Blender-authored LOD0 pivot row.
	lod0_arrays, lod0_count = color_lod(lod_records[0], lod0_palette, 0)
	for record, logical in zip(lod_records[0], lod0_arrays):
		all_logical_groups.append(logical)
		lod0_positions.append(record["positions"])
		lod0_global_groups.append(logical)
		ref = _list_or_empty(getattr(record["submesh"], "normalGroupList", None))
		if len(ref) == int(record["count"]):
			lod0_reference_groups.extend(int(value) for value in ref)
			lod0_reference_valid.extend([True] * int(record["count"]))
		else:
			lod0_reference_groups.extend([0] * int(record["count"]))
			lod0_reference_valid.extend([False] * int(record["count"]))
	lod_vertex_counts.append(lod0_count)
	lod0_joined = np.concatenate(lod0_arrays) if lod0_arrays else np.zeros(0, dtype=np.int16)
	used_lod0_groups = tuple(sorted(set(int(value) for value in lod0_joined if value)))
	lod_used_groups.append(used_lod0_groups)
	higher_palette = used_lod0_groups if used_lod0_groups else (1,)

	for lod_index, records in enumerate(lod_records[1:], start=1):
		logical_arrays, current_count = color_lod(records, higher_palette, lod_index)
		all_logical_groups.extend(logical_arrays)
		lod_vertex_counts.append(current_count)
		joined = np.concatenate(logical_arrays) if logical_arrays else np.zeros(0, dtype=np.int16)
		lod_used_groups.append(tuple(sorted(set(int(value) for value in joined if value))))

	groups = np.concatenate(all_logical_groups) if all_logical_groups else np.zeros(0, dtype=np.int16)
	positions = np.concatenate(lod0_positions) if lod0_positions else np.zeros((0, 3), dtype=np.float32)
	lod0_groups = np.concatenate(lod0_global_groups) if lod0_global_groups else np.zeros(0, dtype=np.int16)

	if len(groups) != global_vertex_count:
		raise RuntimeError(
			"Generated Blender normal-group count differs from physical vertices: "
			f"{len(groups)}/{global_vertex_count}"
		)
	if np.any(groups < 0) or np.any(groups > 254):
		raise RuntimeError("Generated Wilds normal groups exceeded 0..254")

	# Every used logical group gets a pivot selected from CURRENT LOD0 geometry.
	# The group IDs are just deterministic buckets; no imported pivot identity is
	# required. Unused slots collapse to slot 0, as in the previous writer.
	pivot_indices = [0] * 256
	if len(positions) == 0:
		raise RuntimeError("Generated Wilds pivot table has no LOD0 positions")
	# Deterministic special slots from Blender geometry, independent of source file.
	lex_order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
	slot0_index = int(lex_order[0])
	slot255_index = int(lex_order[-1])
	pivot_indices[0] = slot0_index
	pivot_indices[255] = slot255_index
	used_lod0_groups = list(used_lod0_groups)
	for group in used_lod0_groups:
		rows = np.flatnonzero(lod0_groups == group)
		if len(rows):
			pivot_indices[group] = int(rows[0])
	for group in range(1, 255):
		if group not in used_lod0_groups:
			pivot_indices[group] = slot0_index

	group_lane = bytearray(len(groups))
	for row, logical in enumerate(groups):
		logical = int(logical)
		if logical == 0:
			encoded = 0
		elif logical <= 127:
			encoded = logical
		else:
			encoded = logical + 1
		group_lane[row] = encoded

	# Diagnostic only: imported attributes can tell us how different our generated
	# topology is on a vanilla reference, but deleting every attribute must not
	# change one byte of generated output.
	valid_mask = np.asarray(lod0_reference_valid, dtype=bool)
	if len(valid_mask) == len(lod0_groups) and np.any(valid_mask):
		reference = np.asarray(lod0_reference_groups, dtype=np.int16)
		agreement = float(np.mean(reference[valid_mask] == lod0_groups[valid_mask]))
		print(
			"reference-only NormalGroup comparison: "
			f"rows={int(np.sum(valid_mask))}, exactAgreement={agreement:.4f}"
		)
	return {
		"groupLane": bytes(group_lane),
		"pivotIndices": tuple(pivot_indices),
		"vertexCount": len(groups),
		"lodVertexCounts": tuple(lod_vertex_counts),
		"lodUsedGroups": tuple(lod_used_groups),
		"generatedFromBlender": True,
	}


def _signed_short(value):
	value = int(value) & 0xFFFF
	return value - 0x10000 if value >= 0x8000 else value


def _global_resource_range(re_mesh):
	"""Build one de-duplicated global range for the Blender-authored VBI0."""
	unique_entries = []
	seen = set()
	for lod in re_mesh.lodHeader.lodGroupList:
		for group in lod.meshGroupList:
			for entry in group.vertexInfoList:
				key = (
					int(entry.vertexStartIndex),
					int(entry.faceStartIndex),
					int(entry.faceCount),
				)
				if key not in seen:
					seen.add(key)
					unique_entries.append(entry)

	class _GlobalGroup:
		pass

	class _GlobalLOD:
		pass

	group = _GlobalGroup()
	group.vertexInfoList = unique_entries
	lod = _GlobalLOD()
	lod.meshGroupList = [group]
	position = next(
		(element for element in re_mesh.meshBufferHeader.vertexElementList
		 if int(element.typing) == 0),
		None,
	)
	if position is None or int(position.stride) <= 0:
		raise RuntimeError("Unified Wilds export has no position element")
	elements = re_mesh.meshBufferHeader.vertexElementList
	position_index = elements.index(position)
	position_end = (
		int(elements[position_index + 1].posStartOffset)
		if position_index + 1 < len(elements)
		else len(re_mesh.meshBufferHeader.vertexBuffer)
	)
	vertex_count = (
		position_end - int(position.posStartOffset)
	) // int(position.stride)
	return {
		"lod": lod,
		"baseVertex": 0,
		"vertexCount": int(vertex_count),
		"baseFace": 0,
		"faceCount": len(re_mesh.meshBufferHeader.faceBuffer) // (
			4 if re_mesh.lodHeader.has32BitIndexBuffer else 2
		),
	}


def _assign_unified_payload_offsets(bodies, payloads, aliases):
	"""Assign global row offsets while omitting aliased duplicate payloads."""
	owners = []
	for body_index in range(len(bodies)):
		owner = int(aliases.get(body_index, body_index))
		while owner in aliases:
			owner = int(aliases[owner])
		owners.append(owner)

	owner_base_rows = {}
	payload_blob = bytearray()
	for body_index, (body, payload) in enumerate(zip(bodies, payloads)):
		owner = owners[body_index]
		if owner != body_index:
			continue
		body_row_base = len(payload_blob) // 4
		owner_base_rows[body_index] = body_row_base
		target_row_base = body_row_base
		for target in body.blendTargetList[:int(body.targetCount)]:
			entries = list(target.subMeshEntryList)
			target_rows = sum(int(entry.vertCount) for entry in entries)
			entry_prefix = 0
			for entry in entries:
				entry.vertOffset = target_row_base + entry_prefix
				entry_prefix += int(entry.vertCount)
			target_row_base += int(target.blendShapeNum) * target_rows
		useful_bytes = (target_row_base - body_row_base) * 4
		if useful_bytes > len(payload):
			raise RuntimeError(
				"Wilds body payload is smaller than its authored target rows"
			)
		payload_blob.extend(payload)

	for body_index, body in enumerate(bodies):
		owner = owners[body_index]
		if owner == body_index:
			continue
		owner_body = bodies[owner]
		for target, owner_target in zip(
			body.blendTargetList[:int(body.targetCount)],
			owner_body.blendTargetList[:int(owner_body.targetCount)],
		):
			for entry, owner_entry in zip(
				target.subMeshEntryList,
				owner_target.subMeshEntryList,
			):
				entry.vertOffset = int(owner_entry.vertOffset)
	return bytes(payload_blob)


def finalize_mhwilds_export(parsed_mesh, re_mesh):
	"""Keep Blender geometry and blend-target addressing in global VBI0."""
	mode = _export_mode(parsed_mesh=parsed_mesh)
	setattr(re_mesh, MHWILDS_BLENDSHAPE_MODE_ATTRIBUTE, mode)
	plan = re_mesh._blendShapeExportPlan
	lods = list(re_mesh.lodHeader.lodGroupList)
	bodies = list(re_mesh.blendShapeHeader.blendShapeList)
	payloads = list(plan["payloads"])
	if len(bodies) != len(payloads):
		raise RuntimeError("Wilds blend body/payload count differs")
	if len(bodies) > len(lods):
		raise RuntimeError("Wilds has more blend bodies than authored LODs")

	# The original exporter has already built the Blender-authoritative global
	# vertex/face namespaces. Keep those exact ranges and keep every visible
	# subdivision on the inline VBI0.
	for lod in lods:
		for group in lod.meshGroupList:
			for entry in group.vertexInfoList:
				entry.vertexBufferIndex = 0
				entry.streamingOffsetBytes = 0
				entry.streamingPlatormSpecificOffsetBytes = 0

	aliases = _resolve_alias_map(
		getattr(parsed_mesh, "_blendShapeSharedLODMap", {}),
		len(lods),
	)
	for lod_index, lod in enumerate(lods):
		lod.reserved = int(aliases.get(lod_index, 0))

	# Preserve the vanilla shared-LOD pointer contract where Blender collections
	# are genuinely shared. Physical records are still written for deterministic
	# round-tripping, but consumers see the same logical aliases as vanilla.
	for target, source in aliases.items():
		re_mesh.lodHeader.lodGroupOffsetList[target] = (
			re_mesh.lodHeader.lodGroupOffsetList[source]
		)

	physical_blend_offsets = list(
		re_mesh.blendShapeHeader.blendShapeOffsetList
	)
	for target, source in aliases.items():
		if target < len(bodies) and source < len(bodies):
			re_mesh.blendShapeHeader.blendShapeOffsetList[target] = (
				re_mesh.blendShapeHeader.blendShapeOffsetList[source]
			)
	re_mesh.blendShapeHeader._physicalBlendOffsets = (
		physical_blend_offsets
	)

	# Every target uses the same global VBI0 vertex namespace as its visible
	# MaterialSubdivision. vertOffset selects the body's packed delta rows.
	payload_blob = _assign_unified_payload_offsets(
		bodies, payloads, aliases
	)

	mesh_header = re_mesh.meshBufferHeader
	visual = bytes(mesh_header.vertexBuffer)
	elements = copy.deepcopy(mesh_header.vertexElementList)
	profile = _list_or_empty(
		getattr(parsed_mesh, "mhwildsCanonicalTableProfile", None)
	)
	use_tables = any(
		bool(profile[index]) if index < len(profile) else True
		for index in range(len(bodies))
	)
	if use_tables:
		global_range = _global_resource_range(re_mesh)
		face_table, vertex_table = _canonical_tables(
			re_mesh,
			0,
			bytes(mesh_header.faceBuffer),
			visual,
			elements,
			global_range,
			source_vertex_indices=plan.get("sourceVertexIndices"),
			source_identity_reliable=plan.get("sourceIdentityReliable"),
		)
	else:
		face_table, vertex_table = b"", b""

	# Wilds starts the canonical face table at the next 16-byte boundary
	# after the ordinary vertex elements. Head meshes happen to end aligned
	# already, while meshes with a 44-byte combined vertex stride (including
	# this chainmail mesh) require explicit padding here.
	visual_end = len(visual)
	canonical_start = _align16(visual_end)
	visual_padding = canonical_start - visual_end
	face_table_end = canonical_start + len(face_table)
	payload_start = face_table_end + len(vertex_table)
	vertex_blob = bytearray(visual)
	vertex_blob.extend(b"\x00" * visual_padding)
	vertex_blob.extend(face_table)
	vertex_blob.extend(vertex_table)
	vertex_blob.extend(payload_blob)
	vertex_length = _align16(len(vertex_blob))
	vertex_blob.extend(b"\x00" * (vertex_length - len(vertex_blob)))

	mesh_header.vertexBuffer = vertex_blob
	mesh_header.vertexElementList = elements
	mesh_header.mainVertexElementCount = len(elements)
	mesh_header.vertexElementCount = len(elements)
	mesh_header.vertexBufferSize = vertex_length
	mesh_header.faceBufferSize = len(mesh_header.faceBuffer)
	mesh_header.block2FaceBufferOffset = (
		vertex_length + mesh_header.faceBufferSize
	)
	mesh_header.NULL = mesh_header.block2FaceBufferOffset
	mesh_header.totalBufferSize = _align16(
		mesh_header.block2FaceBufferOffset
	)

	# Wilds stores the first canonical boundary in the two adjacent signed
	# 16-bit fields at MeshBufferHeader + 44. Together they form one 32-bit
	# value. The following qword stores boundaries two and three.
	canonical_boundary0 = int(canonical_start)
	mesh_header.vertexElementSize = _signed_short(
		canonical_boundary0 & 0xFFFF
	)
	mesh_header.unkn1 = _signed_short(
		(canonical_boundary0 >> 16) & 0xFFFF
	)
	re_mesh._mhwildsCanonicalBoundary0 = canonical_boundary0

	# The two real 32-bit Wilds boundaries remain in sunbreakSecondUnknown.
	mesh_header.sunbreakSecondUnknown = (
		(int(payload_start) << 32) | int(face_table_end)
	)
	mesh_header.sunbreakOffset = 0
	mesh_header.sf6unkn0 = 0
	mesh_header.streamingVertexElementOffset = 0
	mesh_header.sf6unkn2 = 0
	mesh_header.streamingBufferHeaderList = []
	re_mesh.streamingInfoHeader = None
	re_mesh.streamingBuffer = None

	# normal groups and all 256 pivot identities are generated from
	# current Blender topology. Imported Wilds point attributes are optional
	# diagnostics only and are not required for serialization.
	normal_plan = _normal_contract(parsed_mesh)
	mesh_offset = int(re_mesh.fileHeader.meshOffset)
	if normal_plan is not None:
		descriptor_offset = mesh_offset + 80
		mesh_header.vertexElementOffset = _align16(descriptor_offset + 16)
		re_mesh._mhwildsFloatDescriptorOffset = descriptor_offset
	else:
		mesh_header.vertexElementOffset = _align16(mesh_offset + 80)
		re_mesh._mhwildsFloatDescriptorOffset = 0

	marker_offset = int(
		getattr(re_mesh, "_mhwildsNormalMarkerOffset", 0)
	)
	if normal_plan is not None:
		if marker_offset <= 0:
			raise RuntimeError("did not reserve a normal marker slot")
		if marker_offset + 16 != int(re_mesh.fileHeader.blendShapesOffset):
			raise RuntimeError(
				"not directly before blendShapesOffset: "
				f"0x{marker_offset:X}/0x{int(re_mesh.fileHeader.blendShapesOffset):X}"
			)
	mesh_header.vertexBufferOffset = _align16(
		int(mesh_header.vertexElementOffset)
		+ _align16(len(elements) * 8)
	)
	mesh_header.faceBufferOffset = (
		mesh_header.vertexBufferOffset + mesh_header.vertexBufferSize
	)
	inline_end = (
		mesh_header.vertexBufferOffset + mesh_header.totalBufferSize
	)

	re_mesh.fileHeader.verticesOffset = mesh_header.vertexBufferOffset
	# A zero-entry StreamingInfo view is represented by the zero fields already
	# present at meshOffset+64, matching the original unified exporter.
	re_mesh.fileHeader.streamingInfoOffset = mesh_offset + 64
	re_mesh.fileHeader.normalRecalcOffset = marker_offset
	re_mesh.fileHeader.floatsOffset = int(
		getattr(re_mesh, "_mhwildsFloatDescriptorOffset", 0)
	)
	re_mesh.fileHeader.fileSize = inline_end
	# Both the FloatData pivot package and vanilla-positioned marker are active.
	re_mesh.fileHeader.contentFlag.bitFlag |= 0x800C
	re_mesh.fileHeader.contentFlag.parseBitFlag()

	re_mesh._mhwildsStreamBlob = b""
	re_mesh._mhwildsStreamBase = mesh_header.vertexBufferOffset
	re_mesh._mhwildsInlineEnd = inline_end
	re_mesh._mhwildsDeclarationStride = _align16(
		mesh_header.mainVertexElementCount * 8
	)
	re_mesh._blendShapeWriter = "MHWILDS"

	if normal_plan is not None:
		position = next(
			element for element in elements if int(element.typing) == 0
		)
		normal = next(
			element for element in elements if int(element.typing) == 1
		)
		global_vertex_count = int(
			_global_resource_range(re_mesh)["vertexCount"]
		)
		if int(normal_plan["vertexCount"]) != global_vertex_count:
			raise RuntimeError(
				"Blender and generated unified all-LOD Normal.w counts differ: "
				f"{int(normal_plan['vertexCount'])}/{global_vertex_count}"
			)
		normal_plan.update({
			"positionOffset": int(position.posStartOffset),
			"positionStride": int(position.stride),
			"normalOffset": int(normal.posStartOffset),
			"normalStride": int(normal.stride),
		})
		re_mesh._mhwildsNormalContract = normal_plan


def _write_generated_mesh_buffer(self, file, version):
	re_mesh = self._mhwildsOwner
	stream_blob = re_mesh._mhwildsStreamBlob
	stream_info = re_mesh.streamingInfoHeader
	headers = self.streamingBufferHeaderList
	mesh_format.write_uint64(file, self.vertexElementOffset)
	mesh_format.write_uint64(file, self.vertexBufferOffset)
	mesh_format.write_uint64(file, self.sunbreakOffset)
	mesh_format.write_uint(file, self.totalBufferSize)
	mesh_format.write_uint(file, self.vertexBufferSize)
	mesh_format.write_ushort(file, self.mainVertexElementCount)
	mesh_format.write_ushort(file, self.vertexElementCount)
	mesh_format.write_uint(file, self.block2FaceBufferOffset)
	mesh_format.write_uint(file, self.NULL)
	mesh_format.write_short(file, self.vertexElementSize)
	mesh_format.write_short(file, self.unkn1)
	mesh_format.write_uint64(file, self.sunbreakSecondUnknown)
	mesh_format.write_uint64(file, self.sf6unkn0)
	mesh_format.write_uint64(file, self.streamingVertexElementOffset)
	mesh_format.write_uint64(file, self.sf6unkn2)
	for header in headers:
		header.write(file, version)

	if headers:
		file.seek(stream_info.entryOffset)
		for entry in stream_info.streamingInfoEntryList:
			entry.write(file)
		file.seek(re_mesh.fileHeader.streamingInfoOffset)
		stream_info.write(file)
	file.seek(self.vertexElementOffset)
	for element in self.vertexElementList:
		element.write(file)
	if headers:
		file.seek(self.streamingVertexElementOffset)
		stride = re_mesh._mhwildsDeclarationStride
		for header in headers:
			start = file.tell()
			for element in header.vertexElementList[
				:self.mainVertexElementCount
			]:
				element.write(file)
			file.seek(start + stride)
	file.seek(re_mesh._mhwildsStreamBase)
	file.write(stream_blob)
	file.seek(self.vertexBufferOffset)
	file.write(self.vertexBuffer)
	file.seek(self.faceBufferOffset)
	file.write(self.faceBuffer)


def _apply_normal_contract(data, re_mesh):
	"""Write the Blender-generated R38 normal/pivot contract in mode 0 or 1."""
	mode = _export_mode(re_mesh=re_mesh)
	plan = getattr(re_mesh, "_mhwildsNormalContract", None)
	if plan is None:
		return data
	data = bytearray(data)
	lod0_start = int(re_mesh.meshBufferHeader.vertexBufferOffset)
	position_start = lod0_start + int(plan["positionOffset"])
	normal_start = lod0_start + int(plan["normalOffset"])

	for row, encoded in enumerate(plan["groupLane"]):
		data[
			normal_start + row * int(plan["normalStride"]) + 3
		] = int(encoded)

	pivot_bytes = bytearray()
	for row in plan["pivotIndices"]:
		start = position_start + int(row) * int(plan["positionStride"])
		pivot_bytes.extend(data[start:start + 12])
	if len(pivot_bytes) != 3072:
		raise RuntimeError("Wilds pivot table must contain 256 Vec3 rows")

	descriptor_offset = int(
		getattr(re_mesh, "_mhwildsFloatDescriptorOffset", 0)
	)
	marker_offset = int(
		getattr(re_mesh, "_mhwildsNormalMarkerOffset", 0)
	)
	if descriptor_offset <= 0:
		raise RuntimeError("did not reserve a FloatData descriptor")
	if marker_offset <= 0:
		raise RuntimeError("did not reserve a normal marker")
	if marker_offset + 16 != int(re_mesh.fileHeader.blendShapesOffset):
		raise RuntimeError(
			"blendshape adjacency changed before serialization"
		)

	# The reserved slot must be unused metadata padding before we populate it.
	if any(data[marker_offset:marker_offset + 16]):
		raise RuntimeError("slot was overwritten by another structure")
	struct.pack_into("<IIII", data, marker_offset, mode, 0, 0, 0)

	pivot_offset = len(data)
	data.extend(pivot_bytes)
	struct.pack_into("<QQ", data, descriptor_offset, 3072, pivot_offset)

	struct.pack_into("<Q", data, 0x48, marker_offset)
	struct.pack_into("<Q", data, 0x68, descriptor_offset)
	flag = int(re_mesh.fileHeader.contentFlag.bitFlag) | 0x800C
	struct.pack_into("<H", data, 0x16, flag & 0xFFFF)
	struct.pack_into("<I", data, 0x08, len(data))

	re_mesh.fileHeader.contentFlag.bitFlag = flag
	re_mesh.fileHeader.contentFlag.parseBitFlag()
	re_mesh.fileHeader.normalRecalcOffset = marker_offset
	re_mesh.fileHeader.floatsOffset = descriptor_offset
	re_mesh.fileHeader.fileSize = len(data)
	return bytes(data)


def write_mhwilds_mesh(re_mesh, filepath):
	"""Serialize the generated stream regions inside the main mesh file."""
	print("Writing unified MHWs mesh to " + filepath)
	try:
		mesh_version = int(os.path.splitext(filepath)[1].replace(".", ""))
	except Exception:
		mesh_version = 241111606
	version = mesh_format.meshFileVersionToNewVersionDict.get(
		mesh_version,
		mesh_format.getNearestRemapVersion(mesh_version),
	)
	if version != mesh_format.VERSION_MHWILDS:
		raise RuntimeError("Unified Wilds writer received another mesh format")
	re_mesh.meshVersion = mesh_version
	re_mesh.meshBufferHeader._mhwildsOwner = re_mesh
	original_write = mesh_format.MeshBufferHeader.write
	mesh_format.MeshBufferHeader.write = _write_generated_mesh_buffer
	try:
		buffer = BytesIO()
		re_mesh.write(buffer, version)
		data = buffer.getvalue()
	finally:
		mesh_format.MeshBufferHeader.write = original_write
		del re_mesh.meshBufferHeader._mhwildsOwner
	if len(data) != re_mesh._mhwildsInlineEnd:
		raise RuntimeError(
			f"Unified Wilds size mismatch "
			f"{len(data)}/{re_mesh._mhwildsInlineEnd}"
		)
	data = _apply_normal_contract(data, re_mesh)
	with open(filepath, "wb") as output:
		output.write(data)

	# Verify all three serialized canonical boundaries.
	with open(filepath, "rb") as verification_file:
		verification_data = verification_file.read()
	mesh_header_offset = int(re_mesh.fileHeader.meshOffset)
	written_low, written_high = struct.unpack_from(
		"<hh", verification_data, mesh_header_offset + 44
	)
	written_boundary0 = (
		(int(written_low) & 0xFFFF)
		| ((int(written_high) & 0xFFFF) << 16)
	)
	packed_boundaries = struct.unpack_from(
		"<Q", verification_data, mesh_header_offset + 48
	)[0]
	written_face_table_end = int(packed_boundaries & 0xFFFFFFFF)
	written_payload_start = int(packed_boundaries >> 32)
	if written_face_table_end % 16 or written_payload_start % 16:
		raise RuntimeError(
			"canonical boundaries are not 16-byte aligned: "
			f"0x{written_face_table_end:X}/0x{written_payload_start:X}"
		)
	expected_boundary0 = int(re_mesh._mhwildsCanonicalBoundary0)
	if written_boundary0 != expected_boundary0:
		raise RuntimeError(
			"Serialized first canonical boundary mismatch: "
			f"0x{written_boundary0:08X}/0x{expected_boundary0:08X}"
		)
	if not (
		written_boundary0
		<= written_face_table_end
		<= written_payload_start
		<= int(re_mesh.meshBufferHeader.vertexBufferSize)
	):
		raise RuntimeError(
			"Serialized canonical boundaries are not monotonic: "
			f"0x{written_boundary0:X}/"
			f"0x{written_face_table_end:X}/"
			f"0x{written_payload_start:X}/"
			f"0x{int(re_mesh.meshBufferHeader.vertexBufferSize):X}"
		)
	del re_mesh._mhwildsCanonicalBoundary0

	# Verify vanilla marker placement plus the already validated pivot package.
	written_content_flag = struct.unpack_from("<H", verification_data, 0x16)[0]
	written_normal_offset = struct.unpack_from("<Q", verification_data, 0x48)[0]
	written_blend_offset = struct.unpack_from("<Q", verification_data, 0x50)[0]
	written_floats_offset = struct.unpack_from("<Q", verification_data, 0x68)[0]
	descriptor_offset = int(
		getattr(re_mesh, "_mhwildsFloatDescriptorOffset", 0)
	)
	marker_offset = int(
		getattr(re_mesh, "_mhwildsNormalMarkerOffset", 0)
	)
	mode = _export_mode(re_mesh=re_mesh)

	if not (written_content_flag & 0x0008):
		raise RuntimeError("FloatData content-flag bit was not set")
	if written_normal_offset != marker_offset or marker_offset <= 0:
		raise RuntimeError(
			"normal marker offset mismatch: "
			f"0x{written_normal_offset:X}/0x{marker_offset:X}"
		)
	if written_blend_offset != marker_offset + 16:
		raise RuntimeError(
			"marker is not immediately before blendShapesOffset: "
			f"0x{written_normal_offset:X}/0x{written_blend_offset:X}"
		)
	marker = struct.unpack_from("<IIII", verification_data, marker_offset)
	expected_marker = (mode, 0, 0, 0)
	if marker != expected_marker:
		raise RuntimeError(
			f"marker mismatch for mode {mode}: "
			f"{marker}/{expected_marker}"
		)
	if written_floats_offset != descriptor_offset or descriptor_offset <= 0:
		raise RuntimeError(
			"FloatData offset mismatch: "
			f"0x{written_floats_offset:X}/0x{descriptor_offset:X}"
		)

	descriptor_size, pivot_offset = struct.unpack_from(
		"<QQ", verification_data, descriptor_offset
	)
	if descriptor_size != 3072:
		raise RuntimeError(
			f"FloatData size mismatch: {descriptor_size}/3072"
		)
	if pivot_offset != int(re_mesh._mhwildsInlineEnd):
		raise RuntimeError(
			"pivot offset mismatch: "
			f"0x{pivot_offset:X}/0x{int(re_mesh._mhwildsInlineEnd):X}"
		)
	if len(verification_data) != int(re_mesh._mhwildsInlineEnd) + 3072:
		raise RuntimeError(
			"final size does not contain exactly one 3072-byte pivot table"
		)

	plan = getattr(re_mesh, "_mhwildsNormalContract", None)
	if plan is None:
		raise RuntimeError("generated no normal/pivot plan")
	normal_start = (
		int(re_mesh.meshBufferHeader.vertexBufferOffset)
		+ int(plan["normalOffset"])
	)
	for row, expected in enumerate(plan["groupLane"]):
		actual = verification_data[
			normal_start + row * int(plan["normalStride"]) + 3
		]
		if actual != int(expected):
			raise RuntimeError(
				"Normal.w lane verification failed at row "
				f"{row}: {actual}/{int(expected)}"
			)

	position_start = (
		int(re_mesh.meshBufferHeader.vertexBufferOffset)
		+ int(plan["positionOffset"])
	)
	expected_pivots = bytearray()
	for row in plan["pivotIndices"]:
		start = position_start + int(row) * int(plan["positionStride"])
		expected_pivots.extend(verification_data[start:start + 12])
	actual_pivots = verification_data[pivot_offset:pivot_offset + 3072]
	if actual_pivots != bytes(expected_pivots):
		raise RuntimeError("serialized pivot table verification failed")

	# Verify logical aliases do not consume another shape-remap range.
	plan = getattr(re_mesh, "_blendShapeExportPlan", {})
	expected_remap_bases = list(plan.get("bodyRemapBases", ()))
	expected_aliases = dict(plan.get("aliases", {}))
	blend_header_offset = struct.unpack_from(
		"<Q", verification_data, 0x50
	)[0]
	serialized_body_count = struct.unpack_from(
		"<Q", verification_data, blend_header_offset
	)[0]
	if serialized_body_count != len(expected_remap_bases):
		raise RuntimeError(
			"serialized blend body count mismatch: "
			f"{serialized_body_count}/{len(expected_remap_bases)}"
		)
	serialized_body_offsets = [
		struct.unpack_from(
			"<Q",
			verification_data,
			blend_header_offset + 32 + index * 8,
		)[0]
		for index in range(serialized_body_count)
	]
	for target, source in expected_aliases.items():
		if serialized_body_offsets[target] != serialized_body_offsets[source]:
			raise RuntimeError(
				"serialized blend alias mismatch: "
				f"LOD{target}->LOD{source}"
			)
	for body_index, (body_offset, expected_base) in enumerate(zip(
		serialized_body_offsets, expected_remap_bases
	)):
		serialized_flag = struct.unpack_from(
			"<I", verification_data, body_offset + 4
		)[0]
		serialized_data_offset = struct.unpack_from(
			"<Q", verification_data, body_offset + 16
		)[0]
		serialized_target_base = struct.unpack_from(
			"<H", verification_data, serialized_data_offset
		)[0]
		if (serialized_flag & 0xFFFF) != int(expected_base):
			raise RuntimeError(
				"body remap flag mismatch at LOD"
				f"{body_index}: {(serialized_flag & 0xFFFF)}/{expected_base}"
			)
		if serialized_target_base != int(expected_base):
			raise RuntimeError(
				"target remap base mismatch at LOD"
				f"{body_index}: {serialized_target_base}/{expected_base}"
			)
	return None
