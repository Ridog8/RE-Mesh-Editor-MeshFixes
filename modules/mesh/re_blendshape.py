import copy
import heapq
import io
import math
import warnings
import os
import struct
import zlib
from collections import Counter
from io import BytesIO
from types import MethodType

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


def set_export_mode(parsed_mesh, mode, game="MHWILDS"):
	if game == "RE9":
		mode = int(mode)
		if mode not in (0, 1):
			raise ValueError(f"RE9 blendshape mode must be 0 or 1, got {mode}")
		parsed_mesh._re9BlendShapeMode = mode
		return mode
	if game != "MHWILDS":
		raise ValueError(f"Unsupported blendshape game: {game}")
	mode = int(mode)
	if mode not in MHWILDS_BLENDSHAPE_MODES:
		raise ValueError(f"MHWILDS blendshape mode must be 0 or 1, got {mode}")
	setattr(parsed_mesh, MHWILDS_BLENDSHAPE_MODE_ATTRIBUTE, mode)
	return mode


def _export_mode(parsed_mesh=None, re_mesh=None):
	"""Use the saved export mode, or mode 0 if none was set."""
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
	"""Leave 16 bytes for the normal marker before blendShapesOffset."""
	marker_offset = _align16(current_offset)
	re_mesh.fileHeader.normalRecalcOffset = marker_offset
	re_mesh._mhwildsNormalMarkerOffset = marker_offset
	return marker_offset + 16


def _list_or_empty(value):
	"""Avoid NumPy's ambiguous truth-value check."""
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
	"""Pack Wilds 11/10/11 deltas. Packed zero is 0x7FEFFBFF."""
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
	"""Group submeshes that use the same shape keys."""
	groups = []
	for order, viscon in enumerate(
		_list_or_empty(getattr(lod, "visconGroupList", None))
	):
		submeshes = []
		for parsed_submesh in _list_or_empty(
			getattr(viscon, "subMeshList", None)
		):
			# Shared LODs still need a blend body, but can reuse the vertex buffer.
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

		# Only include submeshes that have these shape keys.
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


def _target_coverage_signature(target):
	"""Match targets by vertex ranges, not shape-key names."""
	return tuple(sorted(
		(int(entry.subMeshVertexStartIndex), int(entry.vertCount))
		for entry in target.subMeshEntryList
		if int(entry.vertCount) > 0
	))


def build_blend_shape_plan(parsed_mesh, parsed_to_mesh_info, game="MHWILDS"):
	if game == "RE9":
		return _build_re9_plan(parsed_mesh, parsed_to_mesh_info)
	if game != "MHWILDS":
		raise ValueError(f"Unsupported blendshape game: {game}")
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
		first_target_by_coverage = {}

		for group in active_groups:
			shape_names = list(group["shapeNames"])
			target = mesh_format.BlendTarget(mesh_format.VERSION_MHWILDS)
			target.blendSSIndex = body_remap_base + body_channel_count
			target.blendShapeNum = len(shape_names)
			target.unkn2 = 1
			target.subMeshEntryList = []
			for _parsed, mesh_info, vertex_count in group["submeshes"]:
				entry = mesh_format.BlendSubMesh()
				# Match the global vertex starts used by MaterialSubdivision.
				entry.subMeshVertexStartIndex = int(mesh_info.vertexStartIndex)
				entry.vertOffset = 0
				entry.vertCount = int(vertex_count)
				entry.paramUnkn3 = 0
				target.subMeshEntryList.append(entry)
			target.subMeshEntryCount = len(target.subMeshEntryList)
			coverage_signature = _target_coverage_signature(target)
			targetCoverageIndex = first_target_by_coverage.setdefault(
				coverage_signature, len(active_targets)
			)
			if targetCoverageIndex > 0xFFFF:
				raise RuntimeError("Wilds target coverage index exceeds 16-bit range")
			target.targetCoverageIndex = targetCoverageIndex

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
			target = mesh_format.BlendTarget(mesh_format.VERSION_MHWILDS)
			target.blendSSIndex = 0
			target.blendShapeNum = 0
			target.targetCoverageIndex = 0
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
	# Track Blender vertices when building the normal tables.
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
	"""Assign file offsets to the blendshape header and its tables."""
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



def _reorder_overflowing_canonical_faces(face_buffer, intervals, canonical, index_stride):
	"""Fit each 1536-index face window into 1024 vertex lanes."""
	index_format = "<3I" if index_stride == 4 else "<3H"
	changed_intervals = 0
	for start, end, vertex_start in intervals:
		count = end - start
		if count < 3:
			continue
		if count % 3:
			raise RuntimeError("Wilds canonical face interval is not triangle-aligned")
		triangles = []
		canonical_triangles = []
		for coordinate in range(start, end, 3):
			triangle = struct.unpack_from(index_format, face_buffer, coordinate * index_stride)
			triangles.append(triangle)
			canonical_triangles.append(tuple(
				canonical[vertex_start + index]
				if 0 <= vertex_start + index < len(canonical)
				else vertex_start + index
				for index in triangle
			))

		def window_unique_count(order, window_start):
			return len({
				value
				for triangle_index in order[window_start:window_start + 512]
				for value in canonical_triangles[triangle_index]
			})

		original_order = range(len(triangles))
		if all(
			window_unique_count(original_order, window_start) <= 1024
			for window_start in range(0, len(triangles), 512)
		):
			continue

		adjacent_triangles = {}
		for triangle_index, vertices in enumerate(canonical_triangles):
			for vertex in set(vertices):
				adjacent_triangles.setdefault(vertex, []).append(triangle_index)

		remaining = bytearray(b"\x01") * len(triangles)
		order = []
		first_remaining = 0
		while len(order) < len(triangles):
			window_vertices = set()
			neighbor_scores = {}
			candidates = []
			for _ in range(min(512, len(triangles) - len(order))):
				chosen = None
				while candidates:
					negative_score, triangle_index = heapq.heappop(candidates)
					if remaining[triangle_index] and -negative_score == neighbor_scores.get(triangle_index, 0):
						chosen = triangle_index
						break
				if chosen is None:
					while first_remaining < len(triangles) and not remaining[first_remaining]:
						first_remaining += 1
					chosen = first_remaining
				remaining[chosen] = 0
				order.append(chosen)
				for vertex in set(canonical_triangles[chosen]) - window_vertices:
					window_vertices.add(vertex)
					for neighbor in adjacent_triangles[vertex]:
						if remaining[neighbor]:
							score = neighbor_scores.get(neighbor, 0) + 1
							neighbor_scores[neighbor] = score
							heapq.heappush(candidates, (-score, neighbor))
			if len(window_vertices) > 1024:
				warnings.warn(
					f"Wilds canonical triangle ordering could not fit face interval "
					f"[{start}, {end}) into 1024 lanes per 1536-index window "
					f"({len(window_vertices)} unique IDs in one attempted window). "
					"Original triangle order retained; shading may be incorrect.",
					RuntimeWarning,
					stacklevel=2,
				)
				break
		else:
			for triangle_index, source_index in enumerate(order):
				struct.pack_into(
					index_format, face_buffer,
					(start + 3 * triangle_index) * index_stride,
					*triangles[source_index],
				)
			changed_intervals += 1
	if changed_intervals:
		print(f"Wilds canonical face ordering: reordered {changed_intervals} overflowing intervals")


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
		# Use Blender vertex IDs for rows split at UV/normal seams.
		# For 1:1 exports, match coincident rows by position and normal.
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
	_reorder_overflowing_canonical_faces(local_faces, intervals, canonical, index_stride)
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
	"""Choose a repeatable starting group; Python's hash() changes between runs."""
	if not palette:
		return 0
	crc = zlib.crc32(preferred_bytes) & 0xFFFFFFFF
	return int(palette[crc % len(palette)])


def _color_blender_submesh(vertex_positions, face_list, source_indices, palette):
	"""Give adjacent vertices different normal-recalculation groups."""
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
	"""Build normal groups and 256 pivots from Blender geometry.

	Imported group/pivot attributes are only used for comparison.
	"""
	lods = _list_or_empty(getattr(parsed_mesh, "mainMeshLODList", None))
	if not lods:
		return None

	# Use the exporter's submesh order. Shared LODs don't add vertices.
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

	# Groups 0 and 255 are reserved.
	# Use up to 254 groups, limited by LOD0's blendshape vertex count.
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

	# Higher LODs may only use groups with a LOD0 pivot.
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

	# Pick pivots from LOD0. Unused slots reuse pivot 0.
	pivot_indices = [0] * 256
	if len(positions) == 0:
		raise RuntimeError("Generated Wilds pivot table has no LOD0 positions")
	# Set the two special pivots from Blender geometry.
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

	# Compare imported attributes for logging only; they don't affect export.
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
	"""Collect the shared VBI0 vertex and face ranges."""
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
	"""Set delta offsets without storing shared LOD payloads twice."""
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


def _finalize_wilds_export(parsed_mesh, re_mesh):
	"""Use the exported vertex and face ranges for Wilds blendshapes."""
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

	# Keep the exporter's existing vertex/face ranges in VBI0.
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

	# Point shared LODs at their owner. Keep the physical records for re-import.
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

	# Target vertex ranges use VBI0 indices; vertOffset addresses delta rows.
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
		face_buffer = bytearray(mesh_header.faceBuffer)
		face_table, vertex_table = _canonical_tables(
			re_mesh,
			0,
			face_buffer,
			visual,
			elements,
			global_range,
			source_vertex_indices=plan.get("sourceVertexIndices"),
			source_identity_reliable=plan.get("sourceIdentityReliable"),
		)
		mesh_header.faceBuffer = face_buffer
	else:
		face_table, vertex_table = b"", b""

	# The canonical face table starts on a 16-byte boundary.
	# Add padding when the vertex buffer doesn't end there.
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

	# The two int16 fields at MeshBufferHeader + 44 store boundary 0.
	# The next uint64 stores boundaries 1 and 2.
	canonical_boundary0 = int(canonical_start)
	mesh_header.vertexElementSize = _signed_short(
		canonical_boundary0 & 0xFFFF
	)
	mesh_header.unkn1 = _signed_short(
		(canonical_boundary0 >> 16) & 0xFFFF
	)
	re_mesh._mhwildsCanonicalBoundary0 = canonical_boundary0

	# Keep boundaries 1 and 2 packed in sunbreakSecondUnknown.
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

	# Build normal groups and pivots from Blender, not imported attributes.
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
	# Reuse the zero fields at meshOffset + 64 for empty StreamingInfo.
	re_mesh.fileHeader.streamingInfoOffset = mesh_offset + 64
	re_mesh.fileHeader.normalRecalcOffset = marker_offset
	re_mesh.fileHeader.floatsOffset = int(
		getattr(re_mesh, "_mhwildsFloatDescriptorOffset", 0)
	)
	re_mesh.fileHeader.fileSize = inline_end
	# Enable the normal marker and pivot data.
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
	"""Write the normal groups, pivots, and mode 0/1 marker."""
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

	# Check that nothing else has written into the reserved marker slot.
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


def _write_wilds_mesh(re_mesh, filepath):
	"""Write all mesh data into a single file."""
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

	# Check the three canonical-table boundaries.
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

	# Check the normal marker and pivot data.
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

	# Shared LODs must reuse their owner's shape-remap range.
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


RE9_INTERNAL_VERSION = 250904410
RE9_SOURCE_VERTEX_ATTRIBUTE = "_RE9_ExportSourceVertex"


def read_re9_header(file):
    position = file.tell()
    file.seek(0, 2)
    length = file.tell()

    def unpack(offset, fmt, count=1):
        size = struct.calcsize(fmt)
        if offset < 0 or count < 0 or offset + size * count > length:
            raise ValueError("RE9 blendshape metadata exceeds the mesh file")
        file.seek(offset)
        return [struct.unpack(fmt, file.read(size)) for _ in range(count)]

    header = mesh_format.BlendShapeHeader()
    header.count, header.zero, header.mainOffset, header.hash = unpack(position, "<4Q")[0]
    if header.count > 256:
        raise ValueError("Invalid RE9 blendshape LOD count")
    header.blendShapeOffsetList = [row[0] for row in unpack(header.mainOffset, "<Q", header.count)]
    for offset in header.blendShapeOffsetList:
        body = mesh_format.BlendShapeData()
        (body.targetCount, body.typing, body.unknFlag, body.padding1, body.padding2,
         body.dataOffset, body.aabbOffset, body.blendSOffset, body.blendSSOffset) = unpack(offset, "<HHIII4Q")[0]
        body.blendTargetList = []
        for record in unpack(body.dataOffset, "<HHHBBQ", body.targetCount + body.typing):
            target = mesh_format.BlendTarget(mesh_format.VERSION_RE9)
            (target.blendSSIndex, target.blendShapeNum, target.unkn0, target.subMeshEntryCount,
             target.unkn2, target.subMeshEntryOffset) = record
            for values in unpack(target.subMeshEntryOffset, "<4I", target.subMeshEntryCount):
                entry = mesh_format.BlendSubMesh()
                entry.subMeshVertexStartIndex, entry.vertOffset, entry.vertCount, entry.paramUnkn3 = values
                target.subMeshEntryList.append(entry)
            body.blendTargetList.append(target)
        body.aabbList = []
        for values in unpack(body.aabbOffset, "<8f", body.targetCount):
            if not np.all(np.isfinite(values)):
                raise ValueError("Non-finite RE9 blendshape bounds")
            box = mesh_format.AABB()
            box.min.x, box.min.y, box.min.z, box.min.w = values[:4]
            box.max.x, box.max.y, box.max.z, box.max.w = values[4:]
            body.aabbList.append(box)
        body.blendS = list(unpack(body.blendSOffset, "<3i")[0])
        body.blendSSList = [row[0] for row in unpack(body.blendSSOffset, "<i", body.unknFlag >> 16)]
        header.blendShapeList.append(body)
    file.seek(position)
    return header


def _decode_re9_rows(data, box):
    if len(data) % 4:
        raise ValueError("Incomplete RE9 blendshape delta row")
    packed = np.frombuffer(data, dtype="<u4")
    result = np.empty((len(packed), 3), dtype=np.float32)
    result[:, 0] = ((packed & 2047).astype(np.float32) - 1023.0) / 1023.0
    result[:, 1] = (((packed >> 11) & 1023).astype(np.float32) - 511.0) / 511.0
    result[:, 2] = (((packed >> 21) & 2047).astype(np.float32) - 1023.0) / 1023.0
    scale = np.array([max(abs(getattr(box.min, axis)), abs(getattr(box.max, axis))) for axis in ("x", "y", "z")], dtype=np.float32)
    result = np.clip(result, -1.0, 1.0) * scale
    result[packed == 0x7FEFFBFF] = 0.0
    return result


def apply_re9_imported_shapes(raw, target_lods, lods, payloads):
    from .re_mesh_parse import BlendShape

    header = raw.blendShapeHeader
    if header is None:
        return lods
    source_lods = raw.lodHeader.lodGroupList
    for raw_lod, lod in zip(target_lods, lods):
        lod_index = next((i for i, entry in enumerate(source_lods) if entry is raw_lod), None)
        if lod_index is None or lod_index >= len(header.blendShapeList):
            continue
        body = header.blendShapeList[lod_index]
        records = []
        for raw_group, group in zip(raw_lod.meshGroupList, lod.visconGroupList):
            for info, submesh in zip(raw_group.vertexInfoList, group.subMeshList):
                records.append((info, submesh))
        for target in body.blendTargetList[:body.targetCount]:
            if not target.blendShapeNum:
                continue
            if target.unkn0 >= len(body.aabbList):
                raise ValueError("RE9 blendshape target references missing bounds")
            box = body.aabbList[target.unkn0]
            row_count = sum(entry.vertCount for entry in target.subMeshEntryList)
            for entry in target.subMeshEntryList:
                matches = []
                for info, submesh in records:
                    start = int(info.vertexStartIndex)
                    end = start + len(submesh.vertexPosList)
                    lo = max(start, entry.subMeshVertexStartIndex)
                    hi = min(end, entry.subMeshVertexStartIndex + entry.vertCount)
                    if hi > lo:
                        matches.append((info, submesh, lo, hi))
                banks = {int(info.vertexBufferIndex) for info, _submesh, _lo, _hi in matches}
                if not matches or len(banks) != 1:
                    raise ValueError("RE9 blendshape range does not identify one vertex bank")
                payload = payloads[next(iter(banks))]
                for channel in range(target.blendShapeNum):
                    remap = target.blendSSIndex + channel
                    if remap >= len(raw.blendShapeNameRemapList):
                        raise ValueError("RE9 blendshape name remap is out of range")
                    name = raw.rawNameList[raw.blendShapeNameRemapList[remap]]
                    row = int(entry.vertOffset) + channel * row_count
                    encoded = payload[row * 4:(row + entry.vertCount) * 4]
                    if len(encoded) != entry.vertCount * 4:
                        raise ValueError(f"Incomplete RE9 blendshape data for {name}")
                    deltas = _decode_re9_rows(encoded, box)
                    for info, submesh, lo, hi in matches:
                        shape = next((shape for shape in submesh.blendShapeList if shape.blendShapeName == name), None)
                        if shape is None:
                            shape = BlendShape()
                            shape.blendShapeName = name
                            shape.deltas = np.zeros((len(submesh.vertexPosList), 3), dtype=np.float32)
                            submesh.blendShapeList.append(shape)
                        shape.deltas[lo - info.vertexStartIndex:hi - info.vertexStartIndex] += deltas[lo - entry.subMeshVertexStartIndex:hi - entry.subMeshVertexStartIndex]
    return lods


def _pack_re9_rows(deltas, box):
    deltas = np.asarray(deltas, dtype=np.float32).reshape((-1, 3))
    scale = np.array((box.max.x, box.max.y, box.max.z), dtype=np.float32)
    normalized = np.divide(deltas, scale, out=np.zeros_like(deltas), where=scale != 0)
    bias = np.array((1023, 511, 1023), dtype=np.float32)
    packed = np.rint(bias + np.clip(normalized, -1.0, 1.0) * bias).astype(np.uint32)
    return ((packed[:, 0] & 2047) | ((packed[:, 1] & 1023) << 11) | ((packed[:, 2] & 2047) << 21)).astype("<u4").tobytes()


def _build_re9_plan(parsed, mesh_infos):
    mode = set_export_mode(parsed, getattr(parsed, "_re9BlendShapeMode", 1), game="RE9")
    lods = parsed.mainMeshLODList
    authored = [i for i, lod in enumerate(lods) if any(shape.blendShapeName and not shape.blendShapeName.startswith("DUMMY_") for group in lod.visconGroupList for submesh in group.subMeshList for shape in submesh.blendShapeList)]
    if not authored:
        return None
    header = mesh_format.BlendShapeHeader()
    names = []
    payload = bytearray()
    for lod in lods[:max(authored) + 1]:
        body = mesh_format.BlendShapeData()
        body.aabbList = []
        body.blendTargetList = []
        body.blendS = [0, 0, 0]
        local_names = []
        signature_offsets = {}
        inactive = []
        for group in lod.visconGroupList:
            partitions = {}
            for submesh in group.subMeshList:
                signature = tuple(shape.blendShapeName for shape in submesh.blendShapeList if shape.blendShapeName and not shape.blendShapeName.startswith("DUMMY_"))
                if len(signature) != len(set(signature)):
                    raise ValueError("Duplicate RE9 blendshape names on one submesh")
                partitions.setdefault(signature, []).append(submesh)
            for signature, submeshes in partitions.items():
                target = mesh_format.BlendTarget(mesh_format.VERSION_RE9)
                if signature and signature not in signature_offsets:
                    signature_offsets[signature] = len(names) + len(local_names)
                    local_names.extend(signature)
                target.blendSSIndex = signature_offsets[signature] if signature else 0
                target.blendShapeNum = len(signature)
                target.unkn0 = len(body.aabbList) if signature else 0
                target.unkn2 = int(bool(signature))
                arrays = []
                start_row = len(payload) // 4
                running = 0
                for submesh in submeshes:
                    info = mesh_infos[submesh]
                    entry = mesh_format.BlendSubMesh()
                    entry.subMeshVertexStartIndex = int(info.vertexStartIndex)
                    entry.vertOffset = start_row + running if signature else 0
                    entry.vertCount = len(submesh.vertexPosList)
                    if not 0 <= group.visconGroupNum <= 255 or not 0 <= submesh.materialIndex <= 255:
                        raise ValueError("RE9 blendshape group/material ID exceeds 8-bit range")
                    entry.paramUnkn3 = (int(submesh.materialIndex) << 8) | int(group.visconGroupNum)
                    target.subMeshEntryList.append(entry)
                    running += entry.vertCount
                if len(target.subMeshEntryList) > 255:
                    raise ValueError("RE9 blendshape target exceeds 255 submesh ranges")
                target.subMeshEntryCount = len(target.subMeshEntryList)
                if signature:
                    for name in signature:
                        chunks = []
                        for submesh in submeshes:
                            shape = next(shape for shape in submesh.blendShapeList if shape.blendShapeName == name)
                            delta = np.asarray(shape.deltas, dtype=np.float32).reshape((-1, 3))
                            if len(delta) != len(submesh.vertexPosList) or not np.all(np.isfinite(delta)):
                                raise ValueError(f"Invalid RE9 blendshape rows for {name}")
                            chunks.append(delta)
                        arrays.append(np.concatenate(chunks))
                    maximum = np.max(np.abs(np.concatenate(arrays)), axis=0)
                    box = mesh_format.AABB()
                    box.min.x, box.min.y, box.min.z = map(float, -maximum)
                    box.max.x, box.max.y, box.max.z = map(float, maximum)
                    box.min.w = box.max.w = 0.0
                    body.aabbList.append(box)
                    for array in arrays:
                        payload.extend(_pack_re9_rows(array, box))
                    body.blendTargetList.append(target)
                else:
                    inactive.append(target)
        if len(names) + len(local_names) > 65535 or len(body.blendTargetList) > 65535:
            raise ValueError("RE9 blendshape channel or target count exceeds 16-bit range")
        body.targetCount = len(body.blendTargetList)
        body.typing = len(inactive)
        body.blendTargetList.extend(inactive)
        body.unknFlag = (len(local_names) << 16) | len(names)
        body.blendSSList = list(range(len(local_names)))
        names.extend(local_names)
        header.blendShapeList.append(body)
    header.count = len(header.blendShapeList)
    return dict(header=header, shapeRemapNames=names, payload=bytes(payload), mode=mode)


def reserve_re9_metadata(raw, plan, offset):
    offset = mesh_format.getPaddedPos(offset, 16)
    raw._re9NormalMarkerOffset = offset
    set_export_mode(raw, plan["mode"], game="RE9")
    raw.fileHeader.normalRecalcOffset = offset
    raw.fileHeader.blendShapesOffset = offset + 16
    raw.blendShapeHeader = plan["header"]
    raw._re9BlendShapePlan = plan
    return relocate_blend_shape_header(raw.blendShapeHeader, offset + 16, mesh_format.VERSION_RE9)


def _finalize_re9_export(parsed, raw):
    bank = raw.meshBufferHeader
    plan = raw._re9BlendShapePlan
    geometry = bytearray(bank.vertexBuffer)
    faces = bytearray(bank.faceBuffer)
    normal_plan = _normal_contract(parsed)
    position_element = next(entry for entry in bank.vertexElementList if entry.typing == 0)
    normal_element = next(entry for entry in bank.vertexElementList if entry.typing == 1)
    source_rows = [-1] * normal_plan["vertexCount"]
    reliable = [False] * len(source_rows)
    row = 0
    for lod in parsed.mainMeshLODList:
        for group in lod.visconGroupList:
            for submesh in group.subMeshList:
                if submesh.isReusedMesh:
                    continue
                count = len(submesh.vertexPosList)
                indices = getattr(submesh, "blendShapeSourceVertexIndexList", [])
                if len(indices) == count:
                    source_rows[row:row + count] = list(indices)
                    reliable[row:row + count] = [bool(getattr(submesh, "blendShapeSplitLoopVertices", False))] * count
                row += count
    if row != len(source_rows):
        raise ValueError("RE9 normal-contract vertex count differs from exported geometry")
    canonical_range = _global_resource_range(raw)
    canonical_lods = raw.lodHeader.lodGroupList[:len(raw.blendShapeHeader.blendShapeList)]
    canonical_range["lod"].meshGroupList = [group for lod in canonical_lods for group in lod.meshGroupList]
    canonical_range["vertexCount"] = sum(len(submesh.vertexPosList) for lod in parsed.mainMeshLODList[:len(canonical_lods)]
                                         for group in lod.visconGroupList for submesh in group.subMeshList if not submesh.isReusedMesh)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            table_faces, table_vertices = _canonical_tables(raw, 0, faces, geometry, bank.vertexElementList,
                                                            canonical_range, source_rows, reliable)
        except RuntimeWarning as error:
            raise ValueError("RE9 normal-table window exceeds its lane limit; split or reduce the affected submesh") from error
    visual_end = mesh_format.getPaddedPos(len(geometry), 16)
    geometry.extend(bytes(visual_end - len(geometry)))
    geometry.extend(table_faces)
    vertex_table_end = len(geometry) + len(table_vertices)
    geometry.extend(table_vertices)
    geometry.extend(plan["payload"])
    geometry.extend(bytes(mesh_format.getPaddingAmount(len(geometry), 16)))
    for index, value in enumerate(normal_plan["groupLane"]):
        geometry[normal_element.posStartOffset + index * normal_element.stride + 3] = value
    pivots = bytearray()
    for index in normal_plan["pivotIndices"]:
        start = position_element.posStartOffset + int(index) * position_element.stride
        pivots.extend(geometry[start:start + 12])
    bank.vertexBuffer = geometry
    bank.faceBuffer = faces
    bank.vertexBufferSize = len(geometry)
    bank.faceBufferSize = len(faces)
    bank.block2FaceBufferOffset = len(geometry) + len(faces)
    bank.NULL = bank.block2FaceBufferOffset
    bank.totalBufferSize = mesh_format.getPaddedPos(bank.block2FaceBufferOffset, 16)
    low = visual_end & 65535
    high = visual_end >> 16
    bank.vertexElementSize = low if low < 32768 else low - 65536
    bank.unkn1 = high if high < 32768 else high - 65536
    bank.sunbreakSecondUnknown = (vertex_table_end << 32) | (visual_end + len(table_faces))
    bank.sunbreakOffset = bank.sf6unkn0 = bank.streamingVertexElementOffset = bank.sf6unkn2 = 0
    raw.fileHeader.floatsOffset = raw.fileHeader.meshOffset + 96
    bank.vertexElementOffset = raw.fileHeader.floatsOffset + 16
    bank.vertexBufferOffset = mesh_format.getPaddedPos(bank.vertexElementOffset + len(bank.vertexElementList) * 8, 16)
    bank.faceBufferOffset = bank.vertexBufferOffset + bank.vertexBufferSize
    raw.fileHeader.verticesOffset = bank.vertexBufferOffset
    raw.fileHeader.streamingInfoOffset = raw.fileHeader.meshOffset + 80
    raw._re9PivotOffset = bank.vertexBufferOffset + bank.totalBufferSize
    raw._re9PivotBytes = bytes(pivots)
    raw.fileHeader.fileSize = raw._re9PivotOffset + len(pivots)
    raw.fileHeader.contentFlag.bitFlag |= 0x800C
    raw.fileHeader.contentFlag.parseBitFlag()
    raw._blendShapeWriter = "RE9"


def _write_mesh_buffer(bank, file, version):
    raw = bank._re9Owner
    file.write(struct.pack("<QQQIIHHQQIIhhQQQQ", bank.vertexElementOffset, bank.vertexBufferOffset, bank.sunbreakOffset,
                           bank.totalBufferSize, bank.vertexBufferSize, bank.mainVertexElementCount, bank.vertexElementCount,
                           bank.prag_unknOffset0, bank.prag_unknOffset1, bank.block2FaceBufferOffset, bank.NULL,
                           bank.vertexElementSize, bank.unkn1, bank.sunbreakSecondUnknown, bank.sf6unkn0,
                           bank.streamingVertexElementOffset, bank.sf6unkn2))
    file.seek(raw.fileHeader.floatsOffset)
    file.write(struct.pack("<QQ", len(raw._re9PivotBytes), raw._re9PivotOffset))
    file.seek(bank.vertexElementOffset)
    for element in bank.vertexElementList:
        element.write(file)
    file.seek(bank.vertexBufferOffset)
    file.write(bank.vertexBuffer)
    file.seek(bank.faceBufferOffset)
    file.write(bank.faceBuffer)
    file.seek(raw._re9PivotOffset)
    file.write(raw._re9PivotBytes)


def _write_re9_mesh(raw, path):
    if raw.fileHeader.version != RE9_INTERNAL_VERSION:
        raise ValueError("RE9 blendshape writer received another game version")
    bank = raw.meshBufferHeader
    bank._re9Owner = raw
    bank.write = MethodType(_write_mesh_buffer, bank)
    try:
        buffer = io.BytesIO()
        raw.write(buffer, mesh_format.VERSION_RE9)
        data = buffer.getvalue()
    finally:
        del bank.write
        del bank._re9Owner
    if len(data) != raw.fileHeader.fileSize:
        raise ValueError("RE9 unified blendshape file size does not match EOF")
    if struct.unpack_from("<I", data, 8)[0] != len(data):
        raise ValueError("RE9 serialized file size does not match EOF")
    if struct.unpack_from("<4I", data, raw.fileHeader.normalRecalcOffset) != (raw._re9BlendShapeMode, 0, 0, 0):
        raise ValueError("RE9 normal recalculation marker is invalid")
    if raw.fileHeader.blendShapesOffset != raw.fileHeader.normalRecalcOffset + 16:
        raise ValueError("RE9 normal marker is not adjacent to the blendshape header")
    if raw.streamingInfoHeader is not None and raw.streamingInfoHeader.entryCount:
        raise ValueError("RE9 blendshape export must be unified")
    with open(path, "wb") as output:
        output.write(data)


def build_re9_blender_shapes(obj, source_indices, transform, evaluated_basis, enabled):
    if not enabled or obj.data.shape_keys is None:
        return []
    import bpy
    from .re_mesh_parse import BlendShape

    keys = obj.data.shape_keys
    if not keys.use_relative:
        raise ValueError("RE9 blendshape export requires relative Blender shape keys")
    basis = keys.reference_key
    exported = [key for key in keys.key_blocks if key != basis and not key.name.startswith("DUMMY_")]
    if not exported:
        return []
    attribute = evaluated_basis.attributes.get(RE9_SOURCE_VERTEX_ATTRIBUTE)
    if attribute is None or len(attribute.data) != len(evaluated_basis.vertices):
        raise ValueError("RE9 evaluated mesh lost its export vertex provenance")
    mapping = np.empty(len(attribute.data), dtype=np.int32)
    attribute.data.foreach_get("value", mapping)
    mapping = mapping[np.asarray(source_indices, dtype=np.int64)]
    stored = [(key, key.value, key.mute) for key in keys.key_blocks]
    show_only = obj.show_only_shape_key
    result = []
    baseline = None
    shape_mesh = None
    try:
        obj.show_only_shape_key = False
        for key in keys.key_blocks:
            if key != basis:
                key.value = 0.0
                key.mute = False
        obj.data.update()
        bpy.context.view_layer.update()
        baseline = bpy.data.meshes.new_from_object(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()))
        if np.any(mapping < 0) or np.any(mapping >= len(baseline.vertices)):
            raise ValueError("RE9 export vertex provenance references a missing evaluated vertex")
        coordinates = np.empty(len(baseline.vertices) * 3, dtype=np.float32)
        baseline.vertices.foreach_get("co", coordinates)
        coordinates = coordinates.reshape((-1, 3))
        topology = tuple(tuple(face.vertices) for face in baseline.polygons)
        matrix = np.asarray(transform.to_3x3(), dtype=np.float32)
        for key in exported:
            key.value = 1.0
            obj.data.update()
            bpy.context.view_layer.update()
            shape_mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()))
            if len(shape_mesh.vertices) != len(baseline.vertices) or tuple(tuple(face.vertices) for face in shape_mesh.polygons) != topology:
                raise ValueError(f"RE9 shape key {key.name} changes evaluated topology")
            values = np.empty(coordinates.size, dtype=np.float32)
            shape_mesh.vertices.foreach_get("co", values)
            delta = (values.reshape((-1, 3)) - coordinates) @ matrix.T
            entry = BlendShape()
            entry.blendShapeName = key.name
            entry.deltas = delta[mapping].copy()
            result.append(entry)
            bpy.data.meshes.remove(shape_mesh)
            shape_mesh = None
            key.value = 0.0
    finally:
        if shape_mesh is not None:
            bpy.data.meshes.remove(shape_mesh)
        if baseline is not None:
            bpy.data.meshes.remove(baseline)
        for key, value, mute in stored:
            key.value = value
            key.mute = mute
        obj.show_only_shape_key = show_only
        obj.data.update()
        bpy.context.view_layer.update()
    return result

def finalize_export(parsed_mesh, re_mesh, game="MHWILDS"):
    if game == "RE9":
        return _finalize_re9_export(parsed_mesh, re_mesh)
    if game == "MHWILDS":
        return _finalize_wilds_export(parsed_mesh, re_mesh)
    raise ValueError(f"Unsupported blendshape game: {game}")


def write_mesh(re_mesh, filepath, game="MHWILDS"):
    if game == "RE9":
        return _write_re9_mesh(re_mesh, filepath)
    if game == "MHWILDS":
        return _write_wilds_mesh(re_mesh, filepath)
    raise ValueError(f"Unsupported blendshape game: {game}")
