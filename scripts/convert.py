#!/usr/bin/env python3
"""
convert.py — Convert PlayCanvas compressed PLY to standard Gaussian splat PLY.

Pure stdlib Python. No external dependencies.

Usage: python3 convert.py <input.compressed.ply> <output.ply>

Exit codes:
    0 — success
    1 — error
    2 — usage error
"""

import struct
import sys
import math
import os

# Constants
SH_C0 = 0.28209479177387814
SQRT2 = math.sqrt(2.0)
CHUNK_SIZE = 256

# Property type sizes (for stride calculation)
TYPE_SIZES = {
    'char': 1, 'uchar': 1, 'int8': 1, 'uint8': 1,
    'short': 2, 'ushort': 2, 'int16': 2, 'uint16': 2,
    'int': 4, 'uint': 4, 'int32': 4, 'uint32': 4,
    'float': 4, 'float32': 4,
    'double': 8, 'float64': 8,
}


def die(msg):
    """Print error and exit with code 1."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def usage():
    """Print usage and exit with code 2."""
    print("Usage: python3 convert.py <input.compressed.ply> <output.ply>", file=sys.stderr)
    sys.exit(2)


def unpack_111011(value):
    """Unpack 11-10-11 bit packed value to (x, y, z) in [0, 1]."""
    x = ((value >> 21) & 0x7FF) / 2047.0
    y = ((value >> 11) & 0x3FF) / 1023.0
    z = (value & 0x7FF) / 2047.0
    return x, y, z


def unpack_8888(value):
    """Unpack 8-8-8-8 bit packed value to (r, g, b, a) in [0, 1]."""
    r = ((value >> 24) & 0xFF) / 255.0
    g = ((value >> 16) & 0xFF) / 255.0
    b = ((value >> 8) & 0xFF) / 255.0
    a = (value & 0xFF) / 255.0
    return r, g, b, a


def unpack_rotation(value):
    """Unpack rotation to quaternion (w, x, y, z), normalized."""
    which = (value >> 30) & 0x3
    a_raw = ((value >> 20) & 0x3FF) / 1023.0
    b_raw = ((value >> 10) & 0x3FF) / 1023.0
    c_raw = (value & 0x3FF) / 1023.0

    # Convert from [0,1] to [-1/sqrt(2), 1/sqrt(2)]
    a = (a_raw - 0.5) * SQRT2
    b = (b_raw - 0.5) * SQRT2
    c = (c_raw - 0.5) * SQRT2

    # Compute the missing component
    m_sq = 1.0 - a*a - b*b - c*c
    m = math.sqrt(max(0.0, m_sq))

    # Insert m at position 'which'
    if which == 0:
        quat = (m, a, b, c)
    elif which == 1:
        quat = (a, m, b, c)
    elif which == 2:
        quat = (a, b, m, c)
    else:
        quat = (a, b, c, m)

    # Normalize (already should be close to unit length, but ensure)
    w, x, y, z = quat
    length = math.sqrt(w*w + x*x + y*y + z*z)
    if length > 0:
        w, x, y, z = w/length, x/length, y/length, z/length

    return w, x, y, z


def parse_ply_header(f):
    """
    Parse PLY header from file.
    Returns: (format, version, elements) where elements is list of (name, count, properties)
             and properties is list of (prop_name, prop_type)
    """
    # Read magic
    line = f.readline()
    if line.strip() != b'ply':
        die("not a PLY file (missing 'ply' magic)")

    format_str = None
    format_version = None
    elements = []
    current_element = None

    while True:
        line = f.readline()
        if not line:
            die("unexpected EOF in PLY header")

        line = line.strip()
        if line == b'end_header':
            break

        parts = line.split()
        if not parts:
            continue

        keyword = parts[0]

        if keyword == b'format':
            if len(parts) < 3:
                die("malformed format line")
            format_str = parts[1].decode('ascii')
            format_version = parts[2].decode('ascii')

        elif keyword == b'element':
            if current_element is not None:
                elements.append(current_element)
            if len(parts) < 3:
                die("malformed element line")
            name = parts[1].decode('ascii')
            try:
                count = int(parts[2])
            except ValueError:
                die(f"invalid element count for '{name}': {parts[2]!r}")
            if count < 0:
                die(f"invalid negative element count for '{name}': {count}")
            current_element = (name, count, [])

        elif keyword == b'property':
            if current_element is None:
                die("property before element")
            if len(parts) < 3:
                die("malformed property line")
            # Handle list properties
            if parts[1] == b'list':
                # list count_type value_type name
                if len(parts) < 5:
                    die("malformed list property line")
                prop_type = ('list', parts[2].decode('ascii'), parts[3].decode('ascii'))
                prop_name = parts[4].decode('ascii')
            else:
                prop_type = parts[1].decode('ascii')
                prop_name = parts[2].decode('ascii')
            current_element[2].append((prop_name, prop_type))

        elif keyword == b'comment':
            pass  # ignore comments

        elif keyword == b'obj_info':
            pass  # ignore obj_info

    if current_element is not None:
        elements.append(current_element)

    if format_str is None:
        die("missing format in PLY header")
    if format_version is None:
        die("missing format version in PLY header")

    return format_str, format_version, elements


def is_compressed_ply(elements):
    """
    Check if this is a PlayCanvas compressed PLY.
    Must have 'chunk' element and 'vertex' element with 'packed_position' property.
    """
    has_chunk = False
    has_packed_vertex = False

    for name, count, props in elements:
        if name == 'chunk':
            has_chunk = True
        elif name == 'vertex':
            prop_names = [p[0] for p in props]
            if 'packed_position' in prop_names:
                has_packed_vertex = True

    return has_chunk and has_packed_vertex


def compute_element_stride(props):
    """Compute byte stride for an element's properties."""
    stride = 0
    for prop_name, prop_type in props:
        if isinstance(prop_type, tuple) and prop_type[0] == 'list':
            die(f"list properties not supported: {prop_name}")
        if prop_type not in TYPE_SIZES:
            die(f"unknown property type: {prop_type}")
        stride += TYPE_SIZES[prop_type]
    return stride


def read_chunks(f, count, props):
    """
    Read chunk element data.
    Returns list of dicts, one per chunk.

    Expected properties (12 or 18):
    - min_x, max_x, min_y, max_y, min_z, max_z (position bbox)
    - min_scale_x, max_scale_x, min_scale_y, max_scale_y, min_scale_z, max_scale_z (scale bbox)
    - [optional] min_r, max_r, min_g, max_g, min_b, max_b (color bbox)
    """
    prop_names = [p[0] for p in props]
    num_props = len(props)
    prop_set = set(prop_names)

    required_bbox_props = {
        'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z',
        'min_scale_x', 'max_scale_x', 'min_scale_y', 'max_scale_y', 'min_scale_z', 'max_scale_z',
    }
    color_bbox_props = {'min_r', 'max_r', 'min_g', 'max_g', 'min_b', 'max_b'}

    missing = sorted(required_bbox_props - prop_set)
    if missing:
        die(f"chunk element missing required properties: {', '.join(missing)}")

    has_any_color_bbox = any(name in prop_set for name in color_bbox_props)
    has_all_color_bbox = all(name in prop_set for name in color_bbox_props)
    if has_any_color_bbox and not has_all_color_bbox:
        missing_color = sorted(color_bbox_props - prop_set)
        die(f"chunk color bbox is incomplete; missing: {', '.join(missing_color)}")

    # All chunk properties should be float32
    for prop_name, prop_type in props:
        if prop_type not in ('float', 'float32'):
            die(f"chunk property '{prop_name}' has unexpected type '{prop_type}' (expected float32)")

    chunks = []
    fmt = f'<{num_props}f'
    row_size = num_props * 4

    for _ in range(count):
        data = f.read(row_size)
        if len(data) < row_size:
            die("unexpected EOF reading chunks")
        values = struct.unpack(fmt, data)
        chunk = dict(zip(prop_names, values))
        chunks.append(chunk)

    return chunks


def read_vertices(f, count, props):
    """
    Read vertex element data.
    Returns list of (packed_position, packed_rotation, packed_scale, packed_color).

    Expected properties (all uint32):
    - packed_position
    - packed_rotation
    - packed_scale
    - packed_color
    """
    prop_names = [p[0] for p in props]

    # Find indices of the packed properties
    required = ['packed_position', 'packed_rotation', 'packed_scale', 'packed_color']
    indices = {}
    for req in required:
        if req not in prop_names:
            die(f"missing required vertex property: {req}")
        indices[req] = prop_names.index(req)

    # All vertex properties should be uint32
    for prop_name, prop_type in props:
        if prop_type not in ('uint', 'uint32'):
            die(f"vertex property '{prop_name}' has unexpected type '{prop_type}' (expected uint32)")

    num_props = len(props)
    fmt = f'<{num_props}I'
    row_size = num_props * 4

    vertices = []
    for _ in range(count):
        data = f.read(row_size)
        if len(data) < row_size:
            die("unexpected EOF reading vertices")
        values = struct.unpack(fmt, data)
        v = (
            values[indices['packed_position']],
            values[indices['packed_rotation']],
            values[indices['packed_scale']],
            values[indices['packed_color']],
        )
        vertices.append(v)

    return vertices


def skip_element(f, count, props):
    """Skip an element by seeking past its data."""
    stride = compute_element_stride(props)
    total = count * stride
    f.seek(total, os.SEEK_CUR)


def write_ply_header(f, vertex_count):
    """Write standard PLY header for Gaussian splats."""
    header = f"""ply
format binary_little_endian 1.0
element vertex {vertex_count}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
    f.write(header.encode('ascii'))


def decode_and_write(f_out, chunks, vertices):
    """
    Decode all vertices and write to output file.
    Output: 14 float32 per vertex (x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3)
    """
    # Check if chunks have color bbox
    has_color_bbox = 'min_r' in chunks[0] if chunks else False

    for i, (packed_pos, packed_rot, packed_scale, packed_color) in enumerate(vertices):
        chunk_idx = i // CHUNK_SIZE
        if chunk_idx >= len(chunks):
            die(f"vertex {i} references out-of-range chunk {chunk_idx}")
        chunk = chunks[chunk_idx]

        # Unpack position (normalized [0,1]) and scale to chunk bbox
        px, py, pz = unpack_111011(packed_pos)
        x = chunk['min_x'] + px * (chunk['max_x'] - chunk['min_x'])
        y = chunk['min_y'] + py * (chunk['max_y'] - chunk['min_y'])
        z = chunk['min_z'] + pz * (chunk['max_z'] - chunk['min_z'])

        # Unpack scale (normalized [0,1]) and scale to chunk bbox
        sx, sy, sz = unpack_111011(packed_scale)
        scale_0 = chunk['min_scale_x'] + sx * (chunk['max_scale_x'] - chunk['min_scale_x'])
        scale_1 = chunk['min_scale_y'] + sy * (chunk['max_scale_y'] - chunk['min_scale_y'])
        scale_2 = chunk['min_scale_z'] + sz * (chunk['max_scale_z'] - chunk['min_scale_z'])

        # Unpack color and opacity
        r, g, b, a = unpack_8888(packed_color)

        # Apply color bbox if present
        if has_color_bbox:
            r = chunk['min_r'] + r * (chunk['max_r'] - chunk['min_r'])
            g = chunk['min_g'] + g * (chunk['max_g'] - chunk['min_g'])
            b = chunk['min_b'] + b * (chunk['max_b'] - chunk['min_b'])

        # Convert color to SH coefficients
        f_dc_0 = (r - 0.5) / SH_C0
        f_dc_1 = (g - 0.5) / SH_C0
        f_dc_2 = (b - 0.5) / SH_C0

        # Convert alpha to opacity (logit)
        # Clamp to avoid log(0) or division by zero
        a_clamped = max(1.0/255.0, min(254.0/255.0, a))
        opacity = -math.log(1.0 / a_clamped - 1.0)

        # Unpack rotation
        rot_0, rot_1, rot_2, rot_3 = unpack_rotation(packed_rot)

        # Write 14 floats
        f_out.write(struct.pack('<14f',
            x, y, z,
            f_dc_0, f_dc_1, f_dc_2,
            opacity,
            scale_0, scale_1, scale_2,
            rot_0, rot_1, rot_2, rot_3
        ))


def convert(input_path, output_path):
    """Main conversion function."""
    with open(input_path, 'rb') as f:
        # Parse header
        format_str, format_version, elements = parse_ply_header(f)

        # Validate format
        if format_str != 'binary_little_endian':
            die(f"unsupported PLY format: {format_str} (expected binary_little_endian)")
        if format_version != '1.0':
            die(f"unsupported PLY version: {format_version} (expected 1.0)")

        # Check if this is a compressed PLY
        if not is_compressed_ply(elements):
            die("input is not a PlayCanvas compressed PLY (missing 'chunk' element or 'packed_position' property)")

        # Find chunk and vertex elements
        chunk_elem = None
        vertex_elem = None

        for elem in elements:
            name, count, props = elem
            if name == 'chunk':
                chunk_elem = elem
            elif name == 'vertex':
                vertex_elem = elem

        if chunk_elem is None:
            die("missing 'chunk' element")
        if vertex_elem is None:
            die("missing 'vertex' element")

        # Read elements in declaration order
        chunks = None
        vertices = None

        for name, count, props in elements:
            if name == 'chunk':
                chunks = read_chunks(f, count, props)
            elif name == 'vertex':
                vertices = read_vertices(f, count, props)
            else:
                skip_element(f, count, props)

        if chunks is None or vertices is None:
            die("failed to read chunk or vertex data")

    # Write output
    with open(output_path, 'wb') as f_out:
        write_ply_header(f_out, len(vertices))
        decode_and_write(f_out, chunks, vertices)

    # Report success
    print(f"Converted {len(vertices)} splats ({len(chunks)} chunks) → {output_path}", file=sys.stderr)


def main():
    if len(sys.argv) != 3:
        usage()

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        die(f"input file not found: {input_path}")

    try:
        convert(input_path, output_path)
    except (OSError, ValueError, struct.error, UnicodeDecodeError, KeyError) as e:
        die(str(e))


if __name__ == '__main__':
    main()
