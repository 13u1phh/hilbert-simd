
PAIR_TO_CS = [0b01, 0b00, 0b00, 0b11]
PAIR_TO_COORDS = [0b00, 0b01, 0b11, 0b10]

def transform(x, cs, width):
    match cs:
        case 0b00:
            return x 

        case 0b01:
            half = width // 2
            mask = (1 << half) - 1

            hw_0 = x >> half
            hw_1 = x & mask

            return (hw_1 << half) | hw_0

        case 0b10:
            mask = (1 << width) - 1

            return ~x & mask

        case 0b11:
            mask = (1 << width) - 1

            return ~transform(x, 0b01, width) & mask

        case _:
            raise ValueError

def make_seq_table():
    res = []

    for i in range(16):
        p0 = i >> 2
        p1 = i & 3

        t0 = PAIR_TO_CS[p0]
        t1 = PAIR_TO_CS[p1]

        t = t0 ^ t1

        coords_0 = PAIR_TO_COORDS[p0]
        coords_1 = PAIR_TO_COORDS[p1]

        coords_0_pt = transform(coords_0, t1, 2)

        # Layout: x0 y0 x1 y1 

        packed_coords = (coords_0_pt << 2) | coords_1

        # Layout: x0 x1 y0 y1

        swap = (packed_coords ^ (packed_coords >> 1)) & 0b0010
        packed_coords = (packed_coords ^ swap) ^ (swap << 1)

        byte = (t << 4) | packed_coords
        res.append(byte)

    return res

def make_transform_table():
    res = []

    for i in range(64):
        # Layout: [c s] [x0 x1 y0 y1]

        cs = i >> 4
        swizzled_coords = i & 0b1111

        swizzled_coords_t = transform(swizzled_coords, cs, 4)

        # Layout: [x0 x1] [junk junk] [y0 y1]

        xs = swizzled_coords_t >> 2
        byte = (xs << 4) | swizzled_coords_t

        res.append(byte)

    return res

