#include <cstdint>
#include <vector>

#include "hilbert_ref_impl.h"

uint32_t transform(uint32_t prefix, uint32_t sub_coord, uint32_t s_) {
    uint32_t x = sub_coord >> 16;
    uint32_t y = sub_coord & 0xffff;

    uint32_t new_x, new_y;

    switch (prefix) {
        case 0:
            new_x = y;
            new_y = x;
            break;
        case 1:
            new_x = x;
            new_y = y + s_;
            break;
        case 2:
            new_x = x + s_;
            new_y = y + s_;
            break;
	case 3:
            new_x = 2*s_ - 1 - y;
            new_y = s_ - 1 - x;
            break;
	default:
	    __builtin_unreachable();
	    new_x = x;
	    new_y = y;
    }

    return (new_x << 16) | new_y;
}

std::vector<uint32_t> make_table_ref(uint64_t order) {
    if (order == 1) {
	return {0, 1, (1 << 16) | 1, 1 << 16};
    }

    uint64_t dim = 1 << order;
    uint64_t num_indices = dim * dim;

    std::vector<uint32_t> table(num_indices);
    auto sub_table = make_table_ref(order - 1);

    uint64_t k = 2*order - 2;
    uint64_t mask = (1 << k) - 1;

    for (uint64_t i = 0; i < num_indices; ++i) {
        uint64_t prefix = i >> k;
        uint64_t suffix = i & mask;

        uint32_t sub_coord = sub_table[suffix];

	uint32_t dim_ = static_cast<uint32_t>(dim >> 1);  // dim is <= 2^16
        uint32_t coord = transform(prefix, sub_coord, dim_);

        table[i] = coord;
    }

    return table;
}
