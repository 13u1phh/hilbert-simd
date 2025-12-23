#include <cstdint>
#include <utility>
#include <immintrin.h>
#include <algorithm>
#include <vector>
#include <array>

#include "luts.h"

constexpr std::size_t get_num_blocks(uint64_t order){
	constexpr std::array<std::size_t, 4> blocks_table = {0, 0, 1, 4};

	return order <= 3 ? blocks_table[order] : 16;
}

template <uint64_t order>
std::vector<uint32_t> make_table(){
	if constexpr (order == 1){
		return {0, 1, (1 << 16) | 1, 1 << 16};
	}

	uint64_t dim = 1 << order;
	uint64_t num_indices = dim*dim;

	std::vector<uint32_t> table(num_indices);

	CREATE_SEQ_TABLE;
	CREATE_TRANSFORM_TABLE;
	CREATE_LOWEST_NIBS;

	__m512i blocks = _mm512_setr_epi32(0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0xf0);

	const __m512i block_incr = _mm512_set1_epi32(0x100);

	const __m512i control_bit_mask = _mm512_set1_epi32(0x7f7f7f7f);

	__m512i packed_cm;

	if constexpr (order & 1) {
		constexpr uint32_t padding_scalar = 0b01 << (2*order);
		constexpr uint32_t cm_scalar = (1 << order) - 1;
		constexpr uint32_t packed_cm_scalar = (cm_scalar << 16) | cm_scalar;

		__m512i padding = _mm512_set1_epi32(padding_scalar);
		packed_cm = _mm512_set1_epi32(packed_cm_scalar);

		blocks = _mm512_or_epi32(blocks, padding);
	}

	const __m512i cxor_ctrl  = _mm512_set1_epi32(0x30303030);
	const __m512i bit_select_ctrl = _mm512_set1_epi32(0x33333333);

	constexpr uint8_t cxor {0b01101100}; 
	constexpr uint8_t bit_select {0b10101100}; 

	constexpr uint8_t a_bits = 0b11110000;
	constexpr uint8_t b_bits = 0b11001100;
	constexpr uint8_t c_bits = 0b10101010;

	const __m512i unshuffle_mask_4 = _mm512_set1_epi32(0x00f000f0);
	const __m512i unshuffle_mask_8 = _mm512_set1_epi32(0x0000ff00);

	const __m512i bit_select_ctrl_xx = _mm512_set1_epi32(0x00030000);
	const __m512i bit_select_ctrl_yy = _mm512_set1_epi32(0x00000003);

	uint64_t m_count = std::max((uint64_t)1, num_indices/256);

	for(uint64_t m {}; m < m_count; m++){

		__m512i blocks_hi = _mm512_srli_epi32(blocks, 4);
		blocks_hi = _mm512_and_epi32(blocks_hi, control_bit_mask);

		__m512i blocks_lo = _mm512_and_epi32(blocks, control_bit_mask);

		__m512i seqs_hi = _mm512_shuffle_epi8(SEQ_TABLE, blocks_hi);
		__m512i seqs_lo = _mm512_shuffle_epi8(SEQ_TABLE, blocks_lo);

		__m512i acc_lo = _mm512_ternarylogic_epi32(cxor_ctrl, seqs_lo, seqs_hi, cxor);

		__m512i acc_lo_shr = _mm512_srli_epi32(acc_lo, 8);
		acc_lo = _mm512_ternarylogic_epi32(cxor_ctrl, acc_lo, acc_lo_shr, cxor);

		acc_lo_shr = _mm512_srli_epi32(acc_lo, 16);
		acc_lo = _mm512_ternarylogic_epi32(cxor_ctrl, acc_lo, acc_lo_shr, cxor);

		__m512i acc_hi = _mm512_xor_epi32(acc_lo, seqs_lo);

		// We can re-use the control vector for cxor here

		acc_hi = _mm512_ternarylogic_epi32(cxor_ctrl, seqs_hi, acc_hi, bit_select);

		__m512i coords_hi = _mm512_permutex2var_epi8(TRANSFORM_TABLE, acc_hi, TRANSFORM_TABLE);
		__m512i coords_lo = _mm512_permutex2var_epi8(TRANSFORM_TABLE, acc_lo, TRANSFORM_TABLE);

		coords_hi = _mm512_slli_epi32(coords_hi, 2);
		__m512i coords = _mm512_ternarylogic_epi32(bit_select_ctrl, coords_hi, coords_lo, bit_select);

		// Partial unshuffle

		__m512i coords_shr = _mm512_srli_epi32(coords, 4);
		__m512i quartiles_xor = _mm512_ternarylogic_epi32(unshuffle_mask_4, coords, coords_shr, a_bits & (b_bits ^ c_bits));

		__m512i quartiles_xor_shl = _mm512_slli_epi32(quartiles_xor, 4);
		coords = _mm512_ternarylogic_epi32(coords, quartiles_xor_shl, quartiles_xor, a_bits ^ b_bits ^ c_bits);

		coords_shr = _mm512_srli_epi32(coords, 8);
		quartiles_xor = _mm512_ternarylogic_epi32(unshuffle_mask_8, coords, coords_shr, a_bits & (b_bits ^ c_bits));

		quartiles_xor_shl = _mm512_slli_epi32(quartiles_xor, 8);
		coords = _mm512_ternarylogic_epi32(coords, quartiles_xor_shl, quartiles_xor, a_bits ^ b_bits ^ c_bits);

		// Layout: x0 x1 x2 x3 y0 y1 y2 y3 per block - although the last two x bits and the last two y bits are "missing"

		// Unroll for each block

		constexpr size_t num_blocks = get_num_blocks(order);

		[&]<std::size_t... Is>(std::index_sequence<Is... >){
			([&] {
				constexpr std::size_t block = Is;

				__m512i broadcast = _mm512_set1_epi32(block);
				__m512i acc_lo_block = _mm512_permutexvar_epi32(broadcast, acc_lo);

				__m512i nibs_block = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_block, cxor);

				nibs_block = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_block, TRANSFORM_TABLE);

				__m512i coords_block = _mm512_permutexvar_epi32(broadcast, coords);

				coords_block = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_block, nibs_block, bit_select);

				__m512i nibs_block_shl = _mm512_slli_epi32(nibs_block, 14);
				coords_block = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_block, nibs_block_shl, bit_select);

				if constexpr (order & 1){
					coords_block = _mm512_and_epi32(coords_block, packed_cm);

				}

				_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + block*16), coords_block);
				// _mm512_storeu_si512((__m512i*)(&table[m*256 + block*16]), coords_block);

			}(), ...);

		}(std::make_index_sequence<num_blocks>());

		// Update blocks

		blocks = _mm512_add_epi32(blocks, block_incr);
	}

	return table;

}

