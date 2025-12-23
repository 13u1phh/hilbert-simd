#include <cstdint>
#include <cstdlib>
#include <new>
#include <immintrin.h>
#include <vector>

#include "luts.h"

template <uint64_t order>
std::vector<uint32_t> make_table_256_manual(){
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

	// this logic breaks down if order = 1 I think - should just do a compile-time check and return 0, 1, (1 << 16) | 1, 1 << 16
	// in general need to think about how partial unroll interacts with table size/padding

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

	for(uint64_t m {}; m < num_indices/256; m++){

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

		// Layout: x0 x1 x2 x3 y0 y1 y2 y3 - although the last two x bits and the last two y bits are "missing"

		// Block 0

		__m512i broadcast_0 = _mm512_setzero_si512();

		__m512i acc_lo_0 = _mm512_permutexvar_epi32(broadcast_0, acc_lo);
		__m512i nibs_0 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_0, cxor);

		nibs_0 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_0, TRANSFORM_TABLE);

		__m512i coords_0 = _mm512_permutexvar_epi32(broadcast_0, coords);
		coords_0 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_0, nibs_0, bit_select);

		__m512i nibs_0_shl = _mm512_slli_epi32(nibs_0, 14);
		coords_0 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_0, nibs_0_shl, bit_select);

		if constexpr (order & 1) {
			coords_0 = _mm512_and_epi32(coords_0, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 0*16), coords_0);

		// Block 1

		__m512i broadcast_1 = _mm512_set1_epi32(1);

		__m512i acc_lo_1 = _mm512_permutexvar_epi32(broadcast_1, acc_lo);
		__m512i nibs_1 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_1, cxor);

		nibs_1 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_1, TRANSFORM_TABLE);

		__m512i coords_1 = _mm512_permutexvar_epi32(broadcast_1, coords);
		coords_1 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_1, nibs_1, bit_select);

		__m512i nibs_1_shl = _mm512_slli_epi32(nibs_1, 14);
		coords_1 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_1, nibs_1_shl, bit_select);

		if constexpr (order & 1) {
			coords_1 = _mm512_and_epi32(coords_1, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 1*16), coords_1);

		// Block 2

		__m512i broadcast_2 = _mm512_set1_epi32(2);

		__m512i acc_lo_2 = _mm512_permutexvar_epi32(broadcast_2, acc_lo);
		__m512i nibs_2 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_2, cxor);

		nibs_2 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_2, TRANSFORM_TABLE);

		__m512i coords_2 = _mm512_permutexvar_epi32(broadcast_2, coords);
		coords_2 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_2, nibs_2, bit_select);

		__m512i nibs_2_shl = _mm512_slli_epi32(nibs_2, 14);
		coords_2 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_2, nibs_2_shl, bit_select);

		if constexpr (order & 1) {
			coords_2 = _mm512_and_epi32(coords_2, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 2*16), coords_2);

		// Block 3

		__m512i broadcast_3 = _mm512_set1_epi32(3);

		__m512i acc_lo_3 = _mm512_permutexvar_epi32(broadcast_3, acc_lo);
		__m512i nibs_3 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_3, cxor);

		nibs_3 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_3, TRANSFORM_TABLE);

		__m512i coords_3 = _mm512_permutexvar_epi32(broadcast_3, coords);
		coords_3 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_3, nibs_3, bit_select);

		__m512i nibs_3_shl = _mm512_slli_epi32(nibs_3, 14);
		coords_3 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_3, nibs_3_shl, bit_select);

		if constexpr (order & 1) {
			coords_3 = _mm512_and_epi32(coords_3, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 3*16), coords_3);

		// Block 4

		__m512i broadcast_4 = _mm512_set1_epi32(4);

		__m512i acc_lo_4 = _mm512_permutexvar_epi32(broadcast_4, acc_lo);
		__m512i nibs_4 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_4, cxor);

		nibs_4 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_4, TRANSFORM_TABLE);

		__m512i coords_4 = _mm512_permutexvar_epi32(broadcast_4, coords);
		coords_4 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_4, nibs_4, bit_select);

		__m512i nibs_4_shl = _mm512_slli_epi32(nibs_4, 14);
		coords_4 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_4, nibs_4_shl, bit_select);

		if constexpr (order & 1) {
			coords_4 = _mm512_and_epi32(coords_4, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 4*16), coords_4);

		// Block 5

		__m512i broadcast_5 = _mm512_set1_epi32(5);

		__m512i acc_lo_5 = _mm512_permutexvar_epi32(broadcast_5, acc_lo);
		__m512i nibs_5 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_5, cxor);

		nibs_5 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_5, TRANSFORM_TABLE);

		__m512i coords_5 = _mm512_permutexvar_epi32(broadcast_5, coords);
		coords_5 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_5, nibs_5, bit_select);

		__m512i nibs_5_shl = _mm512_slli_epi32(nibs_5, 14);
		coords_5 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_5, nibs_5_shl, bit_select);

		if constexpr (order & 1) {
			coords_5 = _mm512_and_epi32(coords_5, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 5*16), coords_5);

		// Block 6

		__m512i broadcast_6 = _mm512_set1_epi32(6);

		__m512i acc_lo_6 = _mm512_permutexvar_epi32(broadcast_6, acc_lo);
		__m512i nibs_6 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_6, cxor);

		nibs_6 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_6, TRANSFORM_TABLE);

		__m512i coords_6 = _mm512_permutexvar_epi32(broadcast_6, coords);
		coords_6 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_6, nibs_6, bit_select);

		__m512i nibs_6_shl = _mm512_slli_epi32(nibs_6, 14);
		coords_6 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_6, nibs_6_shl, bit_select);

		if constexpr (order & 1) {
			coords_6 = _mm512_and_epi32(coords_6, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 6*16), coords_6);

		// Block 7

		__m512i broadcast_7 = _mm512_set1_epi32(7);

		__m512i acc_lo_7 = _mm512_permutexvar_epi32(broadcast_7, acc_lo);
		__m512i nibs_7 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_7, cxor);

		nibs_7 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_7, TRANSFORM_TABLE);

		__m512i coords_7 = _mm512_permutexvar_epi32(broadcast_7, coords);
		coords_7 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_7, nibs_7, bit_select);

		__m512i nibs_7_shl = _mm512_slli_epi32(nibs_7, 14);
		coords_7 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_7, nibs_7_shl, bit_select);

		if constexpr (order & 1) {
			coords_7 = _mm512_and_epi32(coords_7, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 7*16), coords_7);

		// Block 8

		__m512i broadcast_8 = _mm512_set1_epi32(8);

		__m512i acc_lo_8 = _mm512_permutexvar_epi32(broadcast_8, acc_lo);
		__m512i nibs_8 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_8, cxor);

		nibs_8 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_8, TRANSFORM_TABLE);

		__m512i coords_8 = _mm512_permutexvar_epi32(broadcast_8, coords);
		coords_8 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_8, nibs_8, bit_select);

		__m512i nibs_8_shl = _mm512_slli_epi32(nibs_8, 14);
		coords_8 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_8, nibs_8_shl, bit_select);

		if constexpr (order & 1) {
			coords_8 = _mm512_and_epi32(coords_8, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 8*16), coords_8);

		// Block 9

		__m512i broadcast_9 = _mm512_set1_epi32(9);

		__m512i acc_lo_9 = _mm512_permutexvar_epi32(broadcast_9, acc_lo);
		__m512i nibs_9 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_9, cxor);

		nibs_9 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_9, TRANSFORM_TABLE);

		__m512i coords_9 = _mm512_permutexvar_epi32(broadcast_9, coords);
		coords_9 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_9, nibs_9, bit_select);

		__m512i nibs_9_shl = _mm512_slli_epi32(nibs_9, 14);
		coords_9 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_9, nibs_9_shl, bit_select);

		if constexpr (order & 1) {
			coords_9 = _mm512_and_epi32(coords_9, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 9*16), coords_9);

		// Block 10

		__m512i broadcast_10 = _mm512_set1_epi32(10);

		__m512i acc_lo_10 = _mm512_permutexvar_epi32(broadcast_10, acc_lo);
		__m512i nibs_10 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_10, cxor);

		nibs_10 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_10, TRANSFORM_TABLE);

		__m512i coords_10 = _mm512_permutexvar_epi32(broadcast_10, coords);
		coords_10 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_10, nibs_10, bit_select);

		__m512i nibs_10_shl = _mm512_slli_epi32(nibs_10, 14);
		coords_10 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_10, nibs_10_shl, bit_select);

		if constexpr (order & 1) {
			coords_10 = _mm512_and_epi32(coords_10, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 10*16), coords_10);

		// Block 11

		__m512i broadcast_11 = _mm512_set1_epi32(11);

		__m512i acc_lo_11 = _mm512_permutexvar_epi32(broadcast_11, acc_lo);
		__m512i nibs_11 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_11, cxor);

		nibs_11 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_11, TRANSFORM_TABLE);

		__m512i coords_11 = _mm512_permutexvar_epi32(broadcast_11, coords);
		coords_11 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_11, nibs_11, bit_select);

		__m512i nibs_11_shl = _mm512_slli_epi32(nibs_11, 14);
		coords_11 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_11, nibs_11_shl, bit_select);

		if constexpr (order & 1) {
			coords_11 = _mm512_and_epi32(coords_11, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 11*16), coords_11);

		// Block 12

		__m512i broadcast_12 = _mm512_set1_epi32(12);

		__m512i acc_lo_12 = _mm512_permutexvar_epi32(broadcast_12, acc_lo);
		__m512i nibs_12 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_12, cxor);

		nibs_12 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_12, TRANSFORM_TABLE);

		__m512i coords_12 = _mm512_permutexvar_epi32(broadcast_12, coords);
		coords_12 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_12, nibs_12, bit_select);

		__m512i nibs_12_shl = _mm512_slli_epi32(nibs_12, 14);
		coords_12 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_12, nibs_12_shl, bit_select);

		if constexpr (order & 1) {
			coords_12 = _mm512_and_epi32(coords_12, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 12*16), coords_12);

		// Block 13

		__m512i broadcast_13 = _mm512_set1_epi32(13);

		__m512i acc_lo_13 = _mm512_permutexvar_epi32(broadcast_13, acc_lo);
		__m512i nibs_13 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_13, cxor);

		nibs_13 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_13, TRANSFORM_TABLE);

		__m512i coords_13 = _mm512_permutexvar_epi32(broadcast_13, coords);
		coords_13 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_13, nibs_13, bit_select);

		__m512i nibs_13_shl = _mm512_slli_epi32(nibs_13, 14);
		coords_13 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_13, nibs_13_shl, bit_select);

		if constexpr (order & 1) {
			coords_13 = _mm512_and_epi32(coords_13, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 13*16), coords_13);

		// Block 14

		__m512i broadcast_14 = _mm512_set1_epi32(14);

		__m512i acc_lo_14 = _mm512_permutexvar_epi32(broadcast_14, acc_lo);
		__m512i nibs_14 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_14, cxor);

		nibs_14 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_14, TRANSFORM_TABLE);

		__m512i coords_14 = _mm512_permutexvar_epi32(broadcast_14, coords);
		coords_14 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_14, nibs_14, bit_select);

		__m512i nibs_14_shl = _mm512_slli_epi32(nibs_14, 14);
		coords_14 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_14, nibs_14_shl, bit_select);

		if constexpr (order & 1) {
			coords_14 = _mm512_and_epi32(coords_14, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 14*16), coords_14);

		// Block 15

		__m512i broadcast_15 = _mm512_set1_epi32(15);

		__m512i acc_lo_15 = _mm512_permutexvar_epi32(broadcast_15, acc_lo);
		__m512i nibs_15 = _mm512_ternarylogic_epi32(cxor_ctrl, LOWEST_NIBS, acc_lo_15, cxor);

		nibs_15 = _mm512_permutex2var_epi8(TRANSFORM_TABLE, nibs_15, TRANSFORM_TABLE);

		__m512i coords_15 = _mm512_permutexvar_epi32(broadcast_15, coords);
		coords_15 = _mm512_ternarylogic_epi32(bit_select_ctrl_yy, coords_15, nibs_15, bit_select);

		__m512i nibs_15_shl = _mm512_slli_epi32(nibs_15, 14);
		coords_15 = _mm512_ternarylogic_epi32(bit_select_ctrl_xx, coords_15, nibs_15_shl, bit_select);

		if constexpr (order & 1) {
			coords_15 = _mm512_and_epi32(coords_15, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + m*256 + 15*16), coords_15);

		// Update blocks

		blocks = _mm512_add_epi32(blocks, block_incr);
	}

	return table;

}

