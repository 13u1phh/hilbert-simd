#include <cstdint>
#include <immintrin.h>
#include <vector>

#include "luts.h"

template <uint64_t order>
std::vector<uint32_t> make_table_16(){
	uint64_t dim = 1 << order;
	uint64_t num_indices = dim*dim;

	std::vector<uint32_t> table(num_indices);

	CREATE_SEQ_TABLE;
	CREATE_TRANSFORM_TABLE;

	__m512i indices = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	__m512i incr = _mm512_set1_epi32(16);

	__m512i control_bit_mask = _mm512_set1_epi32(0x7f7f7f7f);

	__m512i packed_cm;

	if constexpr (order & 1) {
		constexpr uint32_t padding_scalar = 0b01 << (2*order);
		constexpr uint32_t cm_scalar = (1 << order) - 1;
		constexpr uint32_t packed_cm_scalar = (cm_scalar << 16) | cm_scalar;

		__m512i padding = _mm512_set1_epi32(padding_scalar);
		packed_cm = _mm512_set1_epi32(packed_cm_scalar);

		indices = _mm512_or_epi32(indices, padding);
	}


	__m512i cxor_control = _mm512_set1_epi32(0x30303030);
	__m512i bit_select_control = _mm512_set1_epi32(0x33333333);

	__m512i unshuffle_mask_4 = _mm512_set1_epi32(0x00f000f0);
	__m512i unshuffle_mask_8 = _mm512_set1_epi32(0x0000ff00);

	constexpr uint8_t a_bits = 0b11110000;
	constexpr uint8_t b_bits = 0b11001100;
	constexpr uint8_t c_bits = 0b10101010;

	constexpr uint8_t cxor {0b01101100}; 
	constexpr uint8_t bit_select {0b10101100}; 

	for(uint64_t b {}; b < num_indices / 16; b++){
		__m512i indices_hi = _mm512_srli_epi32(indices, 4);
		indices_hi = _mm512_and_epi32(indices_hi, control_bit_mask);

		__m512i seqs_hi = _mm512_shuffle_epi8(SEQ_TABLE, indices_hi);

		__m512i indices_lo = _mm512_and_epi32(indices, control_bit_mask);
		__m512i seqs_lo = _mm512_shuffle_epi8(SEQ_TABLE, indices_lo);

		__m512i acc_lo = _mm512_ternarylogic_epi32(cxor_control, seqs_lo, seqs_hi, cxor);

		__m512i acc_lo_shr = _mm512_srli_epi32(acc_lo, 8);
		acc_lo = _mm512_ternarylogic_epi32(cxor_control, acc_lo, acc_lo_shr, cxor);

		acc_lo_shr = _mm512_srli_epi32(acc_lo, 16);
		acc_lo = _mm512_ternarylogic_epi32(cxor_control, acc_lo, acc_lo_shr, cxor);

		__m512i acc_hi = _mm512_xor_epi32(acc_lo, seqs_lo);

		// We can re-use the control vector for cxor here

		acc_hi = _mm512_ternarylogic_epi32(cxor_control, seqs_hi, acc_hi, bit_select);

		__m512i coords_hi = _mm512_permutex2var_epi8(TRANSFORM_TABLE, acc_hi, TRANSFORM_TABLE);
		__m512i coords_lo = _mm512_permutex2var_epi8(TRANSFORM_TABLE, acc_lo, TRANSFORM_TABLE);

		coords_hi = _mm512_slli_epi32(coords_hi, 2);
		__m512i coords = _mm512_ternarylogic_epi32(bit_select_control, coords_hi, coords_lo, bit_select);

		// Layout: x0 x1 x2 x3 y0 y1 y2 y3 ...

		__m512i coords_shr = _mm512_srli_epi32(coords, 4);
		__m512i quartiles_xor = _mm512_ternarylogic_epi32(unshuffle_mask_4, coords, coords_shr, a_bits & (b_bits ^ c_bits));

		__m512i quartiles_xor_shl = _mm512_slli_epi32(quartiles_xor, 4);
		coords = _mm512_ternarylogic_epi32(coords, quartiles_xor_shl, quartiles_xor, a_bits ^ b_bits ^ c_bits);

		coords_shr = _mm512_srli_epi32(coords, 8);
		quartiles_xor = _mm512_ternarylogic_epi32(unshuffle_mask_8, coords, coords_shr, a_bits & (b_bits ^ c_bits));

		quartiles_xor_shl = _mm512_slli_epi32(quartiles_xor, 8);
		coords = _mm512_ternarylogic_epi32(coords, quartiles_xor_shl, quartiles_xor, a_bits ^ b_bits ^ c_bits);

		// Mask out bits from padding

		if constexpr (order & 1) {
			coords = _mm512_and_epi32(coords, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + b*16), coords);

		indices = _mm512_add_epi32(indices, incr);
	}

	return table;

}

