#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <vector>

template <uint64_t order>
std::vector<int32_t> make_table_cs(){
	uint64_t dim = 1 << order;
	uint64_t num_indices = dim*dim;

	std::vector<int32_t> table(num_indices);

	__m512i indices = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	__m512i incr = _mm512_set1_epi32(16);

	constexpr uint8_t a_bits = 0b11110000;
	constexpr uint8_t b_bits = 0b11001100;
	constexpr uint8_t c_bits = 0b10101010;

	__m512i evens_mask = _mm512_set1_epi32(0x55555555);
	__m512i packed_cm;
	
	if constexpr (order & 1) {
		constexpr uint32_t padding_scalar = 0b01 << (2*order);
		constexpr uint32_t cm_scalar = (1 << order) - 1;
		constexpr uint32_t packed_cm_scalar = (cm_scalar << 16) | cm_scalar;

		__m512i padding = _mm512_set1_epi32(padding_scalar);
		packed_cm = _mm512_set1_epi32(packed_cm_scalar);

		indices = _mm512_or_epi32(indices, padding);
	}

	const __m512i unshuffle_mask_1 = _mm512_setzero_si512();
	const __m512i unshuffle_mask_2 = _mm512_setzero_si512();
	const __m512i unshuffle_mask_4 = _mm512_setzero_si512();
	const __m512i unshuffle_mask_8 = _mm512_setzero_si512();

	for(uint64_t b {}; b < num_indices / 16; b++){

		__m512i indices_shl = _mm512_srli_epi32(indices, 1);
		__m512i sr = _mm512_and_epi32(indices_shl, evens_mask);

		__m512i indices_evens = _mm512_and_epi32(indices, evens_mask);
		__m512i cs = _mm512_xor_epi32(_mm512_add_epi32(indices_evens, sr), evens_mask);

		cs = _mm512_xor_si512(cs, _mm512_srli_epi32( cs,  2 ));
		cs = _mm512_xor_si512(cs, _mm512_srli_epi32( cs,  4 ));
		cs = _mm512_xor_si512(cs, _mm512_srli_epi32( cs,  8 ));
		cs = _mm512_xor_si512(cs, _mm512_srli_epi32( cs, 16 ));

		__m512i swap = _mm512_and_epi32(cs, evens_mask);

		__m512i cs_shr = _mm512_srli_epi32(cs, 1);
		__m512i comp = _mm512_and_epi32(cs_shr, evens_mask);

		__m512i t = _mm512_ternarylogic_epi32(indices, swap, comp, (a_bits & b_bits) ^ c_bits);

		__m512i t_shl = _mm512_slli_epi32(t, 1);
		
		indices = _mm512_xor_epi32(indices, _mm512_ternarylogic_epi32(sr, t, t_shl, a_bits ^ b_bits ^ c_bits));

		// Unshuffle

		/* TO DO: replace first two unshuffles with a GNFI affine transform */

		// Quartile size: 1 

		__m512i indices_shr = _mm512_srli_epi32(indices, 1);
		__m512i quartiles_xor = _mm512_ternarylogic_epi32(unshuffle_mask_1, indices, indices_shr, a_bits & (b_bits ^ c_bits));

		__m512i quartiles_xor_shl = _mm512_slli_epi32(quartiles_xor, 1);
		indices = _mm512_ternarylogic_epi32(indices, quartiles_xor_shl, quartiles_xor, a_bits ^ b_bits ^ c_bits);

		// Quartile size: 2

		indices_shr = _mm512_srli_epi32(indices, 2);
		quartiles_xor = _mm512_ternarylogic_epi32(unshuffle_mask_2, indices, indices_shr, a_bits & (b_bits ^ c_bits));

		quartiles_xor_shl = _mm512_slli_epi32(quartiles_xor, 2);
		indices = _mm512_ternarylogic_epi32(indices, quartiles_xor_shl, quartiles_xor, a_bits ^ b_bits ^ c_bits);

		// Quartile size: 4

		indices_shr = _mm512_srli_epi32(indices, 4);
		quartiles_xor = _mm512_ternarylogic_epi32(unshuffle_mask_4, indices, indices_shr, a_bits & (b_bits ^ c_bits));

		quartiles_xor_shl = _mm512_slli_epi32(quartiles_xor, 4);
		indices = _mm512_ternarylogic_epi32(indices, quartiles_xor_shl, quartiles_xor, a_bits ^ b_bits ^ c_bits);

		// Quartile size: 8

		indices_shr = _mm512_srli_epi32(indices, 8);
		quartiles_xor = _mm512_ternarylogic_epi32(unshuffle_mask_8, indices, indices_shr, a_bits & (b_bits ^ c_bits));

		quartiles_xor_shl = _mm512_slli_epi32(quartiles_xor, 8);
		indices = _mm512_ternarylogic_epi32(indices, quartiles_xor_shl, quartiles_xor, a_bits ^ b_bits ^ c_bits);

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + b*16), indices);

		indices = _mm512_add_epi32(indices, incr);
	}

	return table;
}

