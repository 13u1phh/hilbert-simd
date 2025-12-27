#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <vector>

template <uint64_t order>
std::vector<uint32_t> make_table_cs(){
	if constexpr (order == 1){
		return {0, 1, (1 << 16) | 1, 1 << 16};
	}

	uint64_t dim = 1 << order;
	uint64_t num_indices = dim*dim;

	std::vector<uint32_t> table(num_indices);

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

	/*

	In a GFNI affine transform, each byte within a 64-bit lane uses the corresponding 64-bit matrix in the transform vector.
	The matrix is applied to each byte in big-endian order i.e. LS bit of the result byte is determined by MS byte of the matrix.

	Our desired unshuffle (in little-endian order) is: x0 y0 x1 y1 x2 y2 x3 y3 --> x0 x1 x2 x3 y0 y1 y2 y3.
	Then, the corresponding matrix bytes are: 01 04 10 40 02 08 20 80

	Again, these bytes are in big-endian order, so our matrix is: 0x0104104002082080.

	*/

	const __m512i unshuffle_matrix_1_2 = _mm512_set1_epi64(0x0104104002082080);

	const __m512i unshuffle_mask_4 = _mm512_set1_epi32(0x00f000f0);
	const __m512i unshuffle_mask_8 = _mm512_set1_epi32(0x0000ff00);

	for(uint64_t b {}; b < num_indices/16; b++){

		__m512i coords_shl = _mm512_srli_epi32(indices, 1);
		__m512i sr = _mm512_and_epi32(coords_shl, evens_mask);

		__m512i coords_evens = _mm512_and_epi32(indices, evens_mask);
		__m512i cs = _mm512_xor_epi32(_mm512_add_epi32(coords_evens, sr), evens_mask);

		cs = _mm512_xor_si512(cs, _mm512_srli_epi32(cs,  2 ));
		cs = _mm512_xor_si512(cs, _mm512_srli_epi32(cs,  4 ));
		cs = _mm512_xor_si512(cs, _mm512_srli_epi32(cs,  8 ));
		cs = _mm512_xor_si512(cs, _mm512_srli_epi32(cs, 16 ));

		__m512i swap = _mm512_and_epi32(cs, evens_mask);

		__m512i cs_shr = _mm512_srli_epi32(cs, 1);
		__m512i comp = _mm512_and_epi32(cs_shr, evens_mask);

		__m512i t = _mm512_ternarylogic_epi32(indices, swap, comp, (a_bits & b_bits) ^ c_bits);

		__m512i t_shl = _mm512_slli_epi32(t, 1);

		__m512i coords = _mm512_xor_epi32(indices, _mm512_ternarylogic_epi32(sr, t, t_shl, a_bits ^ b_bits ^ c_bits));

		// Unshuffle

		coords = _mm512_gf2p8affine_epi64_epi8(coords, unshuffle_matrix_1_2, 0); 

		// Quartile size: 4

		__m512i coords_shr = _mm512_srli_epi32(coords, 4);
		__m512i quartiles_xor = _mm512_ternarylogic_epi32(unshuffle_mask_4, coords, coords_shr, a_bits & (b_bits ^ c_bits));

		__m512i quartiles_xor_shl = _mm512_slli_epi32(quartiles_xor, 4);
		coords = _mm512_ternarylogic_epi32(coords, quartiles_xor_shl, quartiles_xor, a_bits ^ b_bits ^ c_bits);

		// Quartile size: 8

		coords_shr = _mm512_srli_epi32(coords, 8);
		quartiles_xor = _mm512_ternarylogic_epi32(unshuffle_mask_8, coords, coords_shr, a_bits & (b_bits ^ c_bits));

		quartiles_xor_shl = _mm512_slli_epi32(quartiles_xor, 8);
		coords = _mm512_ternarylogic_epi32(coords, quartiles_xor_shl, quartiles_xor, a_bits ^ b_bits ^ c_bits);

		if constexpr (order & 1) {
			coords = _mm512_and_epi32(coords, packed_cm);
		}

		_mm512_storeu_si512(reinterpret_cast<__m512i*>(table.data() + b*16), coords);

		indices = _mm512_add_epi32(indices, incr);
	}

	return table;
}

