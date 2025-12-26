import itertools as it
from typing import Iterator

from make_luts import make_seq_table, make_transform_table

def concat(xs, elem_width: int) -> int: 
	res = 0

	for i, x in enumerate(xs):
		res |= x << (i * elem_width)

	return res

def make_mm512_epi32_def(words: Iterator, name: str) -> str:
	args = ','.join(f'0x{word:08x}' for word in words)

	return f'_mm512_setr_epi32({args})';

def make_mm512_epi32(words: Iterator, name: str, const = True) -> str:
	const_str = 'const ' if const else ''

	return f'#define CREATE_{name} ' + const_str + f'__m512i {name} = {make_mm512_epi32_def(words, name)};'

def gen_seq_table() -> str:
	seq_table = make_seq_table()

	words_128 = (concat(batch, elem_width = 8) for batch in it.batched(seq_table, 4))
	words_512 = it.chain.from_iterable(it.tee(words_128, 4))

	return make_mm512_epi32(words_512, 'SEQ_TABLE')

def gen_transform_table() -> str:
	transform_table = make_transform_table()
	words = (concat(batch, elem_width = 8) for batch in it.batched(transform_table, 4))

	return make_mm512_epi32(words, 'TRANSFORM_TABLE')

def gen_lowest_nibs() -> str:
	seq_table = make_seq_table()

	words = (seq_table[i] for i in range(16))

	return make_mm512_epi32(words, 'LOWEST_NIBS')

print(gen_seq_table())
print()
print(gen_transform_table())
print()
print(gen_lowest_nibs())

