#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "hilbert.h"
#include "hilbert_cs.h"
#include "hilbert_ref_impl.h"

int main(){

	[&]<std::size_t... Is>(std::index_sequence<Is... >){
		([&] {
			constexpr uint64_t order = (uint64_t)(Is + 1);
			std::cout << "ORDER: " << order << std::endl;

			std::vector<uint32_t> table = make_table<order>();
			std::vector<uint32_t> table_cs = make_table_cs<order>();
			std::vector<uint32_t> table_ref = make_table_ref(order);

			uint64_t dim = 1 << order;
			uint64_t num_indices = dim*dim;

			for(uint64_t i {}; i < num_indices; i++){
				if (table[i] != table_ref[i]){
					std::cout << "Test failed for index " << i << std::endl;
					std::cout << "From megablock algorithm: " << table[i] << std::endl;
					std::cout << "From ref: " << table_ref[i] << std::endl;
					break;

				}
				else if(table_cs[i] != table_ref[i]){
					std::cout << "Test failed for index " << i << std::endl;
					std::cout << "From complement-and-swap algorithm: " << table_cs[i] << std::endl;
					std::cout << "From ref: " << table_ref[i] << std::endl;
					break;
				}

			}

		}(), ...);
	}(std::make_index_sequence<16>()); // [0, 16) gets offset to [1, 16]
}
