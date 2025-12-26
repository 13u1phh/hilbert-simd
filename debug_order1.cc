#include <cstdint>
#include <iostream>
#include <iomanip>
#include "hilbert_cs.h"
#include "hilbert_ref_impl.h"

int main() {
    auto table_cs = make_table_cs<1>();
    auto table_ref = make_table_ref(1);
    
    std::cout << "Order 1 comparison:" << std::endl;
    for (int i = 0; i < 4; i++) {
        std::cout << "Index " << i << ": ";
        std::cout << "CS=0x" << std::hex << std::setfill('0') << std::setw(8) << table_cs[i];
        std::cout << " (" << std::dec << (table_cs[i] >> 16) << "," << (table_cs[i] & 0xFFFF) << ")";
        std::cout << " Ref=0x" << std::hex << std::setw(8) << table_ref[i];
        std::cout << " (" << std::dec << (table_ref[i] >> 16) << "," << (table_ref[i] & 0xFFFF) << ")";
        if (table_cs[i] != table_ref[i]) std::cout << " FAIL";
        std::cout << std::endl;
    }
}
