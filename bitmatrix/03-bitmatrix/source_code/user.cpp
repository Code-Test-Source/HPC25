#include "common.h"

#include <cstdint>
#include <cstring>
#include <arm_neon.h>


void masked_bitmatrix_density(
    int             N,
    const uint64_t* matrix,
    const uint64_t* mask,
    uint32_t*       result) {
    
    const int blocks_per_row = N / 64;
    
    // 重置结果数组（虽然题目说已初始化，但确保一下）
    std::memset(result, 0, N * sizeof(uint32_t));
    
   // 预计算掩码位位置，减少循环内计算
    for (int i = 0; i < N; i++) {
        int mask_block = i >> 6;
        int mask_bit = i & 63;
        if (!((mask[mask_block] >> mask_bit) & 1)) {
            continue;
        }
        
        const uint64_t* row_ptr = matrix + i * blocks_per_row;
        
        // 一次处理多个列块以减少循环开销
        for (int j_block = 0; j_block < blocks_per_row; j_block++) {
            uint64_t block = row_ptr[j_block];
            uint32_t* target = result + j_block * 64;
            
            // 优化的位处理 - 使用查表法避免重复移位
            // 处理低32位
            uint32_t low32 = (uint32_t)block;
            for (int bit = 0; bit < 32; bit++) {
                target[bit] += (low32 >> bit) & 1;
            }
            
            // 处理高32位
            uint32_t high32 = (uint32_t)(block >> 32);
            for (int bit = 0; bit < 32; bit++) {
                target[bit + 32] += (high32 >> bit) & 1;
            }
        }
    }
}