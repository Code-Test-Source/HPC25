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
    
    // 按行处理，利用缓存局部性
    for (int i = 0; i < N; i++) {
        // 检查掩码
        int mask_block = i >> 6;
        int mask_bit = i & 63;
        if (!((mask[mask_block] >> mask_bit) & 1)) {
            continue;
        }
        
        const uint64_t* row_ptr = matrix + i * blocks_per_row;
        
        // 处理该行的所有列块
        for (int j_block = 0; j_block < blocks_per_row; j_block++) {
            uint64_t block = row_ptr[j_block];
            uint32_t* target = result + j_block * 64;
            
            // 完全展开的循环，避免分支预测和循环开销
            // 这样可以最大化指令级并行
            target[0] += (block >> 0) & 1;
            target[1] += (block >> 1) & 1;
            target[2] += (block >> 2) & 1;
            target[3] += (block >> 3) & 1;
            target[4] += (block >> 4) & 1;
            target[5] += (block >> 5) & 1;
            target[6] += (block >> 6) & 1;
            target[7] += (block >> 7) & 1;
            target[8] += (block >> 8) & 1;
            target[9] += (block >> 9) & 1;
            target[10] += (block >> 10) & 1;
            target[11] += (block >> 11) & 1;
            target[12] += (block >> 12) & 1;
            target[13] += (block >> 13) & 1;
            target[14] += (block >> 14) & 1;
            target[15] += (block >> 15) & 1;
            target[16] += (block >> 16) & 1;
            target[17] += (block >> 17) & 1;
            target[18] += (block >> 18) & 1;
            target[19] += (block >> 19) & 1;
            target[20] += (block >> 20) & 1;
            target[21] += (block >> 21) & 1;
            target[22] += (block >> 22) & 1;
            target[23] += (block >> 23) & 1;
            target[24] += (block >> 24) & 1;
            target[25] += (block >> 25) & 1;
            target[26] += (block >> 26) & 1;
            target[27] += (block >> 27) & 1;
            target[28] += (block >> 28) & 1;
            target[29] += (block >> 29) & 1;
            target[30] += (block >> 30) & 1;
            target[31] += (block >> 31) & 1;
            target[32] += (block >> 32) & 1;
            target[33] += (block >> 33) & 1;
            target[34] += (block >> 34) & 1;
            target[35] += (block >> 35) & 1;
            target[36] += (block >> 36) & 1;
            target[37] += (block >> 37) & 1;
            target[38] += (block >> 38) & 1;
            target[39] += (block >> 39) & 1;
            target[40] += (block >> 40) & 1;
            target[41] += (block >> 41) & 1;
            target[42] += (block >> 42) & 1;
            target[43] += (block >> 43) & 1;
            target[44] += (block >> 44) & 1;
            target[45] += (block >> 45) & 1;
            target[46] += (block >> 46) & 1;
            target[47] += (block >> 47) & 1;
            target[48] += (block >> 48) & 1;
            target[49] += (block >> 49) & 1;
            target[50] += (block >> 50) & 1;
            target[51] += (block >> 51) & 1;
            target[52] += (block >> 52) & 1;
            target[53] += (block >> 53) & 1;
            target[54] += (block >> 54) & 1;
            target[55] += (block >> 55) & 1;
            target[56] += (block >> 56) & 1;
            target[57] += (block >> 57) & 1;
            target[58] += (block >> 58) & 1;
            target[59] += (block >> 59) & 1;
            target[60] += (block >> 60) & 1;
            target[61] += (block >> 61) & 1;
            target[62] += (block >> 62) & 1;
            target[63] += (block >> 63) & 1;
        }
    }
}