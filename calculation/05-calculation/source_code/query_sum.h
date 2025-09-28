#ifndef QUERY_SUM_H
#define QUERY_SUM_H

#include <cmath>
#include <vector>

size_t sum(short A[], size_t n, size_t m, double k) {
    size_t answer = 0;
    
    // 预先计算 sin(2p) 的值
    size_t max_j = 0;
    for (size_t i = 1; i <= n; ++i) {
        size_t j_limit = std::pow(i, k);
        if (j_limit > max_j) max_j = j_limit;
    }
    
    std::vector<double> sin2p_cache(max_j + 1);
    for (size_t p = 1; p <= max_j; ++p) {
        sin2p_cache[p] = std::sin(2 * p);
    }
    
    // 预先计算每个 j 对应的 sum_sin
    std::vector<double> sum_sin_cache(max_j + 1, 0.0);
    for (size_t j = 1; j <= max_j; ++j) {
        sum_sin_cache[j] = sum_sin_cache[j-1] + 0.5 * sin2p_cache[j];
    }
    
    // 主循环
    for (size_t i = 1; i <= n; ++i) {
        size_t j_limit = std::pow(i, k);
        double sum_val = 0.0;
        size_t row_answer = 0;
        
        // 优化内存访问：连续访问同一行的数据
        short* row_start = A + i * m + 1;
        
        for (size_t j = 1; j <= j_limit; ++j) {
            short val = row_start[j];
            
            // 分支优化：使用位运算避免分支预测失败
            row_answer += val + ((val & 1) ? 0 : val);
            
            // 使用预先计算的 sum_sin
            sum_val += j * sum_sin_cache[j] * std::sin(j);
        }
        
        answer += row_answer - static_cast<size_t>(sum_val);
    }
    
    return answer;
}

#endif // QUERY_SUM_H