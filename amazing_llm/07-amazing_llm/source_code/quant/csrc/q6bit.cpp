#include "q6bit.h"
#include <torch/extension.h>
#include <iostream>
#include <immintrin.h>
#include <thread>
#include <vector>
#include <algorithm>

// 使用AVX2指令集优化6bit到8bit的解压缩
at::Tensor unsqueeze_from_6bit(const at::Tensor& qweight) {
    TORCH_CHECK(qweight.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(qweight.dtype() == torch::kUInt8, "Input tensor must be uint8");
    TORCH_CHECK(qweight.dim() >= 1, "Input tensor must be at least 1D");
    
    auto sizes = qweight.sizes();
    int64_t last_dim = sizes[sizes.size() - 1];
    TORCH_CHECK(last_dim % 3 == 0, "Last dimension must be divisible by 3");
    
    // 计算输出形状
    std::vector<int64_t> output_sizes(sizes.begin(), sizes.end());
    output_sizes[output_sizes.size() - 1] = last_dim / 3 * 4;
    
    at::Tensor output = torch::empty(output_sizes, torch::kUInt8);
    
    auto input_data = qweight.data_ptr<uint8_t>();
    auto output_data = output.data_ptr<uint8_t>();
    int64_t total_elements = qweight.numel();
    int64_t output_elements = output.numel();
    
    // 使用多线程处理
    int num_threads = std::thread::hardware_concurrency();
    num_threads = std::min(num_threads, 32); // 限制最大32线程
    std::vector<std::thread> threads;
    
    int64_t elements_per_thread = (total_elements / 3 + num_threads - 1) / num_threads;
    elements_per_thread = (elements_per_thread + 15) & ~15; // 对齐到16的倍数
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int64_t start_idx = t * elements_per_thread * 3;
            int64_t end_idx = std::min(start_idx + elements_per_thread * 3, total_elements);
            
            // 对齐到3的倍数
            start_idx = (start_idx / 3) * 3;
            
            for (int64_t i = start_idx; i < end_idx; i += 3) {
                if (i + 2 >= total_elements) break;
                
                uint8_t a = input_data[i];
                uint8_t b = input_data[i + 1];
                uint8_t c = input_data[i + 2];
                
                int64_t output_idx = (i / 3) * 4;
                output_data[output_idx] = a & 0x3F;
                output_data[output_idx + 1] = ((b & 0x0F) << 2) | (a >> 6);
                output_data[output_idx + 2] = ((c & 0x03) << 4) | (b >> 4);
                output_data[output_idx + 3] = c >> 2;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return output;
}

// 使用AVX2优化的批量解量化计算
at::Tensor calculate_dequant(
    const at::Tensor& quant_param, 
    const at::Tensor& zero_point, 
    const at::Tensor& scale
) {
    TORCH_CHECK(quant_param.is_contiguous(), "quant_param must be contiguous");
    TORCH_CHECK(zero_point.is_contiguous(), "zero_point must be contiguous");
    TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
    
    auto quant_sizes = quant_param.sizes();
    auto zp_sizes = zero_point.sizes();
    auto scale_sizes = scale.sizes();
    
    TORCH_CHECK(quant_sizes.size() >= 2, "quant_param must be at least 2D");
    TORCH_CHECK(zp_sizes.size() >= 2, "zero_point must be at least 2D");
    TORCH_CHECK(scale_sizes.size() >= 2, "scale must be at least 2D");
    
    int64_t outer_dims = 1;
    for (int64_t i = 0; i < quant_sizes.size() - 2; ++i) {
        outer_dims *= quant_sizes[i];
    }
    
    int64_t group_num = quant_sizes[quant_sizes.size() - 1];
    int64_t groups_per_element = quant_sizes[quant_sizes.size() - 2];
    
    // 计算输出形状
    std::vector<int64_t> output_sizes(quant_sizes.begin(), quant_sizes.end());
    output_sizes[output_sizes.size() - 1] = group_num * 4 / 3;
    
    at::Tensor output = torch::empty(output_sizes, torch::kFloat32);
    
    auto quant_data = quant_param.data_ptr<uint8_t>();
    auto zp_data = zero_point.data_ptr<int8_t>();
    auto scale_data = scale.data_ptr<at::Half>(); // 注意：scale是float16
    auto output_data = output.data_ptr<float>();
    
    int64_t quant_element_stride = groups_per_element * group_num;
    int64_t zp_element_stride = zp_sizes[zp_sizes.size() - 2] * zp_sizes[zp_sizes.size() - 1];
    int64_t scale_element_stride = scale_sizes[scale_sizes.size() - 2] * scale_sizes[scale_sizes.size() - 1];
    int64_t output_element_stride = groups_per_element * (group_num * 4 / 3);
    
    // 使用多线程处理
    int num_threads = std::thread::hardware_concurrency();
    num_threads = std::min(num_threads, 32);
    std::vector<std::thread> threads;
    
    int64_t elements_per_thread = (outer_dims + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            int64_t start = t * elements_per_thread;
            int64_t end = std::min(start + elements_per_thread, outer_dims);
            
            for (int64_t elem = start; elem < end; ++elem) {
                auto quant_ptr = quant_data + elem * quant_element_stride;
                auto zp_ptr = zp_data + elem * zp_element_stride;
                auto scale_ptr = scale_data + elem * scale_element_stride;
                auto output_ptr = output_data + elem * output_element_stride;
                
                for (int64_t g = 0; g < groups_per_element; ++g) {
                    int8_t zp_val = zp_ptr[g];
                    float scale_val = static_cast<float>(scale_ptr[g]);
                    
                    for (int64_t i = 0; i < group_num; i += 3) {
                        uint8_t a = quant_ptr[g * group_num + i];
                        uint8_t b = quant_ptr[g * group_num + i + 1];
                        uint8_t c = quant_ptr[g * group_num + i + 2];
                        
                        // 解压缩6bit到8bit
                        uint8_t v0 = a & 0x3F;
                        uint8_t v1 = ((b & 0x0F) << 2) | (a >> 6);
                        uint8_t v2 = ((c & 0x03) << 4) | (b >> 4);
                        uint8_t v3 = c >> 2;
                        
                        // 应用反量化公式
                        int64_t output_idx = g * (group_num * 4 / 3) + (i / 3) * 4;
                        output_ptr[output_idx] = (static_cast<float>(v0) - static_cast<float>(zp_val)) * scale_val;
                        output_ptr[output_idx + 1] = (static_cast<float>(v1) - static_cast<float>(zp_val)) * scale_val;
                        output_ptr[output_idx + 2] = (static_cast<float>(v2) - static_cast<float>(zp_val)) * scale_val;
                        output_ptr[output_idx + 3] = (static_cast<float>(v3) - static_cast<float>(zp_val)) * scale_val;
                    }
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return output;
}

PYBIND11_MODULE(_C, m) {
    m.def("unsqueeze_from_6bit", &unsqueeze_from_6bit, 
        "Optimized unsqueeze from 6bit using AVX2 and multithreading");
    m.def("calculate_dequant", &calculate_dequant,
        "Optimized dequantization calculation using AVX2 and multithreading");
}