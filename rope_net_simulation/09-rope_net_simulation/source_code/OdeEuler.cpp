#include "OdeEuler.h"
#include <omp.h>

void OdeEuler(double *y0, const double dt, const double tstart, const double tend, POLYHEDRON &p, FNS &fns, const double *SurfacePara, char* solution, double *FAC)
{
    int n = (int) floor((tend - tstart) / dt);
    double* result = new double[2 * fns.NP * 6];
    double* acceleration = new double[fns.NP * 3];
    
    // 初始化结果
    #pragma omp parallel for
    for (int j = 0; j < 6 * fns.NP; j++) {
        result[j] = y0[j];
    }
    
    FILE *infile = fopen(solution, "w");
    
    // 写入初始条件
    for (int j = 0; j < fns.NP * 6 - 1; j++) {
        fprintf(infile, "%.15g ", result[j]);
    }
    fprintf(infile, "%.15g\n", result[6 * fns.NP - 1]);
    
    double* theta = new double[fns.NP];
    double* phi = new double[fns.NP];
    double* r = new double[fns.NP];
    
    // 主时间循环
    for (int i = 1; i < n; i++) {
        // 计算加速度
        calAcceleration(result, acceleration, p, fns, SurfacePara, FAC, theta, phi, r);
        
        // 并行更新速度和位置
        #pragma omp parallel for
        for (int j = 0; j < fns.NP; j++) {
            int vel_idx = 3 * j + fns.NP * 3;
            int pos_idx = 3 * j;
            int acc_idx = 3 * j;
            
            // 更新速度
            result[6 * fns.NP + vel_idx]     = result[vel_idx]     + dt * acceleration[acc_idx];
            result[6 * fns.NP + vel_idx + 1] = result[vel_idx + 1] + dt * acceleration[acc_idx + 1];
            result[6 * fns.NP + vel_idx + 2] = result[vel_idx + 2] + dt * acceleration[acc_idx + 2];
            
            // 更新位置
            result[6 * fns.NP + pos_idx]     = result[pos_idx]     + dt * result[6 * fns.NP + vel_idx];
            result[6 * fns.NP + pos_idx + 1] = result[pos_idx + 1] + dt * result[6 * fns.NP + vel_idx + 1];
            result[6 * fns.NP + pos_idx + 2] = result[pos_idx + 2] + dt * result[6 * fns.NP + vel_idx + 2];
        }
        
        // 定期输出
        if (i % 1000 == 0) {
            for (int j = 0; j < fns.NP * 6 - 1; j++) {
                fprintf(infile, "%.15g ", result[6 * fns.NP + j]);
            }
            fprintf(infile, "%.15g\n", result[6 * fns.NP + (6 * fns.NP - 1)]);
        }
        
        // 交换缓冲区
        #pragma omp parallel for
        for (int j = 0; j < 6 * fns.NP; j++) {
            result[j] = result[j + 6 * fns.NP];
        }
    }
    
    delete[] theta;
    delete[] phi;
    delete[] r;
    delete[] result;
    delete[] acceleration;
    
    fclose(infile);
}