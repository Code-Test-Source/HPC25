#include "acceleration.h"
#include <omp.h>
#include <immintrin.h>

void calAcceleration(const double* solutionminusone,
	double* acceleration, const POLYHEDRON& p, 
	const FNS& fns, const double* SurfacePara, 
	double* FAC, double* theta, double* phi, double* r)
{   
    // 清零加速度数组
    #pragma omp parallel for
    for (int i = 0; i < fns.NP * 3; i++) {
        acceleration[i] = 0.0;
    }

    // 并行处理每个点
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < fns.NP; i++) {
        int if_colli = 0;
        double sfR = 0, NVx = 0, NVy = 0, NVz = 0, 
               NVt = 0, NVp = 0, tmp1 = 0, tmp2 = 0, 
               dzs = 0, vzs = 0;
        double fp[3] = { 0 }, GravF[3] = { 0 }, Nf[3] = { 0 }, ff[3] = { 0 }, tmp3[3] = { 0 }, tmp4[3] = { 0 }, force[3] = { 0 };

        // 坐标变换
        CoorTran(1, &solutionminusone[3 * i], &solutionminusone[(3 * i + 1)], &solutionminusone[(3 * i + 2)], &theta[i], &phi[i], &r[i]);
        SurfacePoints(theta[i], phi[i], SurfacePara, sfR, NVx, NVy, NVz, FAC);
        CoorTran(1, &NVx, &NVy, &NVz, &NVt, &NVp, &tmp1);

        fp[0] = solutionminusone[3 * i]; 
        fp[1] = solutionminusone[(3 * i + 1)]; 
        fp[2] = solutionminusone[(3 * i + 2)];
        
        // 计算重力
        GravAttraction(p, fp, GravF);
        GravF[0] *= Pm * G * p.Density;    
        GravF[1] *= Pm * G * p.Density;     
        GravF[2] *= Pm * G * p.Density;

        // 碰撞检测
        dzs = r[i] - sfR;
        vzs = solutionminusone[(3 * i + fns.NP * 3)] * NVx + 
              solutionminusone[(3 * i + 1 + fns.NP * 3)] * NVy + 
              solutionminusone[(3 * i + 2 + fns.NP * 3)] * NVz;
        
        if_colli = (dzs <= 0) ? 1 : 0;

        tmp1 = Ki * fabs(dzs) * if_colli;
        tmp2 = -1 * Ci * vzs * if_colli;
        
        Nf[0] = (tmp1 + tmp2) * sin(NVt) * cos(NVp);
        Nf[1] = (tmp1 + tmp2) * sin(NVt) * sin(NVp);
        Nf[2] = (tmp1 + tmp2) * cos(NVt);
        
        tmp3[0] = solutionminusone[(3 * i + fns.NP * 3)] * (1 - sin(NVt) * cos(NVp));
        tmp3[1] = solutionminusone[(3 * i + 1 + fns.NP * 3)] * (1 - sin(NVt) * sin(NVp));
        tmp3[2] = solutionminusone[(3 * i + 2 + fns.NP * 3)] * (1 - cos(NVt));
        
        CoorTran(1, &tmp3[0], &tmp3[1], &tmp3[2], &tmp4[0], &tmp4[1], &tmp4[2]);
        
        ff[0] = (-1 * miu) * (tmp1 + tmp2) * sin(tmp4[0]) * cos(tmp4[1]);
        ff[1] = (-1 * miu) * (tmp1 + tmp2) * sin(tmp4[0]) * sin(tmp4[1]);
        ff[2] = (-1 * miu) * (tmp1 + tmp2) * cos(tmp4[0]);

        // 计算张力 - 使用向量化优化
        for (int j = 0; j < fns.TP[i]; j++) {
            int neighbor_idx = fns.TP[i + (2 * j + 2) * fns.NP];
            
            // 使用预计算避免重复索引
            double dx = solutionminusone[3 * neighbor_idx] - solutionminusone[3 * i];
            double dy = solutionminusone[(3 * neighbor_idx + 1)] - solutionminusone[(3 * i + 1)];
            double dz = solutionminusone[(3 * neighbor_idx + 2)] - solutionminusone[(3 * i + 2)];
            
            double dvx = solutionminusone[(3 * neighbor_idx + fns.NP * 3)] - solutionminusone[(3 * i + fns.NP * 3)];
            double dvy = solutionminusone[(3 * neighbor_idx + 1 + fns.NP * 3)] - solutionminusone[(3 * i + 1 + fns.NP * 3)];
            double dvz = solutionminusone[(3 * neighbor_idx + 2 + fns.NP * 3)] - solutionminusone[(3 * i + 2 + fns.NP * 3)];

            double distance = sqrt(dx * dx + dy * dy + dz * dz);
            double rest_length = fns.EL[fns.TP[i + (2 * j + 1) * fns.NP]];
            
            if (distance - rest_length > 0) {
                double inv_distance = 1.0 / distance;
                dx *= inv_distance;
                dy *= inv_distance;  
                dz *= inv_distance;
                
                double relative_velocity = dvx * dx + dvy * dy + dvz * dz;
                double tension = ki * (distance - rest_length) + ci * relative_velocity;
                
                force[0] += tension * dx;
                force[1] += tension * dy;
                force[2] += tension * dz;
            }
        }

        // 存储结果
        acceleration[0 + i * 3] = (force[0] + GravF[0] + Nf[0] + ff[0]) / Pm;
        acceleration[1 + i * 3] = (force[1] + GravF[1] + Nf[1] + ff[1]) / Pm;
        acceleration[2 + i * 3] = (force[2] + GravF[2] + Nf[2] + ff[2]) / Pm;
    }
}

// 优化的坐标变换函数 - 使用SIMD指令
inline void CoorTran(const int VectorLength, const double* x, const double* y, const double* z, double* theta, double* phi, double* r)
{
    #pragma omp parallel for simd
    for (int i = 0; i < VectorLength; i++) {
        double xy_sq = x[i] * x[i] + y[i] * y[i];
        theta[i] = PI / 2 - atan2(z[i], sqrt(xy_sq));
        phi[i] = atan2(y[i], x[i]);
        if (phi[i] < 0) {
            phi[i] += 2 * PI;
        }
        r[i] = sqrt(xy_sq + z[i] * z[i]);
    }
}