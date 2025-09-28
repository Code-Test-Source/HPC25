#include "calGF.h"
#include <omp.h>
#include <immintrin.h>

void GravAttraction(const POLYHEDRON &p, double* r, double* F)
{
    int i, j;
    
    // 初始化结果
    F[0] = 0.0; F[1] = 0.0; F[2] = 0.0;
    
    double *RAY = new double[p.NumVerts * 4];
    
    // 预计算射线向量 - 使用向量化
    #pragma omp parallel for
    for (i = 0; i < p.NumVerts; i++) {
        double dx = p.Vertices[i] - r[0];
        double dy = p.Vertices[i + p.NumVerts] - r[1];
        double dz = p.Vertices[i + 2 * p.NumVerts] - r[2];
        double distance = sqrt(dx * dx + dy * dy + dz * dz);
        
        RAY[i] = dx;
        RAY[i + p.NumVerts] = dy;
        RAY[i + 2 * p.NumVerts] = dz;
        RAY[i + 3 * p.NumVerts] = distance;
    }

    double FF[3] = {0.0, 0.0, 0.0};
    double EE[3] = {0.0, 0.0, 0.0};
    
    // 并行处理面
    #pragma omp parallel for reduction(+:FF[0], FF[1], FF[2])
    for (i = 0; i < p.NumFaces; i++) {
        double tmpVS[9]; // 3x3矩阵
        double tmpV1[3], tmpV2[3];
        double face_FF[3] = {0.0, 0.0, 0.0};
        
        for (j = 0; j < 3; j++) {
            int vert_idx = p.Faces[i + j * p.NumFaces];
            double inv_dist = 1.0 / RAY[vert_idx + 3 * p.NumVerts];
            
            tmpVS[0 + 3 * j] = RAY[vert_idx] * inv_dist;
            tmpVS[1 + 3 * j] = RAY[vert_idx + p.NumVerts] * inv_dist;
            tmpVS[2 + 3 * j] = RAY[vert_idx + 2 * p.NumVerts] * inv_dist;
        }
        
        // 计算叉积
        tmpV2[0] = tmpVS[1 + 1*3] * tmpVS[2 + 2*3] - tmpVS[2 + 1*3] * tmpVS[1 + 2*3];
        tmpV2[1] = tmpVS[2 + 1*3] * tmpVS[0 + 2*3] - tmpVS[0 + 1*3] * tmpVS[2 + 2*3];
        tmpV2[2] = tmpVS[0 + 1*3] * tmpVS[1 + 2*3] - tmpVS[1 + 1*3] * tmpVS[0 + 2*3];
        
        double tmp1 = tmpV2[0] * tmpVS[0] + tmpV2[1] * tmpVS[1] + tmpV2[2] * tmpVS[2];
        double tmp2 = tmpVS[0] * tmpVS[0+3] + tmpVS[1] * tmpVS[1+3] + tmpVS[2] * tmpVS[2+3] +
                     tmpVS[0+3] * tmpVS[0+2*3] + tmpVS[1+3] * tmpVS[1+2*3] + tmpVS[2+3] * tmpVS[2+2*3] +
                     tmpVS[0] * tmpVS[0+2*3] + tmpVS[1] * tmpVS[1+2*3] + tmpVS[2] * tmpVS[2+2*3] + 1.0;
        
        double omegaf = 2.0 * atan2(tmp1, tmp2);
        
        // 使用第一个顶点计算
        int first_vert = p.Faces[i];
        double dot_product = RAY[first_vert] * p.FaceNormVecs[i] +
                           RAY[first_vert + p.NumVerts] * p.FaceNormVecs[i + p.NumFaces] +
                           RAY[first_vert + 2 * p.NumVerts] * p.FaceNormVecs[i + 2 * p.NumFaces];
        
        face_FF[0] = p.FaceNormVecs[i] * dot_product * omegaf;
        face_FF[1] = p.FaceNormVecs[i + p.NumFaces] * dot_product * omegaf;
        face_FF[2] = p.FaceNormVecs[i + 2 * p.NumFaces] * dot_product * omegaf;
        
        FF[0] += face_FF[0]; FF[1] += face_FF[1]; FF[2] += face_FF[2];
    }
    
    // 并行处理边
    #pragma omp parallel for reduction(+:EE[0], EE[1], EE[2])
    for (i = 0; i < p.NumEdges; i++) {
        double edge_EE[3] = {0.0, 0.0, 0.0};
        
        double tmp1 = RAY[p.Edges[i] + 3 * p.NumVerts] + RAY[p.Edges[i + p.NumEdges] + 3 * p.NumVerts];
        double le = log((tmp1 + p.EdgeLens[i]) / (tmp1 - p.EdgeLens[i]));
        
        double re_dot[2];
        for (j = 0; j < 2; j++) {
            int edge_vert = p.Edges[i + j * p.NumEdges];
            re_dot[j] = RAY[edge_vert] * p.EdgeNormVecs[i + (4*j+1) * p.NumEdges] +
                       RAY[edge_vert + p.NumVerts] * p.EdgeNormVecs[i + (4*j+2) * p.NumEdges] +
                       RAY[edge_vert + 2 * p.NumVerts] * p.EdgeNormVecs[i + (4*j+3) * p.NumEdges];
            
            int face_idx = (int)p.EdgeNormVecs[i + (4*j) * p.NumEdges];
            edge_EE[0] += p.FaceNormVecs[face_idx] * re_dot[j] * le;
            edge_EE[1] += p.FaceNormVecs[face_idx + p.NumFaces] * re_dot[j] * le;
            edge_EE[2] += p.FaceNormVecs[face_idx + 2 * p.NumFaces] * re_dot[j] * le;
        }
        
        EE[0] += edge_EE[0]; EE[1] += edge_EE[1]; EE[2] += edge_EE[2];
    }
    
    F[0] = FF[0] - EE[0];
    F[1] = FF[1] - EE[1];
    F[2] = FF[2] - EE[2];
    
    delete[] RAY;
}