#include<iostream>
#include<vector>
#include<random>
#include<cmath>
#include<algorithm>
#include<chrono>
#include<immintrin.h>

using namespace std;

struct Node {
    vector<float> dimensions;
};

inline float VectorizedDistance(const Node& x, const Node& z, int n) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(&x.dimensions[i]);
        __m256 z_vec = _mm256_loadu_ps(&z.dimensions[i]);
        __m256 diff = _mm256_sub_ps(x_vec, z_vec);
        __m256 squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
    }

    // 将 AVX 向量降维为 SSE 向量
    __m128 sum_low = _mm256_castps256_ps128(sum);
    __m128 sum_high = _mm256_extractf128_ps(sum, 1);

    // 对 SSE 向量进行规约
    __m128 sum_total = _mm_add_ps(_mm_hadd_ps(_mm_hadd_ps(sum_low, sum_low), sum_low),
                                 _mm_hadd_ps(_mm_hadd_ps(sum_high, sum_high), sum_high));
    float result = _mm_cvtss_f32(_mm_sqrt_ps(sum_total));
    return result;
}

int main() {
    // 分配内存给 Node 类型的变量 x 和 y
    Node x, y;

    // 给 x 和 y 的 dimensions 成员变量赋值
    x.dimensions = {1, 2, 3, 4, 5};
    y.dimensions = {5, 6, 7, 8, 9};

    // 测试 VectorizedDistance 函数
    float distance = VectorizedDistance(x, y, 5);
    cout << "Distance between x and y: " << distance << endl;

    return 0;
}
