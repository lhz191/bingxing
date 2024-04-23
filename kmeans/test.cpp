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

// SIMD 版本的计算距离函数
inline float VectorizedDistance(const Node& x, const Node& z, int n) {
    __m128 sum = _mm_setzero_ps();
    for (int i = 0; i < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(&x.dimensions[i]);
        __m128 z_vec = _mm_loadu_ps(&z.dimensions[i]);
        __m128 diff = _mm_sub_ps(x_vec, z_vec);
        __m128 squared = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, squared);
    }
            // 处理剩余的数据
        float remaining_sum = 0.0f;
        for (int i = n - (n % 4); i < n; i++) {
            float diff = x.dimensions[i] - z.dimensions[i];
            remaining_sum += diff * diff;
        }
    float result = _mm_cvtss_f32(_mm_sqrt_ps(_mm_hadd_ps(_mm_hadd_ps(sum, sum), sum)));
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
