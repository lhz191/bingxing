#include <iostream>
#include <chrono>
#include <cstdint>

using namespace std;
using namespace chrono;

const int N = 65536;
int a[N];

void init() {
    for(int i = 0; i < N; i++) {
        a[i] = i;
    }
}

int main() {
    init();

    // 使用系统时钟
    auto start = high_resolution_clock::now(); // 开始计时

    int64_t sum = 0; // 将 sum 的类型改为 int64_t，这是 C++ 中固定大小的整数类型

    for(int p = 1; p <= 100; p++) {
        // init();
        sum = 0;
        for(int i = 0; i < N; i++) {
            sum += a[i];
        }
    }

    auto stop = high_resolution_clock::now(); // 结束计时

    auto duration = duration_cast<microseconds>(stop - start); // 计算执行时间

    cout << "Sum: " << sum << endl;
    cout << "Time taken: " << duration.count() / 1000.0 << "ms" << endl; // 将纳秒转换为毫秒

    return 0;
}




