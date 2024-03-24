#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

const long long N = 262144; // 使用 long long 类型
long long a[N];

void init() {
    for(long long i = 0; i < N; i++) { // 使用 long long 类型
        a[i] = i;
    }
}

void rec(long long n) { 
    if(n == 1) {
        return;
    }
    else {
        for(long long i = 0; i < n / 2; i++) { 
            a[i] += a[n - i - 1];
        }
        n = n / 2;
        rec(n);
    }
}

int main() {
    init();
    long long sum = 0; // 使用 long long 类型
    // 使用系统时钟
    auto start = high_resolution_clock::now(); // 开始计时

    for(int p = 1; p <= 50; p++) {
        sum = 0;
        // init();
        rec(N);
        sum = a[0];
    }

    auto stop = high_resolution_clock::now(); // 结束计时

    auto duration = duration_cast<microseconds>(stop - start); // 计算执行时间

    cout << "Sum: " << sum << endl;
    cout << "Time taken: " << duration.count() / 1000.0 << "ms" << endl; // 将纳秒转换为毫秒秒

    return 0;
}
