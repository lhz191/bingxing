#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int N = 262144;
int a[N];

void init() {
    for(int i = 0; i < N; i++) {
        a[i] = i;
    }
}

int main() {
    init();

    high_resolution_clock::time_point start, end;
    duration<double, milli> elapsed_time;

    start = high_resolution_clock::now(); // 开始计时

    long long sum1 = 0; 
    long long sum2 = 0;
    for(int p = 1; p <= 50; p++) {
        sum1 = 0;
        sum2 = 0;
        for(int i = 0; i < N; i += 2) {
            sum1 += a[i];
            sum2 += a[i + 1];
        }
        sum1 += sum2;
    }

    end = high_resolution_clock::now(); // 结束计时

    elapsed_time = duration_cast<duration<double, milli>>(end - start); // 计算执行时间，单位为毫秒

    cout << "Sum: " << sum1 << endl;
    cout << "Time taken: " << elapsed_time.count() << "ms" << endl;

    return 0;
}
