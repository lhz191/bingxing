#include <iostream>
#include <windows.h>

using namespace std;

const int N = 262144/2;
int a[N];

void init() {
    for(int i = 0; i < N; i++) {
        a[i] = i;
    }
}

int main() {
    init();

    LARGE_INTEGER frequency; // 用于存储性能计数器的频率
    LARGE_INTEGER start, stop; // 用于存储开始和结束时间

    QueryPerformanceFrequency(&frequency); // 获取性能计数器的频率
    long long sum1 = 0;
    long long sum2 = 0;
    long long sum3 = 0;
    long long sum4 = 0;
    QueryPerformanceCounter(&start); // 开始计时 
    for(int p = 1; p <= 100; p++) {
        init();
        sum1=0,sum2=0,sum3=0,sum4=0;
        for(int i = 0; i < N; i += 4) {
            sum1 += a[i];
            sum2 += a[i + 1];
            sum1 += a[i+2];
            sum2 += a[i + 3];
        }
        sum1+=sum2;
    }

    QueryPerformanceCounter(&stop); // 结束计时

    double elapsedTime = (stop.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart; // 计算执行时间，单位为毫秒

    cout << "Sum: " << sum1 << endl;
    cout << "Time taken: " << elapsedTime << "ms" << endl;

    return 0;
}
