#include <iostream>
#include <windows.h>

using namespace std;

const long long N = 262144; // 使用 long long 类型
long long a[N];

void init() {
    for(long long i = 0; i < N; i++) { // 使用 long long 类型
        a[i] = i;
    }
}

void rec(long long n) { // 使用 long long 类型
    if(n == 1) {
        return;
    }
    else {
        for(long long i = 0; i < n / 2; i++) { // 使用 long long 类型
            a[i] += a[n - i - 1];
        }
        n = n / 2;
        rec(n);
    }
}

int main() {
    init();

    LARGE_INTEGER frequency;
    LARGE_INTEGER start, stop;

    QueryPerformanceFrequency(&frequency);
   long long sum = 0; // 使用 long long 类型
    QueryPerformanceCounter(&start);
    for(int p = 1; p <= 50; p++) {
        sum = 0;
        init();
        rec(N);
        sum = a[0];
    }

    QueryPerformanceCounter(&stop);

    double elapsedTime = (stop.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;

    cout << "Sum: " << sum << endl;
    cout << "Time taken: " << elapsedTime << "ms" << endl;

    return 0;
}







































