#include <iostream>
#include <windows.h>

using namespace std;

const int N = 100; // matrix size
double b[N][N], col_sum[N];
double a[N];
void init(int n) // generate a N*N matrix
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            b[i][j] = i + j;
    for(int i=0;i<N;i++)
    {
        a[i]=i;
    }
}

int main()
{
    long long head, tail, freq; // timers
    init(N);

    // similar to CLOCKS_PER_SEC
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);

    // start time
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for(int p=1;p<=518;p++){
    for(int i = 0; i < N; i++)
        {col_sum[i] = 0.0;}
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j+=5){
     if(j<N-4){
         col_sum[j] += b[i][j]*a[i];
         col_sum[j+1] += b[i][j+1]*a[i];
         col_sum[j+2] += b[i][j+2]*a[i];
         col_sum[j+3] += b[i][j+3]*a[i];
         col_sum[j+4] += b[i][j+4]*a[i];
     }
     else{
         for(int k=j;k<N;k++){
           col_sum[k]+=b[i][k]*a[i];  
         }
     }
 }
    }
    }

    // end time
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);

    // output the time elapsed in milliseconds
    cout << "Col: " << (tail - head) * 1000.0 / freq << "ms" << endl;
    // for(int i=0;i<N;i++)
    // {
        // cout<<col_sum[i]<<" ";
    // }

    return 0;
}
