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
    for (int i = 0;i<N; i+=5){
        for (int j = 0; j < N; j++){
            if(i<N-4){
                col_sum[i] += b[j][i]*a[j];
                col_sum[i+1] += b[j][i+1]*a[j];
                col_sum[i+2] += b[j][i+2]*a[j];
                col_sum[i+3] += b[j][i+3]*a[j];
                col_sum[i+4] += b[j][i+4]*a[j];
            }
            else{
                for(int k=i;k<N;k++){
                  col_sum[k]+=b[j][k]*a[j];  
                }
            }
        }
    }
    }

    // end time
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);

    // output the time elapsed in milliseconds
    cout << "Col: " << (tail - head) * 1000.0 / freq << "ms" << endl;
//    for(int i=0;i<N;i++)
//    {
    //    cout<<col_sum[i]<<" ";
//    }
    return 0;
}
