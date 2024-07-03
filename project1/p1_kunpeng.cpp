#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

const int N = 1200; // matrix size
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
    init(N);

    // Start time
    auto start = high_resolution_clock::now();

    for(int p=1;p<=80;p++){
        for(int i = 0; i < N; i++)
            col_sum[i] = 0.0;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                col_sum[i] += b[j][i]*a[j];
            }
        }
    }

    // End time
    auto stop = high_resolution_clock::now();

    // Calculate duration
    auto duration = duration_cast<microseconds>(stop - start);

    // Output the time elapsed in milliseconds
    cout << "Col: " << duration.count() /1000.0<< "ms" << endl;
//    for(int i=0;i<N;i++)
//    {
    //    cout<<col_sum[i]<<" ";
//    }
    return 0;
}
