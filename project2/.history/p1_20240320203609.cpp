#include<iostream>
using namespace std;
const int N=16;
int a[N];
void init()
{
    for(int i=0;i<N;i++)
    {
        a[i]=i;
    }
}
int main()
{
    init();
    int sum=0;
    for(int i=0;i<N;i++)
    {
        sum+=a[i];
    }
    cout<<sum;
}