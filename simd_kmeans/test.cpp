#include <iostream>
#include<iomanip>
#include <vector>
#include <cmath>
#include <complex>
#include<Windows.h>
#include<cstring>
#include<cstdio>
#include<algorithm>
using namespace std;

const double PI = acos(-1.0);
/*
// 递归实现FFT
void fft(vector<complex<double>>& a, bool inv) {
	int n = a.size();
	if (n == 1) {
		return;
	}

	//分治
	vector<complex<double>> a0(n / 2), a1(n / 2);
	for (int i = 0, j = 0; i < n; i += 2, j++) {
		a0[j] = a[i];
		a1[j] = a[i + 1];
	}
	fft(a0, inv);
	fft(a1, inv);

	//FFT
	double angle = 2 * PI / n * (inv ? -1 : 1);
	complex<double> w(1), wn(cos(angle), sin(angle));
	for (int i = 0; i < n / 2; i++) {
		a[i] = a0[i] + w * a1[i];
		a[i + n / 2] = a0[i] - w * a1[i];
		w *= wn;
	}
}
// FFT乘法
vector<int> multiply(vector<int> a, vector<int> b) {
	int n = 1;
	// 将多项式的项数变为2的整数次幂
	while (n < a.size() + b.size()) {
		n *= 2;
	}
	a.resize(n), b.resize(n);

	vector<complex<double>> c(n), d(n);
	for (int i = 0; i < n; i++) {
		c[i] = complex<double>(a[i], 0);
		d[i] = complex<double>(b[i], 0);
	}

	// 求原多项式的FFT
	fft(c, false), fft(d, false);
	for (int i = 0; i < n; i++) {
		c[i] *= d[i];
	}

	// 求乘法结果的IFFT
	fft(c, true);

	// 将IFFT中与逆矩阵相差的1/n乘进去
	vector<int> res(n);
	for (int i = 0; i < n; i++) {
		res[i] = (int)(c[i].real() / n + 0.5);
	}

	// 处理进位
	int carry = 0;
	for (int i = 0; i < n; i++) {
		res[i] += carry;
		carry = res[i] / 10;
		res[i] %= 10;
	}

	// 去高位的0
	while (res.size() > 1 && res.back() == 0) {
		res.pop_back();
	}
	return res;
}
*/

/*
// 将大整数字符串转换为vector<int>
vector<int> to_vector(string s) {
	vector<int> res;
	for (int i = s.size() - 1; i >= 0; i--) {
		res.push_back(s[i] - '0');
	}
	return res;
}

// 将vector<int>转换为大整数字符串
string to_string(vector<int> a) {
	string res;
	for (int i = a.size() - 1; i >= 0; i--) {
		res += to_string(a[i]);
	}
	return res;
}

*/

int i, x, len, j;
int deg1, deg2;
const int N = 4e4 + 3;
vector<complex<double>> A(N, { 0,0 });
vector<complex<double>> B(N, { 0,0 });


void FFT(vector<complex<double>>& A, int n, int op) {
	if (n == 1)return;
	vector < complex<double>> A1(n / 2, { 0,0 });
	vector < complex<double>> A2(n / 2, { 0,0 });
	for (int i = 0; i < n / 2; i++) {
		A1[i] = A[i * 2], A2[i] = A[i * 2 + 1];
	}
	FFT(A1, n / 2, op), FFT(A2, n / 2, op);
	complex<double> w1 = { cos(2 * PI / (double)n), sin(2 * PI / (double)n) * (double)op };
	complex<double> wk = { 1, 0 };
	for (int i = 0; i < n / 2; i++) {
		A[i] = A1[i] + A2[i] * wk;
		A[i + n / 2] = A1[i] - A2[i] * wk;
		wk *= w1;
	}
}

int main() {
	string s1, s2;
	long long head1, tail1, freq1; // 计时器1
	// 获取计时器频率，类似于 CLOCKS_PER_SEC
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq1);
	// 记录起始时间
	QueryPerformanceCounter((LARGE_INTEGER*)&head1);



	s1 = "1234125432625543565545345654535654645623573674545362654623796567249457623751063721795612378172315976127109873247934561271897562317893471";
	s2 = "3157654657654365374756346437876554645768684576564736846754567875645643645673217456759347231709576213719576423717965100967";
	deg1 = s1.length()-1;
	deg2 = s2.length()-1;

	for (int i = 1; i <= 10; i++)
	{
		deg1 = s1.length() - 1;
		deg2 = s2.length() - 1;
		for (int i = 0; i <= deg1; i++)A[i].real(s1[i] - '0');
		for (int i = 0; i <= deg2; i++)B[i].real(s2[i] - '0');
		for (deg1 = deg1 + deg2, deg2 = 1; deg2 <= deg1; deg2 <<= 1);
		FFT(A, deg2, 1), FFT(B, deg2, 1);
		for (int i = 0; i < deg2; i++)A[i] *= B[i];
		FFT(A, deg2, -1);
	}

	QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
	cout << "平均运行时间：" << (tail1 - head1) * 1000 / freq1 << "ms" << endl;


	long long head2, tail2, freq2; // 计时器1
	// 获取计时器频率，类似于 CLOCKS_PER_SEC


    //SSE指令集使用 movaps 指令将 XMM0 的值写入内存,一次写 16 字节
    pxor    xmm0, xmm0
    movaps  XMMWORD PTR [rax], xmm0
    movaps  xmm0, XMMWORD PTR [rbp-128]
    //SSE指令集使用 cmpeqps 指令比较两个 XMM 寄存器中的 4 个浮点数是否相等,结果存储在 XMM0 寄存器中。
    cmpeqps xmm0, XMMWORD PTR [rbp-144]
    movaps  XMMWORD PTR [rbp-96], xmm0
    //使用 divps 指令将 XMM0 寄存器中的 4 个浮点数除以 XMM1 寄存器中的 4 个浮点数
    movaps  xmm0, XMMWORD PTR [rbp-336]
    divps   xmm0, XMMWORD PTR [rbp-352]
    movaps  XMMWORD PTR [rbp-80], xmm0

	QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
	// 记录起始时间
	QueryPerformanceCounter((LARGE_INTEGER*)&head2);
	for (int s = 1; s< 10; s++) {
		char a1[10001] = "12341254326255435655453456545356546456235736745453626546";
		char b1[10001] = "315765465765436537475634643787655464576868457656473684675456787564564364";
		int a[10001], b[10001],  c[10001];
		int lena = strlen(a1);//每个部分都很清楚
		int lenb = strlen(b1);//这只是方便你们复制
		for (i = 1; i <= lena; i++)a[i] = a1[lena - i] - '0';
		for (i = 1; i <= lenb; i++)b[i] = b1[lenb - i] - '0';
		for (i = 1; i <= lenb; i++)
			for (j = 1; j <= lena; j++)
				c[i + j - 1] += a[j] * b[i];
		for (i = 1; i < lena + lenb; i++)
			if (c[i] > 9)
			{
				c[i + 1] += c[i] / 10;
				c[i] %= 10;
			}
		len = lena + lenb;
		while (c[len] == 0 && len > 1)len--;
		i=x=len=j=0;
		//for (i = len; i >= 1; i--)cout << c[i];
	}

	QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
	cout << "平均运行时间：" << (tail2 - head2) * 1000 / freq2 << "ms" << endl;
	//cin >> s1 >> s2;
	/*vector<int> a = to_vector(s1), b = to_vector(s2);
	vector<int> c = multiply(a, b);
	cout << to_string(c) << endl;*/
	return 0;
}