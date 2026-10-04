/* Prints how C's printf formats doubles with the conversions C psmc uses
 * in its .psmc output (%lf, %.9lf, %.10lf, %g), one value per line:
 *
 *   <IEEE-754 bits as 16 hex digits> TAB %lf TAB %.9lf TAB %.10lf TAB %g
 *
 * The values are special cases (zeros, infinities, NaNs, exact ties, %g
 * style boundaries) and pseudo-random doubles between 2^-70 and 2^70.
 * Build and run: cc -O1 -o printf_probe printf_probe.c && ./printf_probe
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint64_t state = 0x9E3779B97F4A7C15ULL;

static uint64_t splitmix64(void)
{
	uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

static void print(double x)
{
	uint64_t bits;
	memcpy(&bits, &x, sizeof bits);
	printf("%016llx\t%lf\t%.9lf\t%.10lf\t%g\n", (unsigned long long)bits, x, x, x, x);
}

int main(void)
{
	static const double fixed[] = {
		0.0, -0.0, 1.0, -1.0, 15.0, 4.0, 5.0, 0.1, 1e-9, -1e-9, 1e-7, -1e-7,
		/* exact ties at 6, 9 and 10 decimals and at 6 significant digits */
		0.0078125, 0.25, 0.5, 1.5, 2.5, 4.5, 0.001953125, 0.0009765625,
		0.00048828125, 123456.5, 999999.5, 1234565.0, 0.5e-6, 1.5e-6, 2.5e-6,
		/* %g switches between %f and %e style at exponents -5/-4 and 5/6 */
		1e-4, 0.99999949999e-4, 0.9999995e-4, 1e-5, 99999.95, 999999.0,
		999999.4, 1e6, 1e5, 123456.0, 1234567.0, 0.000123456789, 2.0 / 3.0,
		/* values typical of .psmc files */
		0.050078200, 0.012519550, 18.405650854, -25491.264029, 1878.032457,
		0.1439384744, 1e-30, 1e30, 5e-324, 2.2250738585072014e-308,
	};
	int i, n = sizeof fixed / sizeof fixed[0];
	for (i = 0; i < n; ++i) print(fixed[i]);
	print(INFINITY);
	print(-INFINITY);
	print(NAN);
	print(-NAN);
	for (i = 0; i < 2000; ++i) {
		uint64_t r = splitmix64();
		uint64_t sign = r >> 63;
		uint64_t exponent = 1023 - 70 + (splitmix64() % 141);
		uint64_t bits = (sign << 63) | (exponent << 52) | (r & 0xFFFFFFFFFFFFFULL);
		double x;
		memcpy(&x, &bits, sizeof x);
		print(x);
	}
	return 0;
}
