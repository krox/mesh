#ifndef MESH_RANDOM_H
#define MESH_RANDOM_H

/**
 * Random number generators to be used in the Monte-Carlo simulation.
 */

#include <stdint.h>

/**
 * Originally written in 2015 by Sebastiano Vigna (vigna@acm.org).
 * public domain, taken from http://xoroshiro.di.unimi.it/splitmix64.c
 */
class splitmix64
{
	uint64_t s; // all values are allowed

  public:
	splitmix64() : s(0) {}
	explicit splitmix64(uint64_t x) : s(x) {}

	uint64_t min() const { return 0; }
	uint64_t max() const { return UINT64_MAX; }

	void seed(uint64_t x) { s = x; }

	uint64_t operator()()
	{
		s += 0x9e3779b97f4a7c15;
		uint64_t z = s;
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		return z ^ (z >> 31);
	}
};

/**
 * This is xoroshiro128+, version 1.0.
 * Originally written in 2016-2018 by David Blackman and Sebastiano Vigna.
 * public domain, taken from http://xoroshiro.di.unimi.it/xoroshiro128plus.c
 */
class xoroshiro128plus
{
	uint64_t s[2]; // should not be all zeroes

	static inline uint64_t rotl(uint64_t x, int k)
	{
		// NOTE: compiler will optimize this to a single instruction
		return (x << k) | (x >> (64 - k));
	}

  public:
	xoroshiro128plus() { seed(0); }
	explicit xoroshiro128plus(uint64_t x) { seed(x); }

	uint64_t min() const { return 0; }
	uint64_t max() const { return UINT64_MAX; }

	/** set the internal state to some seed value */
	void seed(uint64_t x)
	{
		splitmix64 gen(x);
		s[0] = gen();
		s[1] = gen();
	}

	/** generate next value in the random sequence */
	uint64_t operator()()
	{
		uint64_t s0 = s[0];
		uint64_t s1 = s[1];
		uint64_t result = s0 + s1;

		s1 ^= s0;
		s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
		s[1] = rotl(s1, 37);

		return result;
	}

	/** discards 2^64 values of the random sequence */
	void jump()
	{
		static const uint64_t JUMP[] = {0xdf900294d8f554a5, 0x170865df4b3201fc};

		uint64_t s0 = 0;
		uint64_t s1 = 0;
		for (int i = 0; i < 2; i++)
			for (int b = 0; b < 64; b++)
			{
				if (JUMP[i] & UINT64_C(1) << b)
				{
					s0 ^= s[0];
					s1 ^= s[1];
				}
				(*this)();
			}

		s[0] = s0;
		s[1] = s1;
	}
};

/**
 * This is xoshiro256**, version 1.0.
 * Originally written in 2018 by David Blackman and Sebastiano Vigna.
 * public domain, from http://http://xoshiro.di.unimi.it/xoshiro256starstar.c
 */
class xoshiro256
{
	uint64_t s[4]; // should not be all zeroes

	static inline uint64_t rotl(uint64_t x, int k)
	{
		// NOTE: compiler will optimize this to a single instruction
		return (x << k) | (x >> (64 - k));
	}

  public:
	xoshiro256() { seed(0); }
	explicit xoshiro256(uint64_t x) { seed(x); }

	uint64_t min() const { return 0; }
	uint64_t max() const { return UINT64_MAX; }

	/** set the internal state to some seed value */
	void seed(uint64_t x)
	{
		splitmix64 gen(x);
		s[0] = gen();
		s[1] = gen();
		s[2] = gen();
		s[3] = gen();
	}

	/** generate next value in the random sequence */
	uint64_t operator()()
	{
		uint64_t result = rotl(s[1] * 5, 7) * 9;

		uint64_t t = s[1] << 17;
		s[2] ^= s[0];
		s[3] ^= s[1];
		s[1] ^= s[2];
		s[0] ^= s[3];
		s[2] ^= t;
		s[3] = rotl(s[3], 45);

		return result;
	}

	/** discards 2^128 values of the random sequence */
	void jump()
	{
		static const uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
		                                0xa9582618e03fc9aa, 0x39abdc4529b1661c};

		uint64_t s0 = 0;
		uint64_t s1 = 0;
		uint64_t s2 = 0;
		uint64_t s3 = 0;
		for (int i = 0; i < 4; i++)
			for (int b = 0; b < 64; b++)
			{
				if (JUMP[i] & UINT64_C(1) << b)
				{
					s0 ^= s[0];
					s1 ^= s[1];
					s2 ^= s[2];
					s3 ^= s[3];
				}
				(*this)();
			}

		s[0] = s0;
		s[1] = s1;
		s[2] = s2;
		s[3] = s3;
	}
};

/** default generator which should be fine for all our purposes */
typedef xoshiro256 rng_t;

#endif
