#pragma once
typedef unsigned int uint;


////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 make_float2(float s)
{
	return make_float2(s, s);
}
inline __host__ __device__ float2 make_float2(float3 a)
{
	return make_float2(a.x, a.y);
}
inline __host__ __device__ float2 make_float2(int2 a)
{
	return make_float2(float(a.x), float(a.y));
}
inline __host__ __device__ float2 make_float2(uint2 a)
{
	return make_float2(float(a.x), float(a.y));
}

inline __host__ __device__ int2 make_int2(int s)
{
	return make_int2(s, s);
}
inline __host__ __device__ int2 make_int2(int3 a)
{
	return make_int2(a.x, a.y);
}
inline __host__ __device__ int2 make_int2(uint2 a)
{
	return make_int2(int(a.x), int(a.y));
}
inline __host__ __device__ int2 make_int2(float2 a)
{
	return make_int2(int(a.x), int(a.y));
}

inline __host__ __device__ uint2 make_uint2(uint s)
{
	return make_uint2(s, s);
}
inline __host__ __device__ uint2 make_uint2(uint3 a)
{
	return make_uint2(a.x, a.y);
}
inline __host__ __device__ uint2 make_uint2(int2 a)
{
	return make_uint2(uint(a.x), uint(a.y));
}

inline __host__ __device__ float3 make_float3(float s)
{
	return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
	return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
	return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
	return make_float3(a.x, a.y, a.z);
}
inline __host__ __device__ float3 make_float3(int3 a)
{
	return make_float3(float(a.x), float(a.y), float(a.z));
}
inline __host__ __device__ float3 make_float3(uint3 a)
{
	return make_float3(float(a.x), float(a.y), float(a.z));
}

inline __host__ __device__ int3 make_int3(int s)
{
	return make_int3(s, s, s);
}
inline __host__ __device__ int3 make_int3(int2 a)
{
	return make_int3(a.x, a.y, 0);
}
inline __host__ __device__ int3 make_int3(int2 a, int s)
{
	return make_int3(a.x, a.y, s);
}
inline __host__ __device__ int3 make_int3(uint3 a)
{
	return make_int3(int(a.x), int(a.y), int(a.z));
}
inline __host__ __device__ int3 make_int3(float3 a)
{
	return make_int3(int(a.x), int(a.y), int(a.z));
}

inline __host__ __device__ uint3 make_uint3(uint s)
{
	return make_uint3(s, s, s);
}
inline __host__ __device__ uint3 make_uint3(uint2 a)
{
	return make_uint3(a.x, a.y, 0);
}
inline __host__ __device__ uint3 make_uint3(uint2 a, uint s)
{
	return make_uint3(a.x, a.y, s);
}
inline __host__ __device__ uint3 make_uint3(uint4 a)
{
	return make_uint3(a.x, a.y, a.z);
}
inline __host__ __device__ uint3 make_uint3(int3 a)
{
	return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

inline __host__ __device__ float4 make_float4(float s)
{
	return make_float4(s, s, s, s);
}
inline __host__ __device__ float4 make_float4(float3 a)
{
	return make_float4(a.x, a.y, a.z, 0.0f);
}
inline __host__ __device__ float4 make_float4(float3 a, float w)
{
	return make_float4(a.x, a.y, a.z, w);
}
inline __host__ __device__ float4 make_float4(int4 a)
{
	return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
inline __host__ __device__ float4 make_float4(uint4 a)
{
	return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

inline __host__ __device__ int4 make_int4(int s)
{
	return make_int4(s, s, s, s);
}
inline __host__ __device__ int4 make_int4(int3 a)
{
	return make_int4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ int4 make_int4(int3 a, int w)
{
	return make_int4(a.x, a.y, a.z, w);
}
inline __host__ __device__ int4 make_int4(uint4 a)
{
	return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}
inline __host__ __device__ int4 make_int4(float4 a)
{
	return make_int4(int(a.x), int(a.y), int(a.z), int(a.w));
}


inline __host__ __device__ uint4 make_uint4(uint s)
{
	return make_uint4(s, s, s, s);
}
inline __host__ __device__ uint4 make_uint4(uint3 a)
{
	return make_uint4(a.x, a.y, a.z, 0);
}
inline __host__ __device__ uint4 make_uint4(uint3 a, uint w)
{
	return make_uint4(a.x, a.y, a.z, w);
}
inline __host__ __device__ uint4 make_uint4(int4 a)
{
	return make_uint4(uint(a.x), uint(a.y), uint(a.z), uint(a.w));
}

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 &a)
{
	return make_float2(-a.x, -a.y);
}
inline __host__ __device__ int2 operator-(int2 &a)
{
	return make_int2(-a.x, -a.y);
}
inline __host__ __device__ float3 operator-(float3 &a)
{
	return make_float3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ int3 operator-(int3 &a)
{
	return make_int3(-a.x, -a.y, -a.z);
}
inline __host__ __device__ float4 operator-(float4 &a)
{
	return make_float4(-a.x, -a.y, -a.z, -a.w);
}
inline __host__ __device__ int4 operator-(int4 &a)
{
	return make_int4(-a.x, -a.y, -a.z, -a.w);
}

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ float2 operator+(float2 a, float b)
{
	return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ float2 operator+(float b, float2 a)
{
	return make_float2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(float2 &a, float b)
{
	a.x += b;
	a.y += b;
}

inline __host__ __device__ int2 operator+(int2 a, int2 b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(int2 &a, int2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ int2 operator+(int2 a, int b)
{
	return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ int2 operator+(int b, int2 a)
{
	return make_int2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(int2 &a, int b)
{
	a.x += b;
	a.y += b;
}

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b)
{
	return make_uint2(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(uint2 &a, uint2 b)
{
	a.x += b.x;
	a.y += b.y;
}
inline __host__ __device__ uint2 operator+(uint2 a, uint b)
{
	return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ uint2 operator+(uint b, uint2 a)
{
	return make_uint2(a.x + b, a.y + b);
}
inline __host__ __device__ void operator+=(uint2 &a, uint b)
{
	a.x += b;
	a.y += b;
}


inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __host__ __device__ int3 operator+(int3 a, int3 b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(int3 &a, int3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ int3 operator+(int3 a, int b)
{
	return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(int3 &a, int b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b)
{
	return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(uint3 &a, uint3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}
inline __host__ __device__ uint3 operator+(uint3 a, uint b)
{
	return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(uint3 &a, uint b)
{
	a.x += b;
	a.y += b;
	a.z += b;
}

inline __host__ __device__ int3 operator+(int b, int3 a)
{
	return make_int3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ uint3 operator+(uint b, uint3 a)
{
	return make_uint3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ float3 operator+(float b, float3 a)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ float4 operator+(float4 a, float b)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ float4 operator+(float b, float4 a)
{
	return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(float4 &a, float b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

inline __host__ __device__ int4 operator+(int4 a, int4 b)
{
	return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(int4 &a, int4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ int4 operator+(int4 a, int b)
{
	return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ int4 operator+(int b, int4 a)
{
	return make_int4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(int4 &a, int b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

inline __host__ __device__ uint4 operator+(uint4 a, uint4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
inline __host__ __device__ void operator+=(uint4 &a, uint4 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
}
inline __host__ __device__ uint4 operator+(uint4 a, uint b)
{
	return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ uint4 operator+(uint b, uint4 a)
{
	return make_uint4(a.x + b, a.y + b, a.z + b, a.w + b);
}
inline __host__ __device__ void operator+=(uint4 &a, uint b)
{
	a.x += b;
	a.y += b;
	a.z += b;
	a.w += b;
}

////////////////////////////////////////////////////////////////////////////////
// subtract
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator-(float2 a, float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, float2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ float2 operator-(float2 a, float b)
{
	return make_float2(a.x - b, a.y - b);
}
inline __host__ __device__ float2 operator-(float b, float2 a)
{
	return make_float2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(float2 &a, float b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ int2 operator-(int2 a, int2 b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(int2 &a, int2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ int2 operator-(int2 a, int b)
{
	return make_int2(a.x - b, a.y - b);
}
inline __host__ __device__ int2 operator-(int b, int2 a)
{
	return make_int2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(int2 &a, int b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ uint2 operator-(uint2 a, uint2 b)
{
	return make_uint2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint2 b)
{
	a.x -= b.x;
	a.y -= b.y;
}
inline __host__ __device__ uint2 operator-(uint2 a, uint b)
{
	return make_uint2(a.x - b, a.y - b);
}
inline __host__ __device__ uint2 operator-(uint b, uint2 a)
{
	return make_uint2(b - a.x, b - a.y);
}
inline __host__ __device__ void operator-=(uint2 &a, uint b)
{
	a.x -= b;
	a.y -= b;
}

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ float3 operator-(float b, float3 a)
{
	return make_float3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(float3 &a, float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __host__ __device__ int3 operator-(int3 a, int3 b)
{
	return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(int3 &a, int3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ int3 operator-(int3 a, int b)
{
	return make_int3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ int3 operator-(int b, int3 a)
{
	return make_int3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(int3 &a, int b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __host__ __device__ uint3 operator-(uint3 a, uint3 b)
{
	return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}
inline __host__ __device__ uint3 operator-(uint3 a, uint b)
{
	return make_uint3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ uint3 operator-(uint b, uint3 a)
{
	return make_uint3(b - a.x, b - a.y, b - a.z);
}
inline __host__ __device__ void operator-=(uint3 &a, uint b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, float4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ float4 operator-(float4 a, float b)
{
	return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ void operator-=(float4 &a, float b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

inline __host__ __device__ int4 operator-(int4 a, int4 b)
{
	return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(int4 &a, int4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ int4 operator-(int4 a, int b)
{
	return make_int4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ int4 operator-(int b, int4 a)
{
	return make_int4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(int4 &a, int b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

inline __host__ __device__ uint4 operator-(uint4 a, uint4 b)
{
	return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint4 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
}
inline __host__ __device__ uint4 operator-(uint4 a, uint b)
{
	return make_uint4(a.x - b, a.y - b, a.z - b, a.w - b);
}
inline __host__ __device__ uint4 operator-(uint b, uint4 a)
{
	return make_uint4(b - a.x, b - a.y, b - a.z, b - a.w);
}
inline __host__ __device__ void operator-=(uint4 &a, uint b)
{
	a.x -= b;
	a.y -= b;
	a.z -= b;
	a.w -= b;
}

////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator*(float2 a, float2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(float2 &a, float2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ float2 operator*(float2 a, float b)
{
	return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a)
{
	return make_float2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(float2 &a, float b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ int2 operator*(int2 a, int2 b)
{
	return make_int2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(int2 &a, int2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ int2 operator*(int2 a, int b)
{
	return make_int2(a.x * b, a.y * b);
}
inline __host__ __device__ int2 operator*(int b, int2 a)
{
	return make_int2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(int2 &a, int b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ uint2 operator*(uint2 a, uint2 b)
{
	return make_uint2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint2 b)
{
	a.x *= b.x;
	a.y *= b.y;
}
inline __host__ __device__ uint2 operator*(uint2 a, uint b)
{
	return make_uint2(a.x * b, a.y * b);
}
inline __host__ __device__ uint2 operator*(uint b, uint2 a)
{
	return make_uint2(b * a.x, b * a.y);
}
inline __host__ __device__ void operator*=(uint2 &a, uint b)
{
	a.x *= b;
	a.y *= b;
}

inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ float3 operator*(float3 a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
	return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(float3 &a, float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ int3 operator*(int3 a, int3 b)
{
	return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(int3 &a, int3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ int3 operator*(int3 a, int b)
{
	return make_int3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ int3 operator*(int b, int3 a)
{
	return make_int3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(int3 &a, int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ uint3 operator*(uint3 a, uint3 b)
{
	return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint3 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
}
inline __host__ __device__ uint3 operator*(uint3 a, uint b)
{
	return make_uint3(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ uint3 operator*(uint b, uint3 a)
{
	return make_uint3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(uint3 &a, uint b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
	return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(float4 &a, float4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ float4 operator*(float4 a, float b)
{
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a)
{
	return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(float4 &a, float b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

inline __host__ __device__ int4 operator*(int4 a, int4 b)
{
	return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(int4 &a, int4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ int4 operator*(int4 a, int b)
{
	return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ int4 operator*(int b, int4 a)
{
	return make_int4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(int4 &a, int b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

inline __host__ __device__ uint4 operator*(uint4 a, uint4 b)
{
	return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint4 b)
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
}
inline __host__ __device__ uint4 operator*(uint4 a, uint b)
{
	return make_uint4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ uint4 operator*(uint b, uint4 a)
{
	return make_uint4(b * a.x, b * a.y, b * a.z, b * a.w);
}
inline __host__ __device__ void operator*=(uint4 &a, uint b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
	a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// divide
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 operator/(float2 a, float2 b)
{
	return make_float2(a.x / b.x, a.y / b.y);
}
inline __host__ __device__ void operator/=(float2 &a, float2 b)
{
	a.x /= b.x;
	a.y /= b.y;
}
inline __host__ __device__ float2 operator/(float2 a, float b)
{
	return make_float2(a.x / b, a.y / b);
}
inline __host__ __device__ void operator/=(float2 &a, float b)
{
	a.x /= b;
	a.y /= b;
}
inline __host__ __device__ float2 operator/(float b, float2 a)
{
	return make_float2(b / a.x, b / a.y);
}

inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(float3 &a, float3 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(float3 &a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}
inline __host__ __device__ float3 operator/(float b, float3 a)
{
	return make_float3(b / a.x, b / a.y, b / a.z);
}

inline __host__ __device__ float4 operator/(float4 a, float4 b)
{
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
inline __host__ __device__ void operator/=(float4 &a, float4 b)
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
}
inline __host__ __device__ float4 operator/(float4 a, float b)
{
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
inline __host__ __device__ void operator/=(float4 &a, float b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
	a.w /= b;
}
inline __host__ __device__ float4 operator/(float b, float4 a)
{
	return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}

////////////////////////////////////////////////////////////////////////////////
// dot product
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float dot(float4 a, float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ int dot(int2 a, int2 b)
{
	return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ int dot(int3 a, int3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ int dot(int4 a, int4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ uint dot(uint2 a, uint2 b)
{
	return a.x * b.x + a.y * b.y;
}
inline __host__ __device__ uint dot(uint3 a, uint3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ uint dot(uint4 a, uint4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
inline __host__ __device__ float3 fmin(float3 a, float3 b)
{
	return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float length(float2 v)
{
	return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float3 v)
{
	return sqrtf(dot(v, v));
}
inline __host__ __device__ float length(float4 v)
{
	return sqrtf(dot(v, v));
}

////////////////////////////////////////////////////////////////////////////////
// normalize
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ float2 normalize(float2 v)
{
	float invLen = sqrtf(dot(v, v));
	return v * invLen;
}
inline __host__ __device__ float3 normalize(float3 v)
{
	float invLen = sqrtf(dot(v, v));
	return v * invLen;
}
inline __host__ __device__ float4 normalize(float4 v)
{
	float invLen = sqrtf(dot(v, v));
	return v * invLen;
}