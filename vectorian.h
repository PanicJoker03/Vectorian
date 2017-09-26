#pragma once
//Use template variables instead...
#ifndef _USE_MATH_DEFINES
#define M_PI				3.14159265358979323846
#define M_PI_2				1.57079632679489661923
#define M_PI_4				0.785398163397448309616
#define M_1_PI				0.318309886183790671538
#define M_2_PI				0.636619772367581343076
#define M_2_SQRTPI			1.12837916709551257390
#endif

#define M_RAD_TO_DEG			57.295779513
#define M_DEG_TO_RAD			0.01745329252

#include <cmath>
#include <string>
#include <cstdarg>
#include <Windows.h>



namespace vian{

	

	template<typename T, typename int size>
	class Vector {
	public:
		T data[size];

		Vector() : data{} {
			M_PI;
			//ZeroMemory(data, sizeof(data));
		}

		template<typename U>
		operator U() {
			Vector<U, size> result;
			for (int i = 0; i < size; i++) {
				result.data[i] = static_cast<U>(this->data[i]);
			}
			return result;
		}
		/* no pude implementar esto :(
		template<typename  U, typename int size>
		Vector<T, size>& operator=(const U& rhs) {
			for (int i = 0; i < size; i++)
				this->data[i] = static_cast<T>(rhs);
			return *this;
		}
		*/
		T& operator[](int index) {
			return data[index];
		}
	};

	template<typename T>
	class Vector<T, 4> {
	public:
		union {
			//Adding more anonymous struct magic...
			T data[4];
			struct { T x, y, z, w; };
			struct { T r, g, b, a; };
			struct { T x; Vector<T, 3> yzw; };
			struct { T x; Vector<T, 2> yz; T w; };
			struct { Vector<T, 2> xy, zw; };
			struct { Vector<T, 3> xyz; T w; };
			//Vector<T, 3> xyz;
			//Vector<T, 2> xy;
			Vector<T, 3> rgb;
		};
		//Prefer initializer list idiom...
		Vector() : data{} {
			//ZeroMemory(data, sizeof(data));
		}
		Vector(T x, T y, T z, T w) : data{x, y ,z ,w} {
		}
		template<typename U>
		Vector(const Vector<U, 3>& xyz, T w) : xyz{ xyz }, w{w} {
			//this->x = static_cast<T>(xyz.x);
			//this->y = static_cast<T>(xyz.y);
			//this->z = static_cast<T>(xyz.z);
			//this->w = w;
		}
		template<typename U>
		Vector(T x, const Vector<U, 3>& yzw) : x{ x }, yzw{yzw} {
			//this->x = x;
			//this->y = static_cast<T>(yzw.y);
			//this->z = static_cast<T>(yzw.z);
			//this->w = static_cast<T>(yzw.w);
		}
		template<typename U>
		Vector(const Vector<U, 2>& xy, T z, T w) : xy{ xy }, z{ z }, w{w} {
			//this->x = static_cast<T>(xy.x);
			//this->y = static_cast<T>(xy.y);
			//this->z = z;
			//this->w = w;
		}
		template<typename U>
		Vector(T x, const Vector<U, 2>& yz, T w) : x{ x }, yz{ yz }, w{w} {
			//this->x = x;
			//this->y = static_cast<T>(yz.y);
			//this->z = static_cast<T>(yz.z);
			//this->w = w;
		}
		template<typename U>
		Vector(T x, T y, const Vector<U, 2>& zw) : x{ x }, y{ y }, zw{zw} {
			//this->x = x;
			//this->y = y;
			//this->z = static_cast<T>(zw.z);
			//this->w = static_cast<T>(zw.w);
		}
		template<typename U, typename V>
		Vector(const Vector<U, 2>& xy, const Vector<V, 2>& zw) {
			//this->x = static_cast<T>(xy.x);
			//this->y = static_cast<T>(xy.y);
			//this->z = static_cast<T>(zw.z);
			//this->w = static_cast<T>(zw.w);
		}

		template<typename U>
		operator Vector<U, 4>() {
			return Vector<U, 4>(
				static_cast<U>(this->x),
				static_cast<U>(this->y),
				static_cast<U>(this->z),
				static_cast<U>(this->w));
		}

		T& operator[](int index) {
			return data[index];
		}
	};

	template<typename T>
	class Vector<T, 3> {
	public:
		union {
			T data[3];
			struct { T x, y, z; };
			struct { T r, g, b; };
			struct { T x; Vector<T, 2> yz; };
			Vector<T, 2> xy;
		};
		Vector() : data{} {
			//ZeroMemory(data, sizeof(data));
		}
		Vector(T x, T y, T z) : data{x, y, z} {
			//this->x = x;
			//this->y = y;
			//this->z = z;
		}
		template<typename U>
		Vector(const Vector<U, 2>& xy, T z) : xy{ xy }, z{z} {
			//this->x = static_cast<T>(xy.x);
			//this->y = static_cast<T>(xy.y);
			//this->z = z;
		}
		template<typename U>
		Vector(T x, const Vector<U, 2>& yz) : x{ x }, yz{yz} {
			//this->x = x;
			//this->y = static_cast<T>(yz.y);
			//this->z = static_cast<T>(yz.z);
		}
		template<typename U>
		operator Vector<U, 3>() {
			return Vector<U, 3>(
				static_cast<U>(this->x), 
				static_cast<U>(this->y), 
				static_cast<U>(this->z));
		}
		T& operator[](int index) {
			return data[index];
		}
	};

	template<typename T>
	class Vector<T, 2> {
	public:
		union {
			T data[2];
			struct { T x, y; };
		};

		Vector() : data{} {
			//ZeroMemory(data, sizeof(data));
		}
		Vector(T x, T y) : x{ x }, y{y} {
			//this->x = x;
			//this->y = y;
		}
		template<typename U>
		operator Vector<U, 2>() {
			return Vector<U, 2>(
				static_cast<U>(this->x), 
				static_cast<U>(this->y));
		}

		T& operator[](int index) {
			return data[index];
		}
	};

	template<typename T>
	class Vector<T, 1> {
	public:
		union {
			T data[1];
			struct { T x; };
		};

		Vector() : data{} {
			//ZeroMemory(data, sizeof(data));
		}
		//???
		Vector(T x, T y) : x{x} {
			//this->x = x;
		}
		template<typename U>
		operator Vector<U, 1>() {
			return Vector<U, 1>(static_cast<U>(this->x));
		}

		T& operator[](int index) {
			return data[index];
		}
	};

	template<typename int size>
	using vecn = Vector<float, size>;
	using vec4 = Vector<float, 4>;
	using vec3 = Vector<float, 3>;
	using vec2 = Vector<float, 2>;

	template<typename int size>
	using vecni = Vector<int, size>;
	using vec4i = Vector<int, 4>;
	using vec3i = Vector<int, 3>;
	using vec2i = Vector<int, 2>;

	template<typename int size>
	using vecnd = Vector<double, size>;
	using vec4d = Vector<double, 4>;
	using vec3d = Vector<double, 3>;
	using vec2d = Vector<double, 2>;

	
	// Vector + Vector
	template<typename T, typename U, typename int size>
	Vector<T, size> operator +(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		Vector<T, size> result;
		for (int i = 0; i < size; i++)
			result.data[i] = lhs.data[i] + static_cast<T>(rhs.data[i]);
		return result;
	}

	//Negated Vector
	template<typename T, typename int size>
	Vector<T, size> operator -(const Vector<T, size>& rhs) {
		Vector<T, size> result;
		for (int i = 0; i < size; i++)
			result.data[i] = -rhs.data[i];
		return result;
	}

	//Vector - Vector
	template<typename T, typename U, typename int size>
	Vector<T, size> operator -(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		Vector<T, size> result;
		result = lhs + (-rhs);
		return result;
	}

	//Vector * Scalar
	template<typename T, typename U, typename int size>
	Vector<T, size> operator *(const Vector<T, size>& lhs, const U& rhs) {
		Vector<T, size> result;
		T s = static_cast<T>(rhs);
		for (int i = 0; i < size; i++)
			result.data[i] = lhs.data[i] * s;
		return result;
	}

	//Dot product between two vectors
	template<typename T, typename U, typename int size>
	T Dot(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		T result = 0;
		for (int i = 0; i < size; i++)
			result += lhs.data[i] * static_cast<T>(rhs.data[i]);
		return result;
	}

	//Cross product between two vectors
	template<typename T, typename U>
	Vector<T,3> Cross(const Vector<T, 3>& lhs, const Vector<U, 3>& rhs) {
		Vector<T, 3> result;

		result.data[0] = (lhs.data[1] * static_cast<T>(rhs.data[2])) - (lhs.data[2] * static_cast<T>(rhs.data[1]));
		result.data[1] = (lhs.data[2] * static_cast<T>(rhs.data[0])) - (lhs.data[0] * static_cast<T>(rhs.data[2]));
		result.data[2] = (lhs.data[0] * static_cast<T>(rhs.data[1])) - (lhs.data[1] * static_cast<T>(rhs.data[0]));

		return result;
	}

	// Magnitude of a N size Vector
	template<typename T, typename int size>
	double Magnitude(const Vector<T, size>& lhs) {
		double result = 0;
		for (int i = 0; i < size; i++) {
			result += static_cast<double>(lhs.data[i] * lhs.data[i]);
		}
		result = sqrt(result);
		return result;
	}

	// Magnitude squared of a N size Vector
	template<typename T, typename int size>
	double MagnitudeSqr(const Vector<T, size>& lhs) {
		double result = 0;
		for (int i = 0; i < size; i++) {
			result += static_cast<double>(lhs.data[i] * lhs.data[i]);
		}
		return result;
	}

	// Normalize a Vector of N size
	template<typename T, typename int size>
	vecnd<size> Normalize(const Vector<T, size>& lhs) {
		Vector<double, size> result;
		double mag = vian::Magnitude(lhs);
		for (int i = 0; i < size; i++) {
			result.data[i] = static_cast<double>(lhs.data[i]) / mag;
		}
		return result;
	}

	// Get a vector that points from lhs to rhs
	template<typename T, typename U, typename int size>
	Vector<T, size> Direction(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		Vector<T, size> result;
		result = rhs - lhs;
		return result;
	}

	// Get the scalar distance between two vectors of the same size
	template<typename T, typename U, typename int size>
	double Distance(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		double result;
		result = vian::Magnitude(rhs - lhs);
		return result;
	}

	// Get the squared scalar distance between two vectors of the same size
	template<typename T, typename U, typename int size>
	double DistanceSqr(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		double result;
		result = vian::MagnitudeSqr(rhs - lhs);
		return result;
	}

	// Get the angle between two Vectors of the same size in radians
	template<typename T, typename U, typename int size>
	double Angle(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		double result;
		double dot = vian::Dot(lhs, rhs);
		result = dot / (vian::Magnitude(rhs) * vian::Magnitude(lhs));
		result = acos(result);
		return result;
	}

	template<typename T, typename U, typename int size>
	double AngleDeg(const Vector<T, size>& lhs, const Vector<U, size>& rhs) {
		double result;
		result = vian::Angle(lhs, rhs) * M_RAD_TO_DEG;
		return result;
	}

	template<typename T, int rows, int cols>
	class Matrix{
	public:
		T data[rows][cols] = {};

		
		Matrix() {
			if (rows == cols)
				LoadIdentity();
		}

		Matrix(std::initializer_list<std::initializer_list<T>> list) {
			int r = (int)list.size();
			int c = (int)(list.begin())->size();
			if (r == rows && c == cols) {
				for (int i = 0; i < rows; i++) {
					for (int j = 0; j < cols; j++) {
						data[i][j] = *((list.begin() + i)->begin() + j);
					}
				}
			}
		}
		
		void LoadIdentity() {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if( i != j)
						data[i][j] = 0;
					else
						data[i][j] = 1;
				}
			}
		}

		template<typename U>
		operator Matrix<U, rows, cols>() {
			Matrix<U, rows, cols> tmp;
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					tmp.data[i][j] = static_cast<U>(this->data[i][j]);
				}
			}
			return tmp;
		}
	};

	template<int rows, int cols>
	using matrix = Matrix<float, rows, cols>;
	template<int rows_cols>
	using matrixn = Matrix<float, rows_cols, rows_cols>;
	using matrix4 = Matrix<float, 4, 4>;
	using matrix3 = Matrix<float, 3, 3>;
	using matrix2 = Matrix<float, 2, 2>;

	template<int rows, int cols>
	using matrixd = Matrix<double, rows, cols>;
	template<int rows_cols>
	using matrixnd = Matrix<double, rows_cols, rows_cols>;
	using matrix4d = Matrix<double, 4, 4>;
	using matrix3d = Matrix<double, 3, 3>;
	using matrix2d = Matrix<double, 2, 2>;

	// Matrix multiplicated by a scalar value
	template<typename T, typename U, int rows, int cols>
	T operator*(const Matrix<T, rows, cols>& lhs, const U& rhs) {
		T result = 0;
		T s = static_cast<T>(rhs);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result += lhs.data[i][j] * s;
			}
		}
		return result;
	}

	template<typename T, typename U, int m, int n, int p>
	Matrix<T, m, p> operator*(const Matrix<T, m, n>& lhs, const Matrix<U, n, p>& rhs) {
		Matrix<T, m, p> result;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < p; j++) {
				T value = 0;
				for (int k = 0; k < n; k++) {
					value += lhs.data[i][k] * static_cast<T>(rhs.data[k][j]);
				}
				result.data[i][j] = value;
			}
		}
		return result;
	}


	namespace experimental {

		#define ALPHA_MASK		0xFF000000
		#define RED_MASK 		0x00FF0000
		#define GREEN_MASK		0x0000FF00
		#define BLUE_MASK		0x000000FF

		#define ALPHA_OFFSET	24
		#define RED_OFFSET		16
		#define GREEN_OFFSET	8

		using color32 = Vector<unsigned char, 4>;
		using color24 = Vector<unsigned char, 3>;

		unsigned int Color32ToHex(const color32& color) {
			unsigned int hex =
				(color.a << ALPHA_OFFSET) | (color.r << RED_OFFSET) | (color.g << GREEN_OFFSET) | color.b;
			return hex;
		}

		//color32 HexToColor32(std::string hex) {
			
		//}

		unsigned int Color24ToHex(const color24& color) {
			unsigned int hex =
				(0xFF000000) | (color.r << RED_OFFSET) | (color.g << GREEN_OFFSET) | color.b;
			return hex;
		}

		//color32 HexToColor24(std::string hex) {

		//}
	}

}