#pragma once

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
#include <array>


namespace vian{

	

	template<typename T, typename int size>
	class Vector {
	public:
		T data[size];

		Vector() : data {} {}

		template<typename U>
		operator U() {
			Vector<U, size> result;
			for (int i = 0; i < size; i++) {
				result.data[i] = static_cast<U>(this->data[i]);
			}
			return result;
		}

		T& operator[](int index) {
			return data[index];
		}
	};
	template<typename T> using Vector4 = Vector<T, 4>;
	template<typename T> using Vector3 = Vector<T, 3>;
	template<typename T> using Vector2 = Vector<T, 2>;
	template<typename T> using Vector1 = Vector<T, 2>;

	template<typename T>
	class Vector<T, 4> {
	public:
		union {
			//Adding more anonymous struct magic...
			T data[4];
			struct { T x, y, z, w; };
			struct { T r, g, b, a; };
			struct { Vector2<T> xy; T z, w; };
			struct { T x; Vector2<T> yz; T w; };
			struct { T x, y; Vector2<T> zw; };
			struct { Vector2<T> xy, zw; };
			struct { Vector3<T> xyz; T w; };
			struct { T x; Vector3<T> yzw; };
			Vector<T, 3> rgb;
		};
		Vector() : data{} { }
		Vector(T x, T y, T z, T w) : data{ x, y ,z ,w } { }
		Vector(const Vector3<T>& xyz, T w) : xyz{ xyz }, w{ w } { }
		Vector(T x, const Vector3<T>& yzw) : x{ x }, yzw{ yzw } { }
		Vector(const Vector2<T>& xy, T z, T w) : xy{ xy }, z{ z }, w{ w } { } 
		Vector(T x, const Vector2<T>& yz, T w) : x{ x }, yz{ yz }, w{ w } { }
		Vector(T x, T y, const Vector2<T>& zw) : x{ x }, y{ y }, zw{ zw } { }
		Vector(const Vector2<T>& xy, const Vector2<T>& zw) { }

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
			struct { T x; Vector2<T> yz; };
			struct { Vector2<T> xy; T z; };
		};
		Vector() : data{} { }
		Vector(T x, T y, T z) : data{ x, y, z } { }
		Vector(const Vector2<T>& xy, T z) : xy{ xy }, z{ z } { }
		Vector(T x, const Vector2<T>& yz) : x{ x }, yz{ yz } { }
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

		Vector() : data{} { }
		Vector(T x, T y) : x{ x }, y{ y } { }

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

		Vector() : data{} { }
		Vector(T x ) : x{ x } { }
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
	using vec4 = Vector4<float>;
	using vec3 = Vector3<float>;
	using vec2 = Vector2<float>;

	template<typename int size>
	using vecni = Vector<int, size>;
	using vec4i = Vector4<int>;
	using vec3i = Vector3<int>;
	using vec2i = Vector2<int>;

	template<typename int size>
	using vecnd = Vector<double, size>;
	using vec4d = Vector4<double>;
	using vec3d = Vector3<double>;
	using vec2d = Vector2<double>;
	
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
	/*
	//Scalar * Vector
	template<typename T, typename U, typename int size>
	Vector<T, size> operator *(const U& rhs, const Vector<T, size>& lhs) {
		return lhs * rhs;
	}*/

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
	Vector3<T> Cross(const Vector3<T>& lhs, const Vector3<U>& rhs) {
		Vector3<T> result;

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
		// It may help for multiplication constraints at compile time...
		static constexpr int matRows = rows;
		static constexpr int matCols = cols;
		
		Matrix() {
			// check at compile time with 'if constexpr'?
			if (matRows == matCols) {
				LoadIdentity();
			}
		}

		Matrix(const std::array<std::array<T, cols>, rows>& args) {
				for (int i = 0; i < rows; i++) {
					for (int j = 0; j < cols; j++) {
						this->data[i][j] = args[i][j];
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

		void LoadIdentity() {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if (i != j)
						data[i][j] = 0;
					else
						data[i][j] = 1;
				}
			}
		}

		/* NOTE: Should I put these in here?. Maybe a static class for this functions? */
		/* Or even in as free functions, they could be wrapped in a namespace...*/
	private:
		using Matrix4x4 = Matrix<T, 4, 4>;
	public:
		static Matrix<T, rows, cols> Identity();

		//Returns a matrix translated by a vector
		static Matrix4x4 Translation(const Vector3<T>& v);
		static Matrix4x4 Translation(const T& x, const T& y, const T& z);

		/*
		Returns a matrix rotated by a the angles (radians) given in the x, y and z axis.
		Order of rotation is Y -> Z -> X;
		*/
		static Matrix4x4 Rotation(const Vector3<T>& v);
		static Matrix4x4 Rotation(const T& x, const T& y, const T& z);

		//Returns a matrix scaled by a vector
		static Matrix4x4 Scale(const Vector3<T>& v);
		static Matrix4x4 Scale(const T& x, const T& y, const T& z);
	};
	template<typename T> using Matrix4x4 = Matrix<T, 4, 4>;
	template<int rows, int cols>
	using mat = Matrix<float, rows, cols>;
	template<int rows_cols>
	using matn = Matrix<float, rows_cols, rows_cols>;
	using mat4 = Matrix<float, 4, 4>;
	using mat3 = Matrix<float, 3, 3>;
	using mat2 = Matrix<float, 2, 2>;

	template<int rows, int cols>
	using matd = Matrix<double, rows, cols>;
	template<int rows_cols>
	using matnd = Matrix<double, rows_cols, rows_cols>;
	using mat4d = Matrix<double, 4, 4>;
	using mat3d = Matrix<double, 3, 3>;
	using mat2d = Matrix<double, 2, 2>;



	template<typename T, int rows, int cols>
	static Matrix<T, rows, cols> Identity() {
		Matrix<T, rows, cols> result;
		if (rows == cols) {
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					if (i != j)
						result.data[i][j] = 0;
					else
						result.data[i][j] = 1;
				}
			}
		}
		return result;
	}

	//Returns a matrix translated by a vector
	template<typename T, int rows, int cols>
	Matrix4x4<T> Matrix<T, rows, cols>::Translation(const Vector3<T>& v) {
		return Matrix<T, rows, cols>::Translation(v.x, v.y, v.z);
	}
	template<typename T, int rows, int cols>
	Matrix4x4<T> Matrix<T, rows, cols>::Translation(const T& x, const T& y, const T& z) {
		Matrix4x4 result;
		result.data[0][3] = x;
		result.data[1][3] = y;
		result.data[2][3] = z;
		result.data[3][3] = 1;
		return result;
	}

	/*
	Returns a matrix rotated by the angles (radians) given in the x, y and z axis.
	Order of rotation is Y -> Z -> X;
	*/
	template<typename T, int rows, int cols>
	Matrix4x4<T> Matrix<T, rows, cols>::Rotation(const Vector3<T>& v) {
		return Matrix<T, rows, cols>::Rotation(v.x, v.y, v.z);
	}
	template<typename T, int rows, int cols>
	Matrix4x4<T> Matrix<T, rows, cols>::Rotation(const T& x, const T& y, const T& z) {
		Matrix4x4 result;
		Matrix4x4 Rx;
		Matrix4x4 Ry;
		Matrix4x4 Rz;

		Rx.data[1][1] = cos(x);
		Rx.data[1][2] = -sin(x);
		Rx.data[2][1] = sin(x);
		Rx.data[2][2] = cos(x);

		Ry.data[0][0] = cos(y);
		Ry.data[0][2] = sin(y);
		Ry.data[2][0] = -sin(y);
		Ry.data[2][2] = cos(y);

		Rz.data[0][0] = cos(z);
		Rz.data[0][1] = -sin(z);
		Rz.data[1][0] = sin(z);
		Rz.data[1][1] = cos(z);

		result = Rx * Rz * Ry;

		return result;
	}



	//Returns a matrix scaled by a vector
	template<typename T, int rows, int cols>
	Matrix4x4<T> Matrix<T, rows, cols>::Scale(const Vector3<T>& v) {
		return Matrix<T, rows, cols>::Scale(v.x, v.y, v.z);
	}
	template<typename T, int rows, int cols>
	Matrix4x4<T> Matrix<T, rows, cols>::Scale(const T& x, const T& y, const T& z) {
		Matrix4x4 result;
		result.data[0][0] = x;
		result.data[1][1] = y;
		result.data[2][2] = z;
		return result;
	}

	template<typename T>
	Matrix4x4<T> Translate(const Matrix4x4<T>& m, const Vector3<T>& v){
		Matrix4x4<T> tmp = Matrix4x4<T>::Translation(v);
		return tmp * m;
	}

	template<typename T>
	Matrix4x4<T> Rotate(const Matrix4x4<T>& m, const Vector3<T>& v) {
		Matrix4x4<T> tmp = Matrix4x4<T>::Rotation(v);
		return tmp * m;
	}

	template<typename T>
	Matrix4x4<T> Scale(const Matrix4x4<T>& m, const Vector3<T>& v) {
		Matrix4x4<T> tmp = Matrix4x4<T>::Scale(v);
		return tmp * m;
	}

	// Transpose of a matrix
	template<typename T, int rows, int cols>
	Matrix<T, cols, rows> Transpose(const Matrix<T, rows, cols>& m) {
		Matrix<T, cols, rows> result;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result.data[j][i] = m.data[i][j];
			}
		}

		return result;
	}

	// Matrix addition
	template<typename T, typename U, int rows, int cols>
	Matrix<T, rows, cols> operator+(const Matrix<T, rows, cols>& lhs, const Matrix<U, rows, cols>& rhs) {
		Matrix<T, rows, cols> result;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result.data[i][j] = lhs.data[i][j] + static_cast<T>(rhs.data[i][j]);
			}
		}
		return result;
	}

	// Matrix substraction
	template<typename T, typename U, int rows, int cols>
	Matrix<T, rows, cols> operator-(const Matrix<T, rows, cols>& lhs, const Matrix<U, rows, cols>& rhs) {
		Matrix<T, rows, cols> result;
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result.data[i][j] = lhs.data[i][j] - static_cast<T>(rhs.data[i][j]);
			}
		}
		return result;
	}

	
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

	// Matrix 4x4 multiplicated by vector 3
	template<typename T, typename U>
	Vector3<T> operator*(const Matrix4x4<T>& lhs, const Vector3<U>& rhs) {
		Vector3<T> result;
		Matrix<T, 4, 1> vm;
		vm.data[0][0] = rhs.data[0];
		vm.data[1][0] = rhs.data[1];
		vm.data[2][0] = rhs.data[2];
		vm.data[3][0] = 1;
		Matrix<T, 4, 1> r = lhs * vm;
		result.data[0] = r.data[0][0];
		result.data[1] = r.data[1][0];
		result.data[2] = r.data[2][0];
		return result;
	}

	// A matrix<m, n> multiplied by a matrix<n, p>
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

	/*
		Matrix TODO list:
		- Matrix transfomation
			- Rotate around an axis of rotation: https://learnopengl.com/#!Getting-started/Transformations
	*/

	template<typename T>
	class Quaternion {
	public:
		union {
			T data[4];
			struct { T w, x, y, z; };
			struct { T w; Vector3<T> xyz; };
		};
		Quaternion() : data{} { }
		Quaternion(T w, T x, T y, T z) {
			double a = static_cast<double>(w) / 2;
			this->w = static_cast<T>(cos(a));
			this->x = x*sin(a);
			this->y = y*sin(a);
			this->z = z*sin(a);
		}
		Quaternion(T w, Vector3<T> xyz){
			double a = static_cast<double>(w) / 2;
			this->w = static_cast<T>(cos(a));
			this->xyz = xyz * sin(a);
		}
		template<typename U>
		operator Quaternion<U>() {
			return Quaternion<U>(static_cast<U>(w), static_cast<Vector3<U>>(xyz));
		}
	};

	using quatf = Quaternion<float>;
	using quat = Quaternion<double>;

	template<typename T, typename U>
	Quaternion<T> operator*(const Quaternion<T>& lhs, const Quaternion<U>& rhs) {
		Quaternion<T> result;
		result.w = (static_cast<T>(rhs.w) * lhs.w) - vian::Dot(rhs.xyz, lhs.xyz);
		result.xyz =  lhs.xyz * static_cast<T>(rhs.w) + rhs.xyz * lhs.w + vian::Cross(lhs.xyz, rhs.xyz);
		return result;
	}

	template<typename T, typename U>
	Vector3<T> operator*(const Quaternion<T>& lhs, const Vector3<U>& rhs) {
		Vector3<T> result;
		result = vian::Cross(lhs.xyz, rhs);
		result = rhs + result * (2 * lhs.w) + (vian::Cross(lhs.xyz, result) * 2);
		return result;
	}

	/*
		Quaterniont TODO list:
		- slerp
		- Quaternion * Rotation Matrix
		- Quaterion -> Rotation Matrix
		- Euler Angles -> Quaternion
		- Quaternion -> Euler Angles
	*/


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



	template<typename T, int size>
	std::string to_string(const Vector<T, size>& v) {
		std::string result = "(";
		for (int i = 0; i < size; i++) {
			result += std::to_string(v.data[i]);
			if (i < size - 1) {
				result += ", ";
			}
		}
		result += ") ";
		return result;
	}

	template<typename T, int rows, int cols>
	std::string to_string(const Matrix<T, rows, cols>& m) {
		std::string result = "";
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result += std::to_string(m.data[i][j]) + " ";
			}
			result += "\n";
		}
		return result;
	}

	template<typename T>
	std::string to_string(const Quaternion<T>& q) {
		std::string result = "(";
		result += std::to_string(q.w) + ", ";
		result += std::to_string(q.x) + ", ";
		result += std::to_string(q.y) + ", ";
		result += std::to_string(q.z) + ", ";
		result += ")";
		return result;
	}

	template<typename T, int size>
	std::ostream& operator <<(std::ostream& o, const Vector<T, size>& v) {
		o << vian::to_string(v);
		return o;
	}

	template<typename T, int rows, int cols>
	std::ostream& operator <<(std::ostream& o, const Matrix<T, rows, cols>& m) {
		o << vian::to_string(m);
		return o;
	}

	template<typename T>
	std::ostream& operator <<(std::ostream& o, const Quaternion<T>& q) {
		o << vian::to_string(q);
		return o;
	}

	inline double DegToRad(double angle) {
		return M_DEG_TO_RAD * angle;
	}

	inline double RadToDeg(double angle) {
		return M_RAD_TO_DEG * angle;
	}
}
