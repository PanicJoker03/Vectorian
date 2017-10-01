#include <iostream>
#include "vectorian.h"
#include <stdio.h>
#include <string>

#define _USE_MATH_DEFINES

using namespace vian;

void test_vector_signment() {
	vec4 a{ vec3d{ vec2i{}, 1.0 }, 1.0 };
	vec4d b{ 0.5f, vec2d{ 0.1f, 0.1f }, 2.0f };
	std::cout << a;
	std::cout << b;
	vec3 woopy{ b.xyz };
	std::cin.get();
}

void test_matrix_transformations() {
	mat4 localMatrix;

	mat4 translation = mat4::Translation(1, 0, 1);
	mat4 rotation = mat4::Rotation(vian::DegToRad(90), 0, 0);
	mat4 scale = mat4::Scale(2.0f, 1.0f, 1.3f);

	vec3 position(1, 0, 0);
	vec3d speed(1, 2, 3);

	localMatrix = rotation * translation;
	localMatrix = vian::Translate(localMatrix, (vec3)speed);
	std::cout << "t:" << std::endl << translation;
	std::cout << "r:" << std::endl << rotation;
	std::cout << "s:" << std::endl << scale;
	std::cout << "localMatrix: " << std::endl << localMatrix;
	std::cout << "postion:" << position;
	std::cout << "postion tranlated:" << localMatrix * position;
	std::cin.get();
}

void test_quaternion_rotations() {
	quat q(vian::DegToRad(90), vec3(0.0f, 1.0f, 0.0f));
	quat q2(vian::DegToRad(45), vec3(1.0f, 0.0f, 0.0f));
	quat q3 = q2 * q;
	quatf q4 = (quatf)q3;
	vec3 point(0.0f, 1.0f, 0.0f);

	std::cout << "q: " << q << std::endl;
	std::cout << "q2: " << q2 << std::endl;
	std::cout << "q3 = q2 * q: " << q3 << std::endl;
	std::cout << "point: " << point << std::endl;
	std::cout << "rotated point: " << q3 * point << std::endl;
	std::cin.get();
}

void main() {

	test_vector_signment();
	test_matrix_transformations();
	test_quaternion_rotations();

}
