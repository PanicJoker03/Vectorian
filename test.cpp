#include "vectorian.h"
#include <iostream>
void printVec(const vian::vec4& v) {
	std::cout << "Values are: " << v.x << ", " << v.y << ", " << v.z << ", " << v.w;
	std::cout << "\n";
}
void main() {
	using namespace vian;
	vec4 a{ vec3{vec2{}, 1.0}, 1.0 };
	vec4 b{ 0.5f, vec2{0.1f, 0.1f}, 2.0f };
	printVec(a);
	printVec(b);
	vec3 woopy{a.yz, b.z};
	std::cin.get();
}