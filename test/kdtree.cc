#include "../nn.h"
#include <iostream>

int main() {
   std::vector<std::array<float,1>> a = {{9},{3},{2},{7},{4},{5},{6},{1},{8}};
   auto kd = nn::kdtree<1>(a);
   for (const auto e : kd.elements) std::cout<<e[0]<<" ";
   std::cout<<std::endl;
   
   std::vector<std::array<float,2>> a2 = {{9,1},{3,1},{2,1},{7,2},{4,2},{5,2},{6,3},{1,3},{8,3}};
   auto kd2 = nn::kdtree<2>(a2);
   for (const auto e : kd2.elements) std::cout<<"{"<<e[0]<<", "<<e[1]<<"} ";
   std::cout<<std::endl;
   
   return 0;
};