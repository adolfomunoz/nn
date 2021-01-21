#include <iostream>
#include <array>
#include <vector>
#include "../nn.h"

template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T,N>& a) {
    os<<"{"<<a[0];
    for (std::size_t i = 1;i<N;++i) os<<", "<<a[i];
    os<<"}";
    return os;
}
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& a) {
    os<<"["<<a[0];
    for (std::size_t i = 1;i<a.size();++i) os<<", "<<a[i];
    os<<"]";
    return os;
}
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T*>& a) {
    os<<"[^"<<*a[0];
    for (std::size_t i = 1;i<a.size();++i) os<<", ^"<<*a[i];
    os<<"]";
    return os;
}

int main() {
    {
        std::vector<std::array<float,1>> a = {{9},{3},{2},{7},{4},{5},{6},{1},{8}};
        auto kd = nn::kdtree(a);
        for (float f = 0.1;f<=10.9;f+=1.0) 
        std::cout<<"Nearest to "<<f<<" -> "<<kd.nearest_neighbors({f})<<std::endl;
    }
    
    {
        std::vector<std::array<float,2>> a;
        for (float i = 1.0f; i<9.001f; i+=1.0f) for (float j = 1.0f; j<9.001f; j+=1.0f) a.push_back({i,j});
        auto kd = nn::kdtree(a);
        for (float i = 0.5f; i<9.501f; i+=1.0f) for (float j = 0.5f; j<9.501f; j+=1.0f) 
            std::cout<<"Nearest to {"<<i<<", "<<j<<"} -> "<<kd.nearest_neighbors({i,j},4)<<std::endl;
        for (float i = 0.5f; i<9.501f; i+=1.0f) for (float j = 0.5f; j<9.501f; j+=1.0f) 
            std::cout<<"Closer to 1.0f {"<<i<<", "<<j<<"} -> "<<kd.nearest_neighbors({i,j},15,1.0f)<<std::endl;
            
    }
    return 0;
};