#include <iostream>
#include <array>
#include <vector>
#include <tuple>
#include <random>
#include "../nn.h"

template<typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T,N>& a) {
    os<<"["<<a[0];
    for (std::size_t i = 1;i<N;++i) os<<", "<<a[i];
    os<<"]";
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
template<typename T1,typename T2>
std::ostream& operator<<(std::ostream& os, const std::tuple<T1,T2>& a) {
    os<<"("<<std::get<0>(a)<<", "<<std::get<1>(a)<<")";
    return os;
}

int main() {
    {
        bool ok = true;
        std::cout<<"1D test   \t"<<std::flush;
        std::vector<std::array<float,1>> a = {{9},{3},{2},{7},{4},{5},{6},{1},{8}};
        auto kd = nn::kdtree(a);
        auto kd2 = nn::kdtree_external(a);
        for (float f = 0.1;f<=10.9;f+=1.0) {
            float v = std::max(1.0f,std::min(9.0f,float(int(f))));
            auto neighbors = kd.nearest_neighbors({f});
            ok = ok && (v == (*neighbors.front())[0]) && (neighbors.size() == 1);
            neighbors = kd2.nearest_neighbors({f});
            ok = ok && (v == (*neighbors.front())[0]) && (neighbors.size() == 1);
        }
        std::cout<<(ok?"[ OK ]":"[FAIL]")<<std::endl;
    }
    
    {   
        bool ok = true;
        std::cout<<"2D test   \t"<<std::flush;
        std::vector<std::tuple<std::array<float,2>,std::string>> a;
        std::mt19937 gen{1}; //Fixed seed
        std::normal_distribution<float> d{0,1};
        int searchable = 10;
        for (int i = 0;i<1000;++i) a.push_back({{d(gen),d(gen)},"NO"});
        for (int i = 0;i<searchable;++i) a.push_back({{d(gen)+10,d(gen)+10},"YES"});
        auto kd = nn::kdtree(a);
        auto kd2 = nn::kdtree_external(a);
        for (const auto& t : kd.nearest_neighbors({10,10},searchable))
            ok = ok && (std::get<1>(*t) == "YES");
        for (const auto& t : kd2.nearest_neighbors({10,10},searchable))
            ok = ok && (std::get<1>(*t) == "YES");
        std::cout<<(ok?"[ OK ]":"[FAIL]")<<std::endl;
    }
    
    {
        bool ok = true;
        std::cout<<"Radius test\t"<<std::flush;
        std::vector<std::array<float,2>> a;
        for (float i = 1.0f; i<9.001f; i+=1.0f) for (float j = 1.0f; j<9.001f; j+=1.0f) a.push_back({i,j});
        auto kd = nn::kdtree(a);
        auto kd2 = nn::kdtree_external(a);
        std::vector<std::array<float,2>> v = {{0.55f,0.55f},{9.45f,0.55f},{9.45f,9.45f},{0.55f,9.45f}};
        for (const auto& q : v) {
            std::array<float,2> s = {std::round(q[0]),std::round(q[1])};
            auto nearest = kd.nearest_neighbors(q,100,1.0f);
            ok = ok && (nearest.size() == 1) && (*nearest[0] == s);
            nearest = kd2.nearest_neighbors(q,100,1.0f); 
            ok = ok && (nearest.size() == 1) && (*nearest[0] == s);
        }
        ok = ok && (kd.nearest_neighbors({-1,-1},100,1.0f).size() == 0);
        ok = ok && (kd2.nearest_neighbors({-1,-1},100,1.0f).size() == 0);
        std::cout<<(ok?"[ OK ]":"[FAIL]")<<std::endl;
    }
    
    return 0;
};