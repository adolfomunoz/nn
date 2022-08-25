#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <tuple>
#include <random>
#include <chrono>
#include "../nn.h"

template<std::size_t N>
std::array<float,N> random_array() {
    static std::mt19937 gen{std::random_device()()}; //Random seed
    std::normal_distribution<float> d{0,1};
    std::array<float,N> sol;
    for (std::size_t i = 0; i<N; ++i) sol[i] = d(gen);
    return sol;
    
}

template<typename F>
void measure_time(const F& f, double range = 0.1, std::enable_if_t<std::is_same_v<std::decay_t<decltype(f())>,void>>* sfinae = nullptr) {
    unsigned long n = 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    f();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = t2-t1;
    while (duration.count() < range) {
        ++n;
        f();
        t2 = std::chrono::high_resolution_clock::now();
        duration = t2-t1;
    }
    
    double seconds = duration.count()/double(n);
    std::cout<<std::setw(6)<<std::fixed<<std::setprecision(0)<<(seconds*1.e6)<<"us\t";
}

template<typename F>
auto measure_time(const F& f, double range = 0.1, std::enable_if_t<!std::is_same_v<std::decay_t<decltype(f())>,void>>* sfinae = nullptr) {
    unsigned long n = 1;
    auto t1 = std::chrono::high_resolution_clock::now();
    auto sol = f();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = t2-t1;
    while (duration.count() < range) {
        ++n;
        sol = f();
        t2 = std::chrono::high_resolution_clock::now();
        duration = t2-t1;
    }
    
    double seconds = duration.count()/double(n);
    std::cout<<std::setw(6)<<std::fixed<<std::setprecision(0)<<(seconds*1.e6)<<"us\t";
    return sol;
}

template<typename NN>
void test_nn(const NN& nn, const std::vector<std::size_t>& numbers) {
    for (auto n : numbers) 
        measure_time([&] () { nn.nearest_neighbors(random_array<NN::dimensions>(),n); });
    
}

template<std::size_t DIMENSIONS, std::size_t SIZE>
std::vector<std::tuple<std::array<float,DIMENSIONS>,std::array<float,SIZE-DIMENSIONS>>> generate(std::size_t n) {
    std::vector<std::tuple<std::array<float,DIMENSIONS>,std::array<float,SIZE-DIMENSIONS>>> sol(n);
    std::mt19937 gen{std::random_device()()}; //Random seed
    std::normal_distribution<float> d{0,1};
    for (std::size_t i = 0;i<n;++i) {
        std::get<0>(sol[i]) = random_array<DIMENSIONS>();
        std::get<1>(sol[i]) = random_array<SIZE-DIMENSIONS>();
    }
    return sol;
}

template<std::size_t DIMENSIONS, std::size_t SIZE>
void test() {
    std::vector<std::size_t> numbers_for_search{1,4,16,64,256,1024,4096,16384};
    std::vector<std::size_t> numbers_for_storage{64,256,1024,4096,16384,65536,262144,1048576};

    std::cout<<"Internal  \tBalance   "; 
    for (std::size_t n : numbers_for_search) std::cout<<"\t"<<std::setw(6)<<n<<"  ";
    std::cout<<std::endl;
     
    for (std::size_t n : numbers_for_storage) {
        std::cout<<std::setw(10)<<n<<"\t";
        auto kd = measure_time([n] () { return nn::kdtree(generate<1,2>(n)); });
        test_nn(kd,numbers_for_search);
        std::cout<<std::endl;
    }
    std::cout<<"External  \tBalance   "; 
    for (std::size_t n : numbers_for_search) std::cout<<"\t"<<std::setw(6)<<n<<"  ";
    std::cout<<std::endl;
    for (std::size_t n : numbers_for_storage) {
        std::cout<<std::setw(10)<<n<<"\t";
        auto kd = measure_time([n] () { return nn::kdtree_external(generate<1,2>(n)); });
        test_nn(kd,numbers_for_search);
        std::cout<<std::endl;
    }  
}

template<std::size_t DIMENSIONS, std::size_t SIZE>
void size_test_row(std::size_t number_for_storage, std::size_t number_for_search) {
    std::cout<<std::setw(10)<<SIZE;
    auto kdi = measure_time([number_for_storage] () { return nn::kdtree(generate<DIMENSIONS,DIMENSIONS+1>(number_for_storage)); });
    auto kde = measure_time([number_for_storage] () { return nn::kdtree_external(generate<DIMENSIONS,DIMENSIONS+1>(number_for_storage)); });
    measure_time([&] () { kdi.nearest_neighbors(random_array<DIMENSIONS>(),number_for_search); });
    measure_time([&] () { kde.nearest_neighbors(random_array<DIMENSIONS>(),number_for_search); });
    std::cout<<std::endl;
}

template<std::size_t DIMENSIONS>
void size_test(std::size_t number_for_storage, std::size_t number_for_search) {
    std::cout<<"Size test for "<<DIMENSIONS<<" dimensions, "<<number_for_storage<<" total data, search for "<<number_for_search<<" data points"<<std::endl;
    
    std::cout<<"Size     \tInt creation\tExt creation\tInt search  \tExt search  "<<std::endl;
    size_test_row<DIMENSIONS,DIMENSIONS+1>(number_for_storage,number_for_search);
    
    
}

int main() {
    std::cout<<std::endl<<"1D small"<<std::endl;
    test<1,2>();    
    std::cout<<std::endl<<"1D big"<<std::endl;
    test<1,40>();
    std::cout<<std::endl<<"2D small"<<std::endl;
    test<2,3>();    
    std::cout<<std::endl<<"2D big"<<std::endl;
    test<2,40>();
    std::cout<<std::endl<<"3D small"<<std::endl;
    test<3,4>();    
    std::cout<<std::endl<<"3D big"<<std::endl;
    test<3,40>();

    
    return 0;
};