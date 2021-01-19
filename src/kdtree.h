#pragma once

#include <vector>
#include <array>
#include <algorithm>

namespace nn {
    
/**
 * T - Data type contained in the KD Tree
 * N - Number of dimensions of the KD Tree (1 == binary tree)
 * A - Axis function to the position
            A axis_position; T t;
            axis_position(t,d) returns the numerical position for the d dimension. If possible it is deduced automatically for random access vectors and the like.
**/
template<typename T, std::size_t N, typename A>
class KDTree {
    using real = decltype(std::declval<A>()(std::declval<T>(),std::size_t(0)));
    
    A axis_position;
    class Node {
    public:
        std::size_t axis; //Axis. We might want to reduce its size in compile time when N<256...
        real position;    //Position for that axis
    };
    //elements and nodes are sorted equally so the element pointed by the ith node in node is the ith element in elements 
    std::vector<Node> nodes;
    public:
    std::vector<T> elements;
    //Given a node / element in position $i$, its left child is in position 2i+1 and its right child 2i+2

    std::array<real,N>& assign(std::array<real,N>& a, const T& t) const {
        for (std::size_t i = 0; i<N; ++i) a[i]=axis_position(t,i);
        return a;
    }
    
    std::array<real,N>& if_less_assign(std::array<real,N>& a, const T& t) const {
        for (std::size_t i = 0; i<N; ++i) if (a[i]>axis_position(t,i)) a[i]=axis_position(t,i);
        return a;        
    }
    
    std::array<real,N>& if_greater_assign(std::array<real,N>& a, const T& t) const {
        for (std::size_t i = 0; i<N; ++i) if (a[i]<axis_position(t,i)) a[i]=axis_position(t,i);
        return a;        
    }
    
    void build_tree(std::size_t left, std::size_t right) {
        if ((right-left) > 1) {
            //We build the bounding box each subdivision because even if it is slow it leaves a better kdtree balance
            std::array<real,N> bbmin, bbmax;
            assign(bbmin,elements[left]); assign(bbmax,elements[left]);
            for (std::size_t i = (left+1); i < right; ++i) {
                if_less_assign(bbmin,elements[i]);
                if_greater_assign(bbmax,elements[i]);
            }
            std::size_t median = (right+left)/2;
            //We find the larger axis
            std::size_t axis = 0; real max_bound = bbmax[0]-bbmin[0];
            for (std::size_t i = 1;i<N;++i) if ((bbmax[i]-bbmin[i])<max_bound) {
                axis = i; max_bound = bbmax[i]-bbmin[i];
            }
            //Partial ordering over that axis (median contains the median, to the left are smaller, to the right are greater)
            std::nth_element(elements.begin()+left,elements.begin()+median,elements.begin()+right,
                [&] (const T& a, const T& b) { return axis_position(a,axis)<axis_position(b,axis); });
            //The median stays in the median, so if in one dimension the vector is ordered (but not the case)
            real pos = axis_position(elements[median],axis);
            //We setup the node as well
            nodes[median].axis = axis;
            nodes[median].position = pos;
            //Recursive calls for the subtrees.
            build_tree(left,median);
            build_tree(median+1,right); 
        }
    }
    
    void build_tree() {
        nodes.resize(elements.size());
        build_tree(0,elements.size());
    }
public:
    KDTree(std::vector<T>&& elements, const A& axis_position) : elements(std::move(elements)), axis_position(axis_position) { build_tree(); }
    template<typename C> //Constructing from a general collection if possible
    KDTree(const C& c, const A& axis_position, typename std::enable_if<std::is_same<T,typename C::value_type>::value>::type* sfinae = nullptr) : elements(c), axis_position(axis_position) { build_tree(); }
};

namespace {
    class RandomAccess {
    public:
        template<typename T>
        operator()(const T& t, std::size_t i) const { return t[i]; }
    };
};

template<std::size_t N,typename C>
auto kdtree(const C& c) { return KDTree<std::decay_t<typename C::value_type>,N,RandomAccess>(c,RandomAccess()); } 

};