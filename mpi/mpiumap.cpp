#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <unordered_map>
#include <list>
#include "mpiumap.hpp"

namespace mpi = boost::mpi;

template <typename T, typename T2>
mpiumap<T, T2>(mpi::communicator& w, const std::unordered_map<T, T2>& m){
  world = w;
  msize = m.size();
  keys.reserve(msize);
  values.reserve(msize);
  for (auto it = m.begin(); it != m.end(); it++) {
    keys.push_back(it->first);
    values.push_back(it->second);
  }
}

template <typename T, typename T2>
void mpiumap<T,T2>::mmbroadcast(){
  broadcast(world, msize, 0);
  if (world.rank() != 0) {
    keys.resize(msize);
    values.resize(msize);
  }
  broadcast(world, keys.data(), (int)keys.size(), 0);
  broadcast(world, values.data(), (int)values.size(), 0);
}

template <typename T, typename T2>
void mpiumap<T,T2>::getMap(std::unordered_map<T, T2>* m){
  for (size_t i = 0; i < msize; i++)
    (*m)[keys.at(i)] = values.at(i);
}

template <typename T, typename T2>
void mpiumap<T,T2>::showsize() {
  std::cout << world.rank() << "'s mapsize:" << msize << std::endl;
  std::cout << "values:" << std::endl;
  for (auto i: values)
    std::cout << i << " ";
  std::cout << std::endl;
}
