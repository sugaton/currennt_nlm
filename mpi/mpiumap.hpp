#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <unordered_map>
#include <list>

template <typename T, typename T2>
classmpiumap
{
private:
  size_t msize;
  mpi::communicator world;
  std::vector<T> keys;
  std::vector<T2> values;
public:
  mpiumap(){};
  mpiumap(mpi::communicator& w, const std::unordered_map<T, T2>& m);
  ~mpiumap(){};

  void mmbroadcast();
  void getMap(std::unordered_map<T, T2>* m);
  void showsize();
};
