#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <map>
#include <list>

template <typename T, typename T2>
class mpimap
{
private:
  size_t msize;
  mpi::communicator world;
  std::vector<T> keys;
  std::vector<T2> values;
public:
  mpimap(){};
  mpimap(mpi::communicator& w, const std::map<T, T2>& m);
  ~mpimap(){};

  void mmbroadcast();
  void getMap(std::map<T, T2>* m);
  void showsize();
};
