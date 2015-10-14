#ifndef EMBEDDING_HPP
#define EMBEDDING_HPP

#include <memory>
#include <>

#include "../Type.hpp"


/*
 * embedding must be able to be stored in gpu and cpu
 * we store both gpu and cpu embedding into Embedding class
 * Embeddings are controled in Embedding_ class.
 */

namespace helpers{

    template <typename TDevice>
    class Embedding_{
        typedef typename TDevice::real_vector real_vector;
    public:
        Embedding_(Cpu::real_vector* vec);
        real_vector get_data();
        size_t get_size();
        template <typename TD> void replace(TD::real_vector& vec);
    private:
        real_vector m_data;
        size_t m_size;
    };

    class Embedding{
    public:
        Embedding(Cpu::real_vector* vec, std::string tname);

        /*
         * if this embedding stored in other device (i.e. we use Gpu but it's stored in cpu), it need to be copied to data, than call get_data(real_vector data)
         */
        template <typename TDevice> Embedding_<TDevice>::real_vector get_data();

        /*
         * if this embedding stored in other device (i.e. we use Gpu but it's stored in cpu), it need to be copied to data, than we should call this.
         */
        template <typename TDevice> Embedding_<TDevice>::real_vector get_data(Embedding_<TDevice>::real_vector *data);
        std::string type();
        template <typename TD> void replace(TD::real_vector& vec);
    private:
        std::string m_tname;
        std::unique_ptr<Embedding_<Gpu>> m_gemb;
        std::unique_ptr<Embedding_<Cpu>> m_cemb;
    };
} // namespace helper
#endif
