#include "Embedding.hpp"


 // TODO write Embedding_::update(real_vector ) and Embedding<TDevice>::update(real_vector)
namespace helpers{


    template<typename TDevice, typename ParentDevice>
    Embedding_<TDevice, ParentDevice>::Embedding_(Cpu::real_vector *vec){
        // m_data = real_vector(vec->size());
        // m_data.resize(vec->size());
        // thrust::copy(vec->begin(), vec->end(), m_data.begin());
        m_data = *(vec);
        // m_size = vec->size;
    }

    template<typename TDevice, typename ParentDevice>
    typename Embedding_<TDevice, ParentDevice>::real_vector* Embedding_<TDevice, ParentDevice>::get_data(){
        return &m_data;
    }
    //
    // template<typename TDevice, typename ParentDevice>
    // size_t Embedding_<TDevice, ParentDevice>::get_size(){
    //     return m_size;
    // }

    template<typename TDevice, typename ParentDevice>
    void Embedding_<TDevice, ParentDevice>::replace(Embedding_<TDevice, ParentDevice>::p_real_vector *vec){
        m_data = *vec;
        // thrust::copy(vec->begin(), vec->end(), m_data.begin());
    }

    template <typename TDevice>
    Embedding<TDevice>::Embedding(Cpu::real_vector *vec, std::string tname){
        m_tname = tname;
        if (tname == typeid(Gpu).name()){
            m_gemb = std::unique_ptr<Embedding_<Gpu, TDevice>>(new Embedding_<Gpu, TDevice>(vec));
            m_cemb = nullptr;
        }
        else if (tname == typeid(Cpu).name()){
            m_cemb = std::unique_ptr<Embedding_<Cpu, TDevice>>(new Embedding_<Cpu, TDevice>(vec));
            m_gemb = nullptr;
        }
    }
    template <typename TDevice>
    Embedding<TDevice>::~Embedding() {
        // printf("the memory of embedding has released\n");
    }


    template<typename TDevice>
    typename Embedding<TDevice>::real_vector* Embedding<TDevice>::get_data(){
        if (typeid(TDevice).name() == typeid(Gpu).name()){
            // no need to copy
            return (Embedding<TDevice>::real_vector*)m_gemb->get_data();
        }
        else if (typeid(TDevice).name() == typeid(Cpu).name()){
            return (Embedding<TDevice>::real_vector*)m_cemb->get_data();
        }
    }

    template<typename TDevice>
    typename Embedding<TDevice>::real_vector* Embedding<TDevice>::get_data(Embedding<TDevice>::real_vector *data){
        if (m_tname == typeid(Gpu).name()){
            *data = *(m_gemb->get_data());
        }
        else if (m_tname == typeid(Cpu).name()){
            *data = *(m_cemb->get_data());
        }
        return data;
    }

    template<typename TDevice>
    std::string Embedding<TDevice>::type(){
        return m_tname;
    }

    // /*
    template<typename TDevice>
    void Embedding<TDevice>::replace(Embedding<TDevice>::real_vector* vec){
        if (m_tname == typeid(Gpu).name()){
            m_gemb->replace(vec);
        }
        else if (m_tname == typeid(Cpu).name()){
            m_cemb->replace(vec);
        }
    }

    // */
    template class Embedding_<Cpu, Cpu>;
    template class Embedding_<Cpu, Gpu>;
    template class Embedding_<Gpu, Gpu>;

    template class Embedding<Cpu>;
    template class Embedding<Gpu>;

} // namespace helpers
