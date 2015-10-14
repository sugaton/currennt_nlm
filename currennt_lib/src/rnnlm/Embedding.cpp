#include "Embedding.hpp"


 // TODO write Embedding_::update(real_vector ) and Embedding::update(real_vector)
namespace helpers{


    template <typename TDevice>
    Embedding_<TDevice>::Embedding_(Cpu::real_vector *vec){
        m_data = *(vec);
        m_size = vec->size;
    }

    template <typename TDevice>
    Embedding_<TDevice>::real_vector& Embedding_<TDevice>::get_data(){
        return m_data;
    }

    template <typename TDevice>
    size_t Embedding_<TDevice>::get_size(){
        return m_size;
    }

    template<typename TDevice, TD>
    void Embedding_<TDevice>::replace(TD::real_vector &vec){
        m_data = vec;
    }

    Embedding::Embedding(Cpu::real_vector *vec, std::string tname){
        m_tname = tname;
        if (tname == typeid(Gpu).name()){
            m_gemb = std::make_unique<Embedding_<Gpu>>(vec);
            m_cemb = nullptr;
        }
        else if (tname == typeid(Cpu).name()){
            m_cemb = std::make_unique<Embedding_<Cpu>>(vec);
            m_gemb = nullptr;
        }
    }


    template<typename TDevice>
    Embedding_<TDevice>::real_vector& Embedding::get_data(){
        if (m_tname == typeid(Gpu).name()){
            // no need to copy
            return m_gemb->get_data();
        }
        else if (m_tname == typeid(Cpu).name()){
            return m_cemb->get_data();
        }
    }

    template<typename TDevice>
    Embedding_<TDevice>::real_vector& Embedding::get_data(Embedding_<TDevice>::real_vector *data){
        if (m_tname == typeid(Gpu).name()){
            *data = m_gemb->get_data();
        }
        else if (m_tname == typeid(Cpu).name()){
            *data = m_cemb->get_data();
        }
        return *data;
    }

    std::string type(){
        return m_tname;
    }

    // /*
    template<typename TDevice>
    void Embedding::replace(TDevice::real_vector& vec){
        if (m_tname == typeid(Gpu).name()){
            m_gemb->replace<TDevice>(vec);
        }
        else if (m_tname == typeid(Cpu).name()){
            m_cemb->replace<TDevice>(vec);
        }
    }
    // */
    template class Layer<Cpu>;
    template class Layer<Gpu>;

} // namespace helpers
