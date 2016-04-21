/******************************************************************************
 * Copyright (c) 2013 Johannes Bergmann, Felix Weninger, Bjoern Schuller
 * Institute for Human-Machine Communication
 * Technische Universitaet Muenchen (TUM)
 * D-80290 Munich, Germany
 *
 * This file is part of CURRENNT.
 *
 * CURRENNT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CURRENNT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CURRENNT.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifndef LAYERS_DNNSOFTMAXLAYER_HPP
#define LAYERS_DNNSOFTMAXLAYER_HPP

#include "FeedForwardLayer.hpp"
#include <cudnn.h>

namespace layers {

    /******************************************************************************************//**
     * Represents a softmax layer in the neural network
     *
     * @param TDevice  The computation device (Cpu or Gpu)
     * @param TFfActFn The activation function to use before the softmax activation function is
     *                 applied (usually activation_functions::Identity
     *********************************************************************************************/
    template <typename TDevice, typename TFfActFn>
    class dnnSoftmaxLayer : public FeedForwardLayer<TDevice, TFfActFn>
    {
        typedef typename TDevice::real_vector real_vector;

    private:
        real_vector m_patTmp;

        cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc, deltaInTensorDesc, deltaOutTensorDesc;
        cudnnHandle_t cudnnHandle;
        real_t m_one, m_zero;
    public:
        /**
         * Constructs the Layer
         *
         * @param layerChild     The layer child of the JSON configuration for this layer
         * @param weightsSection The weights section of the JSON configuration
         * @param precedingLayer The layer preceding this one
         */
        dnnSoftmaxLayer(
            const helpers::JsonValue &layerChild,
            const helpers::JsonValue &weightsSection,
            Layer<TDevice>           &precedingLayer
            );

        /**
         * Destructs the Layer
         */
        virtual ~dnnSoftmaxLayer();

        /**
         * @see Layer::type()
         */
        virtual const std::string& type() const;

        /**
         * @see Layer::computeForwardPass()
         */
        virtual void computeForwardPass();

         /**
         * @see Layer::computeBackwardPass()
         */
        virtual void computeBackwardPass();
    };

} // namespace layers


#endif // LAYERS_DNNSOFTMAXLAYER_HPP
