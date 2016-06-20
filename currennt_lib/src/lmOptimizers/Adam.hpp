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

#ifndef LM_OPTIMIZERS_ADAM_HPP
#define LM_OPTIMIZERS_ADAM_HPP

#include "lmOptimizer.hpp"

#include <vector>
#include <map>


namespace optimizers {

    /******************************************************************************************//**
     * Optimizer that uses steepest descent
     *
     * @param TDevice The computation device (Cpu or Gpu)
     *********************************************************************************************/
    template <typename TDevice>
    class Adam : public lmOptimizer<TDevice>
        {
        typedef typename TDevice::real_vector real_vector;

    private:
        real_t m_learningRate;
        real_t m_learningRateFirst;
        size_t m_MLookupStart;
        std::vector<size_t> m_MStart;
        real_t m_beta1;
        real_t m_beta1t;
        real_t m_beta2;
        real_t m_beta2t;
        real_t m_eps;
        Cpu::real_vector m_beta1t_emb;
        Cpu::real_vector m_beta2t_emb;
        std::vector<real_vector> m_momentum_arr;
        std::vector<real_vector> m_2momentum_arr;
        std::vector<real_t*> m_moment;
        std::vector<real_t*> m_second_moment;

        int m_tlimit;
        int m_t_;


        std::vector<real_vector> m_weightDeltas;

    protected:
        virtual void _updateWeights(int device = 0);
        virtual void _updateWeightsMultiGpu();
        void _SumUpdates(std::map<int, int> &emb_posi);

    public:
        /**
         * Constructs the optimizer
         *
         * @param neuralNetwork   The neural network to operate on
         * @param trainingSet     The set of training sequences
         * @param validationSet   The set of validation sequences
         * @param testSet         The set of test sequences
         * @param maxEpochs       The maximum total number of epochs to train
         * @param maxEpochsNoBest The number of epochs in which no new lowest error could be
         *                        achieved before training is stopped
         * @param validateEvery   After how many epochs the validation error shall be calculated
         * @param testEvery       After how many epochs the test error shall be calculated
         * @param learningRate    The learning rate
         * @param momentum        The momentum
         */
        Adam(
            NeuralNetwork<TDevice> &neuralNetwork,
            data_sets::Corpus     &trainingSet,
            data_sets::Corpus     &validationSet,
            data_sets::Corpus     &testSet,
            int maxEpochs,
            int maxEpochsNoBest,
            int validateEvery,
            int testEvery,
            real_t learningRate,
            int  tmp_show = -1,
            real_t beta1 = 0.9,
            real_t beta2 = 0.99,
            real_t eps   = 1.0e-8
            );

        /**
         * Destructs the optimizer
         */
        virtual ~Adam();

        /**
         * @see Optimizer::exportState
         */
        virtual void exportState(const helpers::JsonDocument &jsonDoc) const;

        /**
         * @see Optimizer::importState
         */
        virtual void importState(const helpers::JsonDocument &jsonDoc);

        /**
         * Sets the learning rate for the first layer.
         */
        void setLearningRateFirst(real_t learningRateFirst);
    };

} // namespace optimizers


#endif // OPTIMIZERS_ADAM_HPP
