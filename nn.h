#pragma once

#include "engine.h"
#include <random> // Uniform random generator

// Uniform weight distribution
float random_weight();

class Neuron {
public:
    std::vector<std::shared_ptr<Value>> m_Weights;
    std::shared_ptr<Value> m_Bias;
    // std::vector<Value*> w;

    Neuron(const int& nin);

    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& x);

    std::vector<std::shared_ptr<Value>> params();
};

class Layer {
public:
    std::vector<Neuron> m_Neurons;

    Layer(const int& nin, const int& nout);

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x);

    std::vector<std::shared_ptr<Value>> params();
};

class MLP {
public:
    std::vector<Layer> m_Layers;

    MLP(const int& nin, const std::vector<int>& nouts);

    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x);

    std::vector<std::shared_ptr<Value>> params();
};

class SGD {
public:
    std::vector<std::shared_ptr<Value>> m_Parameters;
    float alpha;

    SGD(std::vector<std::shared_ptr<Value>> params, const float& lr);

    void zero_grads();

    void step();
};
