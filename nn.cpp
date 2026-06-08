#include "nn.h"

// Uniform weight distribution
float random_weight() {
    static std::random_device rd;
    static std::mt19937 gen{rd()};
    static std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    return dis(gen);
}

Neuron::Neuron(const int& nin) : m_Bias(std::make_shared<Value>(0.0f)) {
    for (int i{0}; i<nin; ++i)
        m_Weights.emplace_back(std::make_shared<Value>(random_weight()));
        // new Vector(random weight())
}

std::shared_ptr<Value> Neuron::operator()(const std::vector<std::shared_ptr<Value>>& x) {
    std::shared_ptr<Value> outAct{m_Bias};
    if(x.size()!=m_Weights.size()) {
        std::cout << "Dimensions do not match\n";
        return outAct;
    }
    // sum(w*x + b)
    for (int i{0}; i<x.size(); i++)
        outAct = add(outAct, mul(m_Weights[i], x[i]));
    
    return tanh(outAct);
}

std::vector<std::shared_ptr<Value>> Neuron::params() {
    std::vector<std::shared_ptr<Value>> parameters{m_Weights};
    parameters.emplace_back(m_Bias);
    return parameters;
}

Layer::Layer(const int& nin, const int& nout) {
    for (int i{0}; i<nout; ++i)
        m_Neurons.emplace_back(Neuron(nin));
}

std::vector<std::shared_ptr<Value>> Layer::operator()(const std::vector<std::shared_ptr<Value>>& x){
    std::vector<std::shared_ptr<Value>> out;
    for(auto& n : m_Neurons)
        out.emplace_back(n(x));
    
    return out;
}

std::vector<std::shared_ptr<Value>> Layer::params() {
    std::vector<std::shared_ptr<Value>> parameters;

    for(auto& n : m_Neurons) {
        // Vector is destroyed after every iteration. Little performance overhead but no significant effect on memory.
        std::vector<std::shared_ptr<Value>> n_Params{n.params()};
        parameters.insert(parameters.end(), n_Params.begin(), n_Params.end());
    }
    return parameters;
}

MLP::MLP(const int& nin, const std::vector<int>& nouts) {
    std::vector<int> l_Sizes{nouts};
    l_Sizes.insert(l_Sizes.begin(), nin);
    for (size_t i{0}; i<l_Sizes.size()-1; i++)
        m_Layers.emplace_back(Layer(l_Sizes[i], l_Sizes[i+1]));
}

std::vector<std::shared_ptr<Value>> MLP::operator()(const std::vector<std::shared_ptr<Value>>& x) {
    std::vector<std::shared_ptr<Value>> out{x};
    for (auto& layer: m_Layers)
        out = layer(out);
    
    return out;
}

std::vector<std::shared_ptr<Value>> MLP::params() {
    std::vector<std::shared_ptr<Value>> parameters;
    for(auto& l : m_Layers) {
        // Vector is destroyed after every iteration. Small performance overhead but no significant effect on memory.
        std::vector<std::shared_ptr<Value>> l_Params{l.params()};
        parameters.insert(parameters.end(), l_Params.begin(), l_Params.end());
    }
    return parameters;
}

SGD::SGD(std::vector<std::shared_ptr<Value>> params, const float& lr) : m_Parameters(params), alpha(lr) {}

void SGD::zero_grads() {
    for (auto& p : m_Parameters)
        p->m_Gradient = 0.0f;
}
void SGD::step() {
    for (auto& p : m_Parameters)
        p->m_Value -= alpha * p->m_Gradient;
}
