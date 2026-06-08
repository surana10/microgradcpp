#include "engine.h"

Value::Value(const float& value) : m_Value(value), m_Gradient(0.0f) {
    m_Backward = [](){};
}

std::vector<std::shared_ptr<Value>>& Value::getChildren() {
    return m_Prev;
}

void Value::backward() {
    // Only used by last node:
    std::vector<std::shared_ptr<Value>> m_Topo;
    std::set<std::shared_ptr<Value>> m_Visited;

    std::function<void(std::shared_ptr<Value>)> buildTopo = [&](std::shared_ptr<Value> p) {
        if (m_Visited.find(p)==m_Visited.end()){
            m_Visited.insert(p);
            for(auto& child : p->getChildren())
                buildTopo(child);
            m_Topo.emplace_back(p);
        }
    };
    // I hate using this because I am scared. Don't see another option here though.
    buildTopo(shared_from_this());
    
    m_Gradient = 1.0f;

    std::reverse(m_Topo.begin(), m_Topo.end());
    for (auto& param : m_Topo) {
        param->m_Backward();
    }
}

std::shared_ptr<Value> add(std::shared_ptr<Value> p1, std::shared_ptr<Value> p2) {
    float temp{p1->m_Value + p2->m_Value};
    std::shared_ptr<Value> out{std::make_shared<Value>(temp)};
    out->m_Prev.emplace_back(p1);
    out->m_Prev.emplace_back(p2);

    out->m_Backward = [p1, p2, out] (){
        p1->m_Gradient += 1.0f * out->m_Gradient;
        p2->m_Gradient += 1.0f * out->m_Gradient;
    };

    return out;
}

std::shared_ptr<Value> mul(std::shared_ptr<Value> p1, std::shared_ptr<Value> p2) {
    float temp{p1->m_Value * p2->m_Value};
    std::shared_ptr<Value> out{std::make_shared<Value>(temp)};
    out->m_Prev.emplace_back(p1);
    out->m_Prev.emplace_back(p2);

    out->m_Backward = [p1, p2, out]() {
        p1->m_Gradient += p2->m_Value * out->m_Gradient;
        p2->m_Gradient += p1->m_Value * out->m_Gradient;
    };

    return out;
}

std::shared_ptr<Value> tanh(std::shared_ptr<Value> p) {
    float temp{(std::exp(2*p->m_Value) - 1)/(std::exp(2*p->m_Value) + 1)};
    std::shared_ptr<Value> out{std::make_shared<Value>(temp)};
    out->m_Prev.emplace_back(p);

    out->m_Backward = [p, out, temp]() {
        p->m_Gradient += (1.0f - (temp * temp)) * out->m_Gradient;
    };

    return out;
}

std::shared_ptr<Value> relu(std::shared_ptr<Value> p) {
    float temp{std::max(0.0f, p->m_Value)};
    std::shared_ptr<Value> out{std::make_shared<Value>(temp)};
    out->m_Prev.emplace_back(p);

    out->m_Backward = [p, out]() {
        if (0.0f < p->m_Value)
            p->m_Gradient += 1.0f * out->m_Gradient;
    };

    return out;
}

std::shared_ptr<Value> sub(std::shared_ptr<Value> p1, std::shared_ptr<Value> p2) {
    float temp = p1->m_Value - p2->m_Value;
    std::shared_ptr<Value> out = std::make_shared<Value>(temp);
    
    out->m_Prev = {p1, p2};
    out->m_Backward = [p1, p2, out]() {
        p1->m_Gradient += 1.0f * out->m_Gradient;
        p2->m_Gradient += -1.0f * out->m_Gradient; // Negative 1 for the subtracted value
    };
    return out;
}


std::shared_ptr<Value> power(std::shared_ptr<Value> p, float exponent) {
    float temp = std::pow(p->m_Value, exponent);
    std::shared_ptr<Value> out = std::make_shared<Value>(temp);
    
    out->m_Prev = {p};
    out->m_Backward = [p, out, exponent]() {
        // Power rule: d/dx(x^n) = n * x^(n-1)
        float derivative = exponent * std::pow(p->m_Value, exponent - 1.0f);
        p->m_Gradient += derivative * out->m_Gradient;
    };
    return out;
}
