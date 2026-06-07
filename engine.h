#pragma once

#include <iostream>
#include <cmath> // tanh and exponentials
#include <vector> // Vectors
#include <functional> // Function wrapper
#include <set> // For Topological sort
#include <memory>
#include <algorithm>

class Value : public std::enable_shared_from_this<Value> {
public:
    float m_Value, m_Gradient;
    // Don't store copies of child nodes
    std::vector<std::shared_ptr<Value>> m_Prev;
    std::function<void()> m_Backward;

    Value(const float& value);

    std::vector<std::shared_ptr<Value>>& getChildren();

    void backward();
};

std::shared_ptr<Value> add(std::shared_ptr<Value> p1, std::shared_ptr<Value> p2);
std::shared_ptr<Value> mul(std::shared_ptr<Value> p1, std::shared_ptr<Value> p2);
std::shared_ptr<Value> tanh(std::shared_ptr<Value> p);
std::shared_ptr<Value> relu(std::shared_ptr<Value> p);
std::shared_ptr<Value> sub(std::shared_ptr<Value> p1, std::shared_ptr<Value> p2);
std::shared_ptr<Value> power(std::shared_ptr<Value> p, float exponent);
