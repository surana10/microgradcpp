#include "nn.h"

int main() {
    MLP model(3, std::vector<int>({4,4,1}));
    SGD optimizer(model.params(), 0.05f);

    std::vector<std::vector<float>> xF{{2.0f, 3.0f, -1.0f},
                                       {3.0f, -1.0f, 0.5f},
                                       {0.5f, 1.0f, 1.0f},                   
                                       {1.0f, 1.0f, -1.0f}};
    std::vector<std::vector<std::shared_ptr<Value>>> x;
    for(const auto& a : xF){
        std::vector<std::shared_ptr<Value>> temp;
        for(const auto&b : a)
            temp.emplace_back(std::make_shared<Value>(b));
        x.emplace_back(std::move(temp));
    }

    std::vector<float> y{{1.0f, -1.0f, -1.0f, 1.0f}}; 

    int epochs{20};

    for (int k{0}; k<epochs; k++){
        // FWD
        std::shared_ptr<Value> loss{std::make_shared<Value>(0.0f)};

        for(size_t i{0}; i<x.size(); i++){
            std::shared_ptr<Value> yTarget{std::make_shared<Value>(y[i])};
            std::shared_ptr<Value> yPred{model(x[i])[0]};

            //LOSS
            std::shared_ptr<Value> sq_error{power(sub(yPred, yTarget), 2.0f)};

            loss = add(loss, sq_error);
        }
         // BCKWD
        optimizer.zero_grads();
        loss->backward();
        optimizer.step();

        std::cout << "Epoch: " << k << " | Loss: " << loss->m_Value << "\n";

    }

    // FINAL
    std::cout << "Expected output : {";
    for(size_t i{0}; i<y.size(); i++)
        std::cout << " " << y[i] << " ";
    std::cout << "}\n"; 

    std::cout << "Model output : {";
    for(size_t i{0}; i<x.size(); i++){
        std::shared_ptr<Value> yPred{model(x[i])[0]};
        std::cout << " " << yPred->m_Value << " ";
    }
    std::cout << "}\n";
}