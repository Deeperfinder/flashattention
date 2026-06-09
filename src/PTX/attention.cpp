#include <torch/extension.h>
#include <torch/types.h>

#define STRINGFY(str) # str
#define TORCH_BIDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));

torch::Tensor attention_sm120(const torch::Tensor& Q, const torch::Tensor& K,
                              const torch::Tensor& V);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    TORCH_BIDING_COMMON_EXTENSION(attention_sm120)
}