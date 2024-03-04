"""
Cost module implements a set of commonly used cost functions with analytical gradient
- This is a proposed framework for now and could possibly be expanded to follow Thesues framework
- What this module is:
    1. A library of commonly used functions whose analytical gradients had been worked out
    2. A library of commonly used composite methods such as Sum or Max to compose different known costs and
        gradients together
- What this module is NOT:
    1. Support for gradient propagation through chained functions, eg f(g(x1)) -> find df/dx1.
        User is supposed to propagate the functions themselves via matmul given gradient shapes.
    2. An extensive support for complex gradient propagation. Theres a reason why autograd for torch requires 
        grads to be stored on the Tensor as it requires a graph to be built to know which leaf nodes the gradients
        propagate to from a source operation.
- Base costs module:
    - contains core cost functions used to build or compose other cost functions together to create
        more complex cost functions with gradient
"""