import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, Parameter
from definitions import *
def fun(x, y):
    gates = RX(x, 0), RY(y, 1), CNOT(1, [0])
    cir = Circuit(*gates, num_qubits=2)
    ket0 = Matrix_Product(cir, Phi_0(2)) 
    loss = Expectation(ket0, Tensor_Product(DENSITY_0, IDENTITY_2))
    return loss

x = Parameter(Tensor(2, ms.float32), name='x')
y = Parameter(Tensor(np.pi/5, ms.float32), name='y')

print(fun(x, y))
grads_fun = ms.grad(fun, (0))
grads = grads_fun(x, y)
print(grads)