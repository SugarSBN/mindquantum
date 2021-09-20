from mindquantum import Circuit, Simulator, Measure
from mindquantum import X, Y, Z, H, RX, RY, RZ, PhaseShift, qft
import math
import numpy as np

def phase_estimation(U, state, n, epsilon):
    '''
    U表示需要进行相位估计的酉算子
    state表示该酉算子对应需要相位估计的本征矢
    n表示需要至少精确到小数点后n位（二进制下）
    epsilon表示至少以1-epsilon的概率成功相位估计
    '''
    t = int(n + math.ceil(math.log2(2 + 1 / (2 * epsilon))))    # 第一寄存器比特数，推导详见Quantum Computation and Quantum Information
    m = int(math.log2(len(state)))                              # 第二寄存器比特数
    
    c = Circuit()
    for k in range(t):
        c += H.on(k)                                            # 制成均匀叠加态
    for k in range(t):
        for j in range(2 ** k):
           c += U.on([i for i in range(t, t + m)], t - k - 1)   # 受控U门
    c += qft([i for i in range(t)]).hermitian                   # 逆Fourier变换

    for k in range(t):
        state = np.kron(state, [1, 0])                          # 初态应为|000000>|state>
    sim = Simulator('projectq', t + m)
    sim.set_qs(np.array(state))
    sim.apply_circuit(c)
    for k in range(t):
        sim.apply_measure(Measure().on(k))                      # 对第一寄存器测量
    qs = sim.get_qs()
    index = np.argmax(qs)
    bit = bin(index)[2:]
    bit = bit.rjust(n + 2, '0')
    bit = bit[::-1]
    bit = bit[0 : n]
    ans = int(bit, 2) / (2 ** len(bit))                         # 进制转换处理
    return ans

print(phase_estimation(RZ(math.pi / 3), [1, 0], 10, 0.3))       # ans = 11/12 = 0.91666667
print(phase_estimation(RZ(math.pi * 2 / 3), [1, 0], 10, 0.3))   # ans = 5/6 = 0.83333333
print(phase_estimation(RZ(math.pi / 6), [1, 0], 10, 0.3))       # ans = 23/24 = 0.95833333
print(phase_estimation(RZ(math.pi / 4), [1, 0], 10, 0.3))       # ans = 15/16 = 0.9375
print(phase_estimation(RZ(math.pi / 2), [1, 0], 10, 0.3))       # ans = 7/8 = 0.875
print(phase_estimation(X, [1 / math.sqrt(2), -1 / math.sqrt(2)], 10, 0.3)) # ans = 1/2 = 0.5
'''
output:
0.916015625
0.8330078125
0.9580078125
0.9375
0.875
0.5
'''
