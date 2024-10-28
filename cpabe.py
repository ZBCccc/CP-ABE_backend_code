#!/usr/bin/env python3

import hashlib
from fractions import Fraction
from itertools import combinations
import numpy as np
import sympy

import policy
from bn254 import big
from bn254 import curve
from bn254 import ecp, ecp2
from bn254 import fp12
from bn254 import pair


# 将字符串转换为模p的整数
def str_to_int_mod_p(s, p):
    # 将字符串编码为字节
    byte_representation = s.encode('utf-8')
    # 将字节转换为整数
    num = int.from_bytes(byte_representation, byteorder='big')
    # 映射到域 Z_p
    result = num % p
    return result

# initialization function
def setup() -> tuple:
    # 初始化曲线参数
    G1 = ecp.generator()
    G2 = ecp2.generator()
    
    # 生成主密钥
    alpha = big.rand(curve.r)
    
    # 生成公共参数
    p = curve.r
    s = [big.rand(p) for _ in range(4)]
    u = s[0] * G1
    h = s[1] * G1
    w = s[2] * G1
    v = s[3] * G1
    
    # 生成接收端id
    receiver_id = big.rand(10001)
    
    # 生成private user-key和public user-key
    sk_id = big.rand(curve.r)
    pk_id = sk_id * G1

    return (p, G1, G2, u, h, w, v), alpha, receiver_id, sk_id, pk_id

def pubKG(pm: tuple, mk: int, pk_id, attributes: list[int]):
    p, G1, G2, u, h, w, v = pm
    # 定义用户属性
    k = len(attributes)

    # 生成长度为k+1的随机数组
    random_array = [big.rand(curve.r) for _ in range(k + 1)]
    pk_1 = mk * pk_id
    pk_1 = pk_1.add(random_array[0] * w)
    pk_2 = random_array[0] * G2
    pk_3 = [random_array[i] * G2 for i in range(1, k + 1)]
    pk_4 = []
    for i in range(1, k + 1):
        pk = attributes[i - 1] * u  # F1 function
        pk = pk.add(h)
        pk = random_array[i] * pk
        tmp = random_array[0] * v
        tmp = -tmp
        pk = pk.add(tmp)
        pk_4.append(pk)
    return pk_1, pk_2, pk_3, pk_4

# 将字符串转换为fp12
def fp12_from_str(e: str) -> fp12.Fp12:
    # 将字符串e哈希
    hash_obj = hashlib.shake_256()
    hash_obj.update(e.encode('utf-8'))
    E = hash_obj.digest(384)
    FS = curve.EFS
    # 字节数组E的长度限定为12*FS，若长度小于，则用0填充，若大于，则截断
    if len(E) < 12 * FS:
        E = E.ljust(12 * FS, b'\0')
    elif len(E) > 12 * FS:
        E = E[:12 * FS]
    # 将字节数组E转化为fp12
    fp12_obj = fp12.Fp12()
    fp12_obj.fromBytes(E)
    return fp12_obj

# 将fp12转换为字节数组
def fp12_to_bytes(fp12_obj: fp12.Fp12) -> bytearray:
    return fp12_obj.toBytes()

# 加密函数
def Encrypt(pm: tuple, alpha: int, A: list, access_structure: list, message: str) -> tuple:
    p, G1, G2, u, h, w, v = pm
    v_array = [big.rand(p) for _ in range(len(access_structure[0]))]
    V_array = np.array(v_array)
    M_array = np.array(access_structure)
    V_M = np.dot(M_array, V_array)
    mul_array = [big.rand(p) for _ in range(len(access_structure))]
    
    c_0 = pair.e(G2, G1)
    c_0 = c_0.pow(alpha * v_array[0])
    processed_message = fp12_from_str(message)
    c_0 = c_0 * processed_message
    
    c_1 = v_array[0] * G2
    c_2 = []
    for i in range(len(access_structure)):
        tmp1 = V_M[i] * w
        tmp2 = mul_array[i] * v
        c2 = tmp1.add(tmp2)
        c_2.append(c2)
    c_3 = []
    for i in range(len(access_structure)):
        tmp1 = A[i] * u
        tmp1 = tmp1.add(h)
        tmp1 = -mul_array[i] * tmp1
        c_3.append(tmp1)
    c_4 = [mul_array[i] * G2 for i in range(len(access_structure))]
    return c_0, c_1, c_2, c_3, c_4

def find_intersection(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return list(set1.intersection(set2))

def solve_equation(A, B):
    A_pseudo_inverse = np.linalg.pinv(A)
    # 计算近似的 w
    w = B @ A_pseudo_inverse
    # 计算 w * A 与 B 的差距
    difference = np.allclose(w @ A, B, atol=1e-8)

    if difference:
        return w
    else:
        return None
    # A_T = A.T
    # B_T = B.T
    #
    # try:
    #     # 首先尝试使用 np.linalg.solve
    #     w_T = np.linalg.solve(A_T, B_T)
    #     return w_T.T  # 转置回原始形状
    # except np.linalg.LinAlgError:
    #     # 如果 solve 失败，则使用 lstsq
    #     try:
    #         w_T, residuals, rank, s = np.linalg.lstsq(A_T, B_T, rcond=None)
    #         error = np.max(np.abs(np.dot(w_T, A_T) - B_T))
    #         if error < 1e-8:
    #             return w_T.T
    #         else:
    #             return w_T.T
    #     except np.linalg.LinAlgError:
    #         return None


def find_integer_w(A, B):
    # 构造增广矩阵 [A.T | B.T]
    aug_matrix = np.hstack((A.T, B.T.reshape(-1, 1)))

    # 使用Fraction来保持精确计算，避免浮点数误差
    aug_matrix_frac = np.array([[Fraction(x) for x in row] for row in aug_matrix])

    # 获取矩阵的维度
    rows, cols = aug_matrix_frac.shape
    n = A.shape[0]  # w的维度

    # 高斯消元
    rank = 0
    for j in range(cols - 1):  # 只处理系数矩阵部分
        # 找主元
        pivot_row = None
        for i in range(rank, rows):
            if aug_matrix_frac[i][j] != 0:
                pivot_row = i
                break

        if pivot_row is not None:
            # 交换行
            if pivot_row != rank:
                aug_matrix_frac[rank], aug_matrix_frac[pivot_row] = \
                    aug_matrix_frac[pivot_row], aug_matrix_frac[rank].copy()

            # 归一化主元行
            pivot = aug_matrix_frac[rank][j]
            aug_matrix_frac[rank] = [elem / pivot for elem in aug_matrix_frac[rank]]

            # 消元
            for i in range(rows):
                if i != rank and aug_matrix_frac[i][j] != 0:
                    factor = aug_matrix_frac[i][j]
                    aug_matrix_frac[i] = [elem - factor * aug_matrix_frac[rank][k]
                                          for k, elem in enumerate(aug_matrix_frac[i])]
            rank += 1

    # 检查解的存在性和唯一性
    for i in range(rank, rows):
        if aug_matrix_frac[i][-1] != 0:
            return None  # 无解

    # 构造一个特解
    solution = np.zeros(n)
    free_vars = set(range(n))  # 自由变量的索引集合

    # 回代求解
    for i in range(rank - 1, -1, -1):
        # 找到当前行的主元列
        for j in range(cols - 1):
            if aug_matrix_frac[i][j] == 1:
                free_vars.discard(j)
                # 计算该变量的值
                value = float(aug_matrix_frac[i][-1])
                if not value.is_integer():
                    return None  # 如果不是整数解，返回None
                solution[j] = int(value)
                break

    # 设置自由变量为0或1，尝试找到整数解
    for free_var in free_vars:
        solution[free_var] = 0

    # 验证解是否正确
    if np.allclose(solution @ A, B):
        return solution.astype(int)

    # 如果设置自由变量为0得到的解不正确，尝试设置为1
    for free_var in free_vars:
        solution[free_var] = 1

    if np.allclose(solution @ A, B):
        return solution.astype(int)

    return None
# 转换函数
def Transform(pk_list, access_structure, A1, A2, CT):
    pk_1, pk_2, pk_3, pk_4 = pk_list
    c_0, c_1, c_2, c_3, c_4 = CT
    M = access_structure
    I = find_intersection(A1, A2)
    
    if len(I) == 0:
        return None
    else:
        # 构造M的子矩阵
        M_1 = np.array([M[i-1] for i in I])
        # 构造向量[1,0,0,...,0]，长度为len(I)
        B = np.array([1] + [0] * (len(access_structure[0])-1))
        # w = solve_equation(M_1, B)
        w = find_integer_w(M_1, B)

        if w is None:
            return None
        else:
            # 将w转化为int列表
            w = [int(x) for x in w]
            C_0 = []
            for i in range(len(I)):
                if w[i] == 0:
                    continue
                tmp1 = pair.e(pk_2, c_2[I[i]-1])
                tmp2 = pair.e(pk_3[i], c_3[I[i]-1])
                tmp3 = pair.e(c_4[I[i]-1], pk_4[i])
                tmp = tmp1 * tmp2 * tmp3
                tmp = tmp.pow(w[i])
                C_0.append(tmp)
            # 将C_0相乘
            result = C_0[0]
            for i in range(1, len(C_0)):
                result = result * C_0[i]
            C_1 = pair.e(c_1, pk_1)
            result = result * C_1.inverse()
            return result

def Decrypt(sk, CT1, CT2):
    tmp1 = big.invmodp(sk, curve.r)
    tmp2 = CT1.pow(tmp1)
    M = tmp2 * CT2
    return M

def main():
    # 初始化
    pm, mk, rid, sk, pk = setup()
    
    # 加密
    bool_exp = "(A AND B) AND (C OR D)"
    access_structure_msp, _  = policy.boolean_to_msp(bool_exp, False)
    sender_attributes = access_structure_msp.row_to_attrib
    access_structure = [[int(elem) for elem in row] for row in access_structure_msp.mat.tolist()]
    A1 = [i+1 for i in range(len(sender_attributes))]
    message = "hello"
    CT = Encrypt(pm, mk, A1, access_structure, message)
    
    # 边辅助解密
    user_attributes = [1, 2, 4]
    user_attributes_str = ['A', 'B', 'C']
    pk_list = pubKG(pm, mk, pk, user_attributes)
    transform_ciphertext = Transform(pk_list, access_structure, A1, user_attributes, CT)
    if transform_ciphertext is None:
        print("transform_ciphertext is None")
    else:
        # 解密
        decrypt_ciphertext = Decrypt(sk, transform_ciphertext, CT[0])
        print(f"before == after = {fp12_from_str(message) == decrypt_ciphertext}")

if __name__ == "__main__":
    main()
