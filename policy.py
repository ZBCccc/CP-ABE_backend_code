import numpy as np
from typing import List, Tuple


class MSP:
    def __init__(self, mat: np.ndarray, row_to_attrib: List[str]):
        self.mat = mat
        self.row_to_attrib = row_to_attrib


def boolean_to_msp(bool_exp: str, convert_to_ones: bool) -> Tuple[MSP, None]:
    # 初始化向量
    vec = np.array([1])

    # 调用递归函数
    msp, _ = boolean_to_msp_iterative(bool_exp, vec, 1)

    if convert_to_ones:
        # 创建可逆矩阵
        inv_mat = np.eye(msp.mat.shape[1])
        inv_mat[0, :] = 1

        # 将MSP矩阵与可逆矩阵相乘
        msp.mat = np.dot(msp.mat, inv_mat)

    return msp, None


def boolean_to_msp_iterative(bool_exp: str, vec: np.ndarray, c: int) -> Tuple[MSP, int]:
    bool_exp = bool_exp.strip()
    num_brc = 0
    found = False

    for i, e in enumerate(bool_exp):
        if e == '(':
            num_brc += 1
            continue
        if e == ')':
            num_brc -= 1
            continue

        if num_brc == 0:
            if bool_exp[i:i + 3] == "AND":
                bool_exp1, bool_exp2 = bool_exp[:i], bool_exp[i + 3:]
                vec1, vec2 = make_and_vecs(vec, c)
                msp1, c1 = boolean_to_msp_iterative(bool_exp1, vec1, c + 1)
                msp2, c_out = boolean_to_msp_iterative(bool_exp2, vec2, c1)
                found = True
                break
            elif bool_exp[i:i + 2] == "OR":
                bool_exp1, bool_exp2 = bool_exp[:i], bool_exp[i + 2:]
                msp1, c1 = boolean_to_msp_iterative(bool_exp1, vec, c)
                msp2, c_out = boolean_to_msp_iterative(bool_exp2, vec, c1)
                found = True
                break

    if not found:
        if bool_exp[0] == '(' and bool_exp[-1] == ')':
            return boolean_to_msp_iterative(bool_exp[1:-1], vec, c)

        if '(' in bool_exp or ')' in bool_exp:
            raise ValueError("Bad boolean expression or attributes contain ( or )")

        mat = np.zeros((1, c))
        mat[0, :len(vec)] = vec
        return MSP(mat, [bool_exp]), c

    # 合并两个MSP结构
    mat = np.zeros((len(msp1.mat) + len(msp2.mat), c_out))
    mat[:len(msp1.mat), :msp1.mat.shape[1]] = msp1.mat
    mat[len(msp1.mat):, :] = msp2.mat
    row_to_attrib = msp1.row_to_attrib + msp2.row_to_attrib

    return MSP(mat, row_to_attrib), c_out


def make_and_vecs(vec: np.ndarray, c: int) -> Tuple[np.ndarray, np.ndarray]:
    vec1 = np.zeros(c + 1)
    vec2 = np.zeros(c + 1)
    vec2[:len(vec)] = vec
    vec1[c] = -1
    vec2[c] = 1
    return vec1, vec2


# 使用示例
if __name__ == "__main__":
    bool_exp = "(A AND B) AND (C OR D)"
    msp, _ = boolean_to_msp(bool_exp, False)
    print("Matrix:")
    print(msp.mat)
    print("Row to Attribute mapping:")
    print(msp.row_to_attrib)