# 交替方向乘子法（ADMM）笔记

## 1. 对偶问题

### 1.1 原始问题与拉格朗日函数
考虑凸优化问题（等式约束）：
```math
\begin{aligned}
\min_x \quad & f(x) \\
\text{s.t.} \quad & Ax = b
\end{aligned}
```
其中 $f(x)$ 是凸函数。

**拉格朗日函数**：

$$ L(x,y) = f(x) + y^T(Ax - b) $$
-  $y$ 是对偶变量（拉格朗日乘子）

**对偶函数**：

$$ g(y) = \inf_x L(x,y) $$

### 1.2 对偶上升法
对偶问题： $\max_y g(y)$

**算法步骤**：
1.  $x$ -更新：  $x^{k+1} = \text{argmin}_x L(x,y^k)$
2. 对偶更新：  $y^{k+1} = y^k + \alpha^k (Ax^{k+1} - b)$

### 1.3 对偶分解
当目标函数可分时（ $f(x)=\sum_i f_i(x_i)$ ），拉格朗日函数可分解：

$$ L(x,y) = \sum_i \left[ f_i(x_i) + y^TA_ix_i \right] - y^Tb $$

**分布式计算**：
1. 各节点并行计算：  $x_i^{k+1} = \text{argmin}_{x_i} L_i(x_i,y^k)$
2. 中心节点聚合：  $y^{k+1} = y^k + \alpha^k (\sum_i A_ix_i^{k+1} - b)$

## 2. 乘子法

### 2.1 增广拉格朗日
为增强稳定性，引入二次惩罚项：

$$ L_\rho(x,y) = f(x) + y^T(Ax-b) + \frac{\rho}{2}\|Ax-b\|_2^2 $$

**算法步骤**：
1.  $x$ -更新：  $x^{k+1} = \text{argmin}_x L_\rho(x,y^k)$
2. 对偶更新：  $y^{k+1} = y^k + \rho(Ax^{k+1}-b)$

### 2.2 优缺点
- 优点：收敛条件更宽松
- 缺点：二次项破坏可分解性

## 3. 交替方向乘子法（ADMM）

### 3.1 基本形式
处理可分结构的优化问题：
```math
\begin{aligned}
\min \quad & f(x) + g(z) \\
\text{s.t.} \quad & Ax + Bz = c
\end{aligned}
```

**增广拉格朗日**：

$$ L_\rho(x,z,y) = f(x)+g(z)+y^T(Ax+Bz-c)+\frac{\rho}{2}\|Ax+Bz-c\|_2^2 $$

### 3.2 算法步骤
1.  $x$ -更新：  $x^{k+1} = \text{argmin}_x L_\rho(x,z^k,y^k)$
2.  $z$ -更新：  $z^{k+1} = \text{argmin}_z L_\rho(x^{k+1},z,y^k)$
3. 对偶更新：  $y^{k+1} = y^k + \rho(Ax^{k+1}+Bz^{k+1}-c)$

### 3.3 缩放形式
令 $u = y/\rho$ ，得到缩放形式：

$$
\begin{aligned}
x^{k+1} &= \text{argmin}_x \left( f(x) + \frac{\rho}{2}\|Ax+Bz^k-c+u^k\|_2^2 \right) \\
z^{k+1} &= \text{argmin}_z \left( g(z) + \frac{\rho}{2}\|Ax^{k+1}+Bz-c+u^k\|_2^2 \right) \\
u^{k+1} &= u^k + (Ax^{k+1}+Bz^{k+1}-c)
\end{aligned}
$$

### 3.4 收敛性
**假设条件**：
1.  $f,g$ 为凸、闭、正常函数
2. 增广拉格朗日存在鞍点

**收敛结果**：
- 残差收敛：  $Ax^k+Bz^k-c \to 0$
- 目标值收敛：  $f(x^k)+g(z^k) \to p^*$

## 4. 常见函数更新

### 4.1 可分函数
当 $f(x)=\sum_i f_i(x_i)$ 且 $A^TA$ 块对角时：
-  $x^+$ -更新可分解为并行子问题

### 4.2 近端算子
当 $A=I$ 时， $x$ 更新为：

$$ \text{prox}_{f,\rho}(v) = \text{argmin}_x \left( f(x) + \frac{\rho}{2}\|x-v\|_2^2 \right) $$

**特例**：
1. 投影： $f=I_C \Rightarrow \Pi_C(v)$
2. L1正则： $f=\lambda\|\cdot\|_1 \Rightarrow $ 软阈值 $ S_{\lambda/\rho}(v_i)$

### 4.3 二次目标
当 $f(x)=\frac{1}{2}x^TPx+q^Tx+r$ 时：

$$ x^+ = (P+\rho A^TA)^{-1}(\rho A^Tv - q) $$

**计算技巧**：
- 矩阵求逆引理
- 预计算分解

## 5. 典型应用：Lasso回归
问题形式：

$$ \min \frac{1}{2}\|Ax-b\|_2^2 + \lambda\|z\|_1 \quad \text{s.t.} \ x-z=0 $$

相当于 $f(x)$ 为二次目标， $f(z)$ 为L1正则。ADMM步骤：

$$
\begin{aligned}
x^{k+1} &= (A^TA+\rho I)^{-1}(A^Tb + \rho z^k - y^k) \\
z^{k+1} &= S_{\lambda/\rho}(x^{k+1}+y^k/\rho) \\
y^{k+1} &= y^k + \rho(x^{k+1}-z^{k+1})
\end{aligned}
$$

## 6. 总结

**优势**：
- 适用于大规模分布式优化
- 对不可微函数友好
- 收敛性有保证

**挑战**：
- 参数 $\rho$ 需要调节
- 非凸问题收敛性复杂
