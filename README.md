# 交替方向乘子法（ADMM）笔记

## 1. 对偶问题

### 1.1 原始问题与拉格朗日函数
考虑凸优化问题（等式约束）：

<!-- $$
\begin{aligned}
\min_x \quad & f(x) \\
\text{s.t.} \quad & Ax = b
\end{aligned}
$$ --> 

其中 $f(x)$ 是凸函数。

**拉格朗日函数**  
<!-- $$
L(x,y) = f(x) + y^T(Ax - b)
$$ --> 
- $y$ 是对偶变量（拉格朗日乘子）

**对偶函数**  
<!-- $$
g(y) = \inf_x L(x,y)
$$ --> 

### 1.2 对偶上升法
对偶问题：$\max_y g(y)$

**算法步骤**  
1. $x$‑更新：$x^{k+1} = \arg\min_x L(x,y^k)$  
2. 对偶更新：$y^{k+1} = y^k + \alpha^k (Ax^{k+1} - b)$  

### 1.3 对偶分解
当目标函数可分时（$f(x)=\sum_i f_i(x_i)$），拉格朗日函数可分解：

<!-- $$
L(x,y) = \sum_i \Bigl[\,f_i(x_i) + y^T A_i x_i \Bigr] - y^T b
$$ --> 

**分布式计算**  
1. 各节点并行计算：$x_i^{k+1} = \arg\min_{x_i} L_i(x_i,y^k)$  
2. 中心节点聚合：$y^{k+1} = y^k + \alpha^k \!\bigl(\sum_i A_i x_i^{k+1} - b\bigr)$  

---

## 2. 乘子法

### 2.1 增广拉格朗日
为增强稳定性，引入二次惩罚项：

<!-- $$
L_\rho(x,y) = f(x) + y^T(Ax-b) + \frac{\rho}{2}\|Ax-b\|_2^2
$$ --> 

**算法步骤**  
1. $x$‑更新：$x^{k+1} = \arg\min_x L_\rho(x,y^k)$  
2. 对偶更新：$y^{k+1} = y^k + \rho \bigl(Ax^{k+1}-b\bigr)$  

### 2.2 优缺点
- **优点**：收敛条件更宽松  
- **缺点**：二次项破坏可分解性  

---

## 3. 交替方向乘子法（ADMM）

### 3.1 基本形式
处理可分结构的优化问题：

<!-- $$
\begin{aligned}
\min \quad & f(x) + g(z) \\\\
\text{s.t.} \quad & Ax + Bz = c
\end{aligned}
$$ --> 

**增广拉格朗日**  
<!-- $$
L_\rho(x,z,y) = f(x) + g(z) + y^T(Ax + Bz - c) + \frac{\rho}{2}\|Ax+Bz-c\|_2^2
$$ --> 

### 3.2 算法步骤  
1. $x$‑更新：$x^{k+1} = \arg\min_x L_\rho(x,z^k,y^k)$  
2. $z$‑更新：$z^{k+1} = \arg\min_z L_\rho(x^{k+1},z,y^k)$  
3. 对偶更新：$y^{k+1} = y^k + \rho \bigl(Ax^{k+1}+Bz^{k+1}-c\bigr)$  

### 3.3 缩放形式
令 $u = y/\rho$，得到缩放形式：

<!-- $$
\begin{aligned}
x^{k+1} &= \arg\min_x \Bigl( f(x) + \tfrac{\rho}{2}\|Ax+Bz^k-c+u^k\|_2^2 \Bigr) \\\\
z^{k+1} &= \arg\min_z \Bigl( g(z) + \tfrac{\rho}{2}\|Ax^{k+1}+Bz-c+u^k\|_2^2 \Bigr) \\\\
u^{k+1} &= u^k + \bigl(Ax^{k+1}+Bz^{k+1}-c\bigr)
\end{aligned}
$$ --> 

### 3.4 收敛性
**假设条件**  
1. $f,g$ 为凸、闭且适当  
2. 增广拉格朗日存在鞍点  

**收敛结果**  
- 残差收敛：$Ax^k + Bz^k - c \to 0$  
- 目标值收敛：$f(x^k) + g(z^k) \to p^\*$  

---

## 4. 常见函数更新

### 4.1 可分函数
若 $f(x)=\sum_i f_i(x_i)$ 且 $A^T A$ 块对角 ⇒ $x$‑更新可并行计算

### 4.2 近端算子
当 $A = I$ 时，$x$‑更新为

<!-- $$
\operatorname{prox}_{f,\rho}(v) = \arg\min_x \Bigl( f(x) + \tfrac{\rho}{2}\|x-v\|_2^2 \Bigr)
$$ --> 

**特例**  
1. **投影**：$f = I_C \;\Rightarrow\; \Pi_C(v)$  
2. **$L_1$ 正则**：$f = \lambda\|\,\cdot\,\|_1 \;\Rightarrow\;$ 软阈值 $S_{\lambda/\rho}(v_i)$  

### 4.3 二次目标
若 $f(x)=\tfrac12 x^T P x + q^T x + r$ ，则

<!-- $$
x^{+} = \bigl(P + \rho A^T A\bigr)^{-1}\!\bigl(\rho A^T v - q\bigr)
$$ --> 

**计算技巧**  
- 矩阵求逆引理  
- 预计算分解（Cholesky / LDLᵀ）  

---

## 5. 典型应用：Lasso 回归

问题形式  
<!-- $$
\min \tfrac12\|Ax-b\|_2^2 + \lambda\|z\|_1 \quad\text{s.t.}\; x - z = 0
$$ --> 

对应 $f(x)$ 为二次项，$g(z)$ 为 $L_1$ 正则。ADMM 步骤：

<!-- $$
\begin{aligned}
x^{k+1} &= \bigl(A^T A + \rho I\bigr)^{-1}\!\bigl(A^T b + \rho z^k - y^k\bigr) \\\\
z^{k+1} &= S_{\lambda/\rho}\!\bigl(x^{k+1} + y^k/\rho\bigr) \\\\
y^{k+1} &= y^k + \rho\bigl(x^{k+1} - z^{k+1}\bigr)
\end{aligned}
$$ --> 

---

## 6. 总结

**优势**  
- 适用于大规模分布式优化  
- 对不可微项友好  
- 收敛性有理论保证  

**挑战**  
- 罚参数 $\rho$ 需要调节  
- 非凸问题的收敛性更复杂  
