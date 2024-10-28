# Lean4 数学教程：从线性方程到牛顿法

## 引言

在数学和计算机科学的交叉领域，形式化方法和定理证明系统正在发挥越来越重要的作用。Lean4 作为一种先进的定理证明助手和函数式编程语言，为我们提供了一个强大的工具来探索数学概念、实现算法，并验证其正确性。

本教程的目标是通过一系列渐进复杂的数学问题，展示 Lean4 在数学建模和计算方面的能力。我们将从简单的线性方程开始，逐步深入到二次方程的求解，最后介绍牛顿法这一强大的数值方法。通过这个过程，我们不仅能学习如何使用 Lean4 解决实际问题，还能深入理解数学概念与计算机实现之间的联系。

## 1. 线性方程：精确性与计算效率的权衡

我们首先从最基本的数学概念之一 —— 线性方程开始。线性方程形如 ax + b = 0，其中 a 和 b 是常数，x 是未知数。在 Lean4 中，我们可以用多种方式表示和求解这类方程，每种方式都有其独特的优势和局限性。

### 1.1 使用有理数 (ℚ)

```lean
import Mathlib.Data.Real.Basic

structure LinearEquation where
  a : ℚ
  b : ℚ
deriving Repr

def solveLinearEquation (eq : LinearEquation) : Option ℚ :=
  if eq.a = 0 then
    if eq.b = 0 then some 0 else none
  else
    some (-eq.b / eq.a)

-- 示例使用代码省略...
```

使用有理数的优势在于它提供了精确的结果，避免了浮点数计算中的舍入误差。这对于需要高精度的数学证明或金融计算特别有用。然而，有理数运算可能在某些情况下效率较低，特别是当分子或分母变得非常大时。

### 1.2 使用浮点数 (Float)

```lean
import Mathlib

structure LinearEquation where
  a : Float
  b : Float

def solveLinearEquation (eq : LinearEquation) : Option Float :=
  if eq.a == 0 then
    if eq.b == 0 then some 0 else none
  else
    some (-eq.b / eq.a)

-- 示例使用代码省略...
```

浮点数实现提供了计算效率和实用性之间的平衡。它允许快速计算，并且可以处理非常大或非常小的数值。然而，浮点数计算可能导致精度损失，这在某些科学计算或金融应用中可能是不可接受的。

### 1.3 使用实数 (ℝ)

```lean
import Mathlib.Data.Real.Basic

structure LinearEquation where
  a : ℝ
  b : ℝ

noncomputable def solveLinearEquation (eq : LinearEquation) : Option ℝ :=
  if eq.a = 0 then
    if eq.b = 0 then some 0 else none
  else
    some (-eq.b / eq.a)
```

使用实数类型提供了理论上的精确性，这在数学证明中非常有用。然而，实数在 Lean4 中被标记为 `noncomputable`，意味着它们不能用于实际的计算。这种实现主要用于形式化数学理论和定理证明。

通过比较这三种实现，我们可以深入理解数值表示和计算的不同方法，以及它们在精确性、效率和理论基础方面的权衡。这种理解对于选择合适的数据类型和算法来解决实际问题至关重要。

## 2. 二次方程：处理复杂性和特殊情况

接下来，我们探讨更复杂的二次方程求解。二次方程不仅引入了更多的数学复杂性，还需要我们考虑多种特殊情况，如无解、单解和双解。

```lean
import Mathlib

structure QuadraticEquation where
  a : Float
  b : Float
  c : Float
deriving Repr

def solveQuadraticEquation (eq : QuadraticEquation) : Option (Float × Float) :=
  if eq.a == 0 then
    -- 处理退化为线性方程的情况
    if eq.b != 0 then
      let x := -eq.c / eq.b
      some (x, x)
    else
      none
  else
    -- 计算判别式
    let discriminant := eq.b * eq.b - 4 * eq.a * eq.c
    if discriminant < 0 then
      none  -- 无实数解
    else if discriminant == 0 then
      -- 唯一解
      let x := -eq.b / (2 * eq.a)
      some (x, x)
    else
      -- 两个不同的解
      let sqrtD := Float.sqrt discriminant
      let x1 := (-eq.b + sqrtD) / (2 * eq.a)
      let x2 := (-eq.b - sqrtD) / (2 * eq.a)
      some (x1, x2)

-- 示例使用代码省略...
```

这个实现展示了如何处理各种边界情况和特殊情况，这是软件工程中的一个关键技能。我们需要考虑：
1. 方程退化为线性方程的情况 (a = 0)
2. 无解的情况（判别式小于0）
3. 单解的情况（判别式等于0）
4. 双解的情况（判别式大于0）

通过这个例子，我们不仅学习了如何在 Lean4 中实现复杂的数学算法，还深入理解了数学理论与实际编程之间的联系。

## 3. 牛顿法：数值方法与迭代算法

最后，我们介绍牛顿法，这是一种强大的数值方法，用于寻找函数的根。牛顿法展示了如何将微积分概念转化为计算机算法。

```lean
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt

structure Function where
  f : Float → Float
  f' : Float → Float  -- 导数

def newtonMethod (func : Function) (x0 : Float) (tolerance : Float) (maxIterations : Nat) : Float :=
  let rec iterate (x : Float) (n : Nat) : Float :=
    if n = 0 then x
    else
      let fx := func.f x
      if fx.abs < tolerance then x
      else
        let x' := x - fx / func.f' x
        iterate x' (n - 1)
  iterate x0 maxIterations

-- 示例使用代码省略...
```

牛顿法的实现体现了几个重要的编程和数学概念：
1. 递归和迭代：通过递归函数实现迭代过程。
2. 收敛条件：使用容差（tolerance）来判断是否达到足够精确的解。
3. 函数作为一等公民(即可以赋值、可以作为参数传递、可以被返回)：将函数及其导数作为参数传递。

```lean
-- 线性方程示例
def example1 := LinearEquation.mk 2 (-4)
#eval printSolution example1

-- 二次方程示例
def quadExample := QuadraticEquation.mk 1 (-5) 6
#eval printSolution quadExample

-- 牛顿法示例
def sqrt2Approximation : Float :=
  newtonMethod squareRootFunction 1.5 0.0001 20
#eval sqrt2Approximation
```