```python
def bubble_sort(arr):
    n = len(arr)
    
    # 遍历所有数组元素
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # 遍历数组从0到n-i-1
            # 如果发现前一个元素比后一个元素大，则交换它们
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# 测试代码
if __name__ == "__main__":
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print("原始数组:", test_array)
    
    sorted_array = bubble_sort(test_array)
    print("排序后数组:", sorted_array)

```

```lean
import Std 

def bubbleSort {α : Type} [Inhabited α] [Ord α] (arr : Array α) : Array α :=
  let rec bubble : Nat → Array α → Array α
    | 0, arr => arr
    | n+1, arr =>
      let rec pass : Nat → Array α → Array α
        | 0, arr => arr
        | i+1, arr =>
          if i < arr.size && compare arr[i-1]! arr[i]! == Ordering.gt then
            pass i (arr.swap! (i-1) i)
          else
            pass i arr
      bubble n (pass arr.size arr)
  
  bubble arr.size arr

#eval bubbleSort #[5, 2, 8, 12, 1, 6]
```


# 在 Lean 4 中使用 `Ord` 类型类进行比较

在 Lean 4 中，我们可以使用 `Ord` 类型类的 `compare` 函数来比较包括但不限于以下类型：

## 1. 基本数值类型

- **整数类型**（如 `Nat`、`Int`、`UInt64` 等）
- **浮点数类型**（如 `Float`）

在进行跨类型比较时，通常需要进行强制类型转换。以下是一个大致的无损转换顺序表，从"小"类型到"大"类型：

1. `Nat` (自然数)
2. `UInt8`
3. `UInt16`
4. `UInt32`
5. `UInt64`
6. `Int8`
7. `Int16`
8. `Int32`
9. `Int64`
10. `Int` (任意精度整数)
11. `Float` (单精度浮点数)
12. `Double` (双精度浮点数)

### 转换顺序原则

1. **无符号整数类型**按位数排序。
2. **有符号整数类型**按位数排序，通常位于同位数的无符号类型之后。
3. `Int` (任意精度整数) 可以无损地表示所有固定位数的整数类型。
4. 浮点数类型位于整数类型之后，因为它们可以表示小数。
5. `Double` 比 `Float` 有更大的范围和精度。

在进行跨类型比较时，通常应该将"小"类型转换为"大"类型以避免信息丢失。例如：

- 比较 `Nat` 和 `Int`：将 `Nat` 转换为 `Int`
- 比较 `Int32` 和 `Int64`：将 `Int32` 转换为 `Int64`
- 比较任何整数类型和 `Float`：将整数转换为 `Float`

### 注意事项

1. 虽然这个顺序表示了无损转换的一般路径，但并非所有相邻类型之间的转换都是无损的。例如，某些大的 `UInt64` 值转换为 `Int64` 时可能会溢出。
2. 浮点数虽然在列表末尾，但它们不能精确表示所有整数。大整数转换为浮点数可能会损失精度。
3. 在实际编程中，Lean 的类型系统会防止许多隐式的不安全转换，你通常需要显式地进行类型转换。
4. 对于特定的应用场景，可能需要根据具体需求调整这个顺序或选择合适的转换策略。

## 2. 字符和字符串

- `Char`
- `String`

在字符串的偏序关系中，`"a" < "b"`。

## 3. 布尔值

- `Bool`

在布尔值的偏序关系中，`false < true`。

## 4. 复合类型

- **元组** (Tuples)
- **列表** (Lists)
- **数组** (Arrays)

### 序列类型的通用偏序关系

对于任何序列类型 `S`（如元组、列表、数组等），其中包含元素类型 `T`（`T` 实现了 `Ord` 类型类），定义偏序关系 `≤` 如下：

1. **空序列**：
   - 空序列 `≤` 任何序列

2. **非空序列**：
   - 对于非空序列 `a = (a₁, a₂, ..., aₙ)` 和 `b = (b₁, b₂, ..., bₘ)`，`a ≤ b` 当且仅当满足以下条件之一：
     - `a₁ < b₁`
     - `a₁ = b₁` 且 `(a₂, ..., aₙ) ≤ (b₂, ..., bₘ)`

3. **序列长度**：
   - 如果 `a` 是 `b` 的真前缀，则 `a < b`

4. **相等性**：
   - `a = b` 当且仅当 `n = m` 且对所有 `i`，`aᵢ = bᵢ`

### 特性

1. **递归性**：这个定义是递归的，基于元素的比较逐步构建整个序列的比较。
2. **字典序**：这种排序方式本质上是字典序（lexicographical order）。
3. **前缀关系**：较短的序列如果是较长序列的前缀，则被视为较小。
4. **类型一致性**：要求序列中的所有元素类型都实现了 `Ord` 类型类。
5. **传递性**：如果 `a ≤ b` 且 `b ≤ c`，则 `a ≤ c`。
6. **反对称性**：如果 `a ≤ b` 且 `b ≤ a`，则 `a = b`。
7. **完全性**：对于任意两个序列 `a` 和 `b`，要么 `a ≤ b`，要么 `b ≤ a`。

## 5. 可选类型

- `Option α`，其中 `α` 是任何可比较的类型

`Option α` 是一个表示可能存在或不存在值的类型。它有两个构造器：

- `none`：表示没有值
- `some a`：表示存在一个类型为 `α` 的值 `a`

### 比较规则

- `none` 被认为小于任何 `some` 值
- 两个 `some` 值的比较取决于它们包含的值的比较

### 例子

```lean
#eval compare (Option.none : Option Nat) (Option.some 5)  -- 结果: Ordering.lt
#eval compare (Option.some 3) (Option.some 5)             -- 结果: Ordering.lt
#eval compare (Option.some 5) (Option.some 5)             -- 结果: Ordering.eq
```

## 6. 结果类型

- `Result ε α`，其中 `ε` 和 `α` 都是可比较的类型

`Result ε α` 是一个表示可能成功或失败的操作结果的类型。它有两个构造器：

- `ok a`：表示操作成功，`a` 是类型为 `α` 的成功值
- `error e`：表示操作失败，`e` 是类型为 `ε` 的错误信息

### 比较规则

- `error` 值小于任何 `ok` 值
- 两个 `error` 值之间的比较取决于它们的错误值 `ε` 的比较
- 两个 `ok` 值之间的比较取决于它们的成功值 `α` 的比较

### 例子

```lean
#eval compare (Result.error "Error A" : Result String Nat) (Result.ok 5)  -- 结果: Ordering.lt
#eval compare (Result.error "Error A") (Result.error "Error B")           -- 结果: Ordering.lt
#eval compare (Result.ok 3) (Result.ok 5)                                 -- 结果: Ordering.lt
```

## 7. 集合类型

如 `Set`、`Map` 等，假设它们的元素类型是可比较的。

### 集合 (Set) 的比较

在 Lean 中，`Set α` 通常表示元素类型为 `α` 的集合。集合的比较基于它们的元素和子集关系。

#### 比较规则

a) **大小比较**：
   - 如果 `A` 是 `B` 的真子集，则 `A < B`
   - 如果 `A = B`（即它们包含相同的元素），则 `A == B`
   - 如果 `A` 和 `B` 都不是对方的子集，则它们是不可比较的

b) **字典序比较**（当需要全序关系时）：
   - 将集合转换为有序列表
   - 按字典序比较这些列表

#### 例子

```lean
import Std.Data.Set

open Std

def setA : Set Nat := Set.ofList [1, 2, 3]
def setB : Set Nat := Set.ofList [1, 2, 3, 4]
def setC : Set Nat := Set.ofList [1, 2, 4]

#eval setA < setB  -- 输出: true （A 是 B 的真子集）
#eval setA == setB -- 输出: false
#eval setA < setC  -- 在标准的集合比较中，这是不可比较的
```
### 映射 (Map) 的比较

在 Lean 中，`Map κ ν` 通常表示键类型为 `κ`，值类型为 `ν` 的映射。映射的比较通常基于它们的键值对。

#### 比较规则

a) **大小比较**：
   - 如果 `M1` 的所有键值对都在 `M2` 中，且 `M2` 比 `M1` 有更多的键值对，则 `M1 < M2`
   - 如果 `M1` 和 `M2` 有完全相同的键值对，则 `M1 == M2`

b) **字典序比较**（当需要全序关系时）：
   - 将映射转换为键值对的有序列表
   - 首先比较键，如果键相同，则比较值
   - 按这个顺序进行字典序比较

#### 例子

```lean
import Std.Data.HashMap

open Std

def mapA : HashMap String Nat := HashMap.ofList [("a", 1), ("b", 2)]
def mapB : HashMap String Nat := HashMap.ofList [("a", 1), ("b", 2), ("c", 3)]
def mapC : HashMap String Nat := HashMap.ofList [("a", 1), ("b", 3)]

#eval mapA < mapB  -- 输出: true （A 的所有键值对都在 B 中，且 B 更大）
#eval mapA == mapB -- 输出: false
#eval mapA < mapC  -- 这种比较可能需要自定义实现
```

## 总结

在 Lean 4 中，`Ord` 类型类提供了一种强大的机制来比较不同类型的值。通过 `compare` 函数，我们可以实现对基本数值类型、字符和字符串、布尔值、复合类型（如元组、列表、数组）、可选类型、结果类型以及集合和映射的比较。


# For 循环变为递归

将 for 循环转换为递归函数是一种常见的函数式编程技巧。这里是一个一般性的方法来将 for 循环转换为递归函数：

1. 确定循环变量和循环体
2. 创建一个递归函数，其参数包括循环变量和任何在循环中更新的状态
3. 使用条件语句来检查循环结束条件
4. 在递归调用中更新循环变量和状态

以下是一个通用模板：

```lean
-- 假设的 for 循环
for (i := start; i < end; i := i + step) {
  // 循环体
}

-- 转换为递归函数
def recursiveFunction (i : Nat) (state : State) : State :=
  if i >= end then
    state  -- 基本情况：循环结束
  else
    let newState := -- 执行循环体，更新状态
    recursiveFunction (i + step) newState  -- 递归调用

-- 使用递归函数
let result := recursiveFunction start initialState
```

让我们看一个具体的例子，比如计算 1 到 n 的和：

```lean
-- 使用 for 循环的伪代码
sum := 0
for (i := 1; i <= n; i := i + 1) {
  sum := sum + i
}

-- 转换为 Lean 中的递归函数
def sumToN (n : Nat) : Nat :=
  let rec loop (i : Nat) (sum : Nat) : Nat :=
    if i > n then
      sum  -- 基本情况：循环结束
    else
      loop (i + 1) (sum + i)  -- 递归调用，更新 sum
  loop 1 0  -- 从 1 开始，初始 sum 为 0

#eval sumToN 5  -- 输出: 15
```

在这个例子中：

1. 循环变量 `i` 成为递归函数 `loop` 的参数。
2. 累加的 `sum` 也作为参数传递。
3. 循环条件 `i <= n` 转换为递归的终止条件 `i > n`。
4. 循环体中的操作（这里是 `sum + i`）在递归调用中执行。

```
def swap (arr : List Nat) (i j : Nat) : List Nat :=
  if i == j then arr
  else if i > j then swap arr j i
  else
    let n := arr.length
    if i >= n || j >= n then arr
    else
      let vi := arr[i]!
      let vj := arr[j]!
      let rec loop (l : List Nat) (idx : Nat) (acc : List Nat) : List Nat :=
        match l with
        | [] => acc.reverse
        | x :: xs =>
          if idx == i then loop xs (idx + 1) (vj :: acc)
          else if idx == j then loop xs (idx + 1) (vi :: acc)
          else loop xs (idx + 1) (x :: acc)
      loop arr 0 []

def bubbleSort (arr : List Nat) : List Nat :=
  let n := arr.length
  let rec outerLoop (arr : List Nat) (i : Nat) : List Nat :=
    if i < n then
      let rec innerLoop (arr : List Nat) (j : Nat) : List Nat :=
        if j < n - i - 1 then
          let arr :=
            if arr[j]! > arr[j+1]! then
              swap arr j (j+1)
            else
              arr
          innerLoop arr (j + 1)
        else
          arr
      outerLoop (innerLoop arr 0) (i + 1)
    else
      arr
  outerLoop arr 0

#eval bubbleSort [64, 34, 25, 12, 22, 11, 90]
```

# 转换的一般逻辑：

将Python代码中的for循环首先转换为递归，再让模型处理递归的代码转换

在lean4中，列表、数组是不可交换

```lean
import Std

partial def quicksort (arr : List Nat) : List Nat :=
  match arr with
  | [] => []
  | pivot :: rest =>
    let left := rest.filter (· < pivot)
    let middle := pivot :: rest.filter (· == pivot)
    let right := rest.filter (· > pivot)
    quicksort left ++ middle ++ quicksort right

#eval quicksort [3, 6, 8, 10, 1, 2, 1]

def swapArray [Inhabited α] (arr : Array α) (i j : Nat) : Except String (Array α) := do
  if i == j then
    pure arr
  else if i < arr.size && j < arr.size then
    let vi := arr[i]!
    let vj := arr[j]!
    pure (arr.set! i vj |>.set! j vi)
  else
    throw s!"Index out of bounds: i = {i}, j = {j}, array size = {arr.size}"

def printSwapResult (arr : Array Nat) (i j : Nat) : IO Unit := do
  match swapArray arr i j with
  | .ok result => 
    IO.println s!"Swapped array: {result}"
  | .error msg => 
    IO.println s!"Error: {msg}"

#eval printSwapResult #[1, 2, 3, 4, 5] 1 3
#eval printSwapResult #[1, 2, 3, 4, 5] 1 41

partial def insert (a : Nat) (xs : List Nat) : List Nat :=
  match xs with
  | [] => [a]
  | x :: rest =>
    if a <= x then
      a :: xs
    else
      x :: insert a rest

partial def insertionSort (xs : List Nat) : List Nat :=
  match xs with
  | [] => []
  | x :: rest =>
    insert x (insertionSort rest)

#eval insertionSort [5, 2, 8, 12, 1, 6]

partial def merge (left : List Nat) (right : List Nat) : List Nat :=
  match left, right with
  | [], r => r
  | l, [] => l
  | x::xs, y::ys =>
    if x <= y then
      x :: merge xs right
    else
      y :: merge left ys

partial def mergeSort (list : List Nat) : List Nat :=
  match list with
  | [] => []
  | [x] => [x]
  | xs =>
    let mid := xs.length / 2
    let (left, right) := xs.splitAt mid
    merge (mergeSort left) (mergeSort right)

#eval mergeSort [5, 2, 8, 12, 1, 6]




def getElementAt (arr : List Nat) (i : Nat) : Except String Nat :=
  if i < arr.length then
    return arr[i]!
  else
    throw s!"Index out of bounds: {i}, list length = {arr.length}"

def swap (arr : List Nat) (i j : Nat) : Except String (List Nat) :=
  if i == j then
    return arr
  else if i > j then
    swap arr j i
  else
    let n := arr.length
    if i >= n || j >= n then
      throw s!"Index out of bounds: i = {i}, j = {j}, list length = {n}"
    else
      let vi := getElementAt arr i
      let vj := getElementAt arr j
      match vi, vj with
      | .ok vi_value, .ok vj_value =>
        let rec loop (l : List Nat) (idx : Nat) (acc : List Nat) : List Nat :=
          match l with
          | [] => acc.reverse
          | x :: xs =>
            if idx == i then loop xs (idx + 1) (vj_value :: acc)
            else if idx == j then loop xs (idx + 1) (vi_value :: acc)
            else loop xs (idx + 1) (x :: acc)
        return loop arr 0 []
      | .error msg1, _ => throw msg1
      | _, .error msg2 => throw msg2

-- 测试代码
def testSwap : IO Unit := do
  let arr := [1, 2, 3, 4, 5]

  match swap arr 1 3 with
  | .ok result =>
    IO.println s!"Swapped array: {result}"
  | .error msg =>
    IO.println s!"Error: {msg}"

  match swap arr 1 5 with
  | .ok result =>
    IO.println s!"Swapped array: {result}"
  | .error msg =>
    IO.println s!"Error: {msg}"

#eval testSwap

def getElementAtArray[Inhabited α] (arr : Array α) (i : Nat) : Except String α :=
  if i < arr.size then
    return arr[i]!
  else
    throw s!"Index out of bounds: {i}, array size = {arr.size}"

-- 测试代码
def testGetElementAt : IO Unit := do
  let arr := #[1, 2, 3, 4, 5]

  match getElementAtArray arr 2 with
  | .ok value => 
    IO.println s!"Value at index 2: {value}"
  | .error msg => 
    IO.println s!"Error: {msg}"

  match getElementAtArray arr 10 with
  | .ok value => 
    IO.println s!"Value at index 10: {value}"
  | .error msg => 
    IO.println s!"Error: {msg}"

#eval testGetElementAt



open Std

-- 定义一个从 HashMap 获取值的函数
def getValueAt (dict : HashMap String Nat) (key : String) : Except String Nat :=
  match dict.get? key with
  | some value => return value
  | none => throw s!"Key not found: {key}"

-- 测试字典查找
def testGetValueAt : IO Unit := do
  -- 创建一个字典 (HashMap)
  let dict : HashMap String Nat := HashMap.empty
    |>.insert "a" 1
    |>.insert "b" 2
    |>.insert "c" 3

  -- 测试有效键
  match getValueAt dict "b" with
  | .ok value => IO.println s!"Value for 'b': {value}"
  | .error msg => IO.println s!"Error: {msg}"

  -- 测试无效键
  match getValueAt dict "x" with
  | .ok value => IO.println s!"Value for 'x': {value}"
  | .error msg => IO.println s!"Error: {msg}"

#eval testGetValueAt
```
# Cp0 基础工具
range的创建：

`.range default_end`来表示从0到default_end-1

`.range start end`来表示从start到end-1

如何使用自定义函数进行步长设置：
```
partial def rangeWithStep (start : Int) (end1 : Int) (step : Int) : List Int :=
  if step == 0 then
    []  -- 步长为0时返回空列表
  else
    let rec loop (current : Int) (acc : List Int) : List Int :=
      if (step > 0 && current >= end1) || (step < 0 && current <= end1) then
        acc
      else
        loop (current + step) (current :: acc)
    
    if step > 0 then
      (loop start []).reverse  -- 正序
    else
      loop start []  -- 倒序

-- 测试
#eval rangeWithStep 0 10 2    -- 输出: [0, 2, 4, 6, 8]
#eval rangeWithStep 10 0 (-2) -- 输出: [10, 8, 6, 4, 2]
#eval rangeWithStep 0 5 1     -- 输出: [0, 1, 2, 3, 4]
#eval rangeWithStep 5 0 (-1)  -- 输出: [5, 4, 3, 2, 1]
#eval rangeWithStep 0 10 0    -- 输出: []
```

关于let和def

1. `let` 声明：
   - 用于创建局部变量或局部函数。
   - 只在其作用域内可见（比如在一个函数内部或一个 `do` 块内）。
   - 不能在模块顶层使用（除非在某些特殊情况下，如 `#eval` 命令中）。
   - 值可以是表达式。

2. `def` 声明：
   - 用于定义全局函数或常量。
   - 在整个模块中可见，也可以被其他模块导入和使用。
   - 可以在模块顶层使用。
   - 通常需要指定类型（虽然在某些情况下可以省略）。

例子：

```lean
def globalValue : Nat := 10  -- 全局常量

def exampleFunction : IO Unit := do
  let localValue := 20  -- 局部变量
  IO.println s!"Global value: {globalValue}"
  IO.println s!"Local value: {localValue}"

-- 这是正确的
#eval globalValue

-- 这会报错，因为 localValue 只在 exampleFunction 内部可见
-- #eval localValue  

-- 在模块顶层使用 let 通常会报错
-- let topLevelLet := 30  

-- 但在 #eval 中使用 let 是允许的
#eval let x := 5; x * 2
```



# Cp1 列表与数组
# Lean 4 与 Python 数组/列表操作对照表

Python 的列表是一种非常灵活和常用的数据结构，它有很多特性和方法。以下是一个综合的列表：
1. 创建和初始化
   - 空列表: `[]`
   - 含元素列表: `[1, 2, 3]`
   - 列表推导式: `[x for x in range(10)]`
   - 用 `list()` 函数创建: `list('abc')`
2. 访问元素
   - 索引访问: `lst[0]`, `lst[-1]`
   - 切片: `lst[1:3]`, `lst[::-1]`
3. 修改元素
   - 单个元素赋值: `lst[0] = 10`
   - 切片赋值: `lst[1:3] = [4, 5]`
4. 添加元素
   - 在末尾添加: `lst.append(4)`
   - 在指定位置插入: `lst.insert(1, 5)`
   - 扩展列表: `lst.extend([4, 5, 6])` 或 `lst += [4, 5, 6]`
5. 删除元素
   - 删除指定索引的元素: `del lst[1]`
   - 移除第一个匹配的值: `lst.remove(3)`
   - 弹出元素: `lst.pop()` 或 `lst.pop(1)`
   - 清空列表: `lst.clear()`
6. 列表长度和包含关系
   - 长度: `len(lst)`
   - 检查元素是否在列表中: `3 in lst`
7. 列表操作
   - 连接列表: `lst1 + lst2`
   - 重复列表: `lst * 3`
8. 排序和反转
   - 排序: `lst.sort()` 或 `sorted(lst)`
   - 反转: `lst.reverse()` 或 `reversed(lst)`
9. 计数和索引
   - 计数元素出现次数: `lst.count(3)`
   - 查找元素索引: `lst.index(3)`
10. 复制
    - 浅复制: `lst.copy()` 或 `lst[:]`
    - 深复制: `import copy; copy.deepcopy(lst)`
11. 列表推导和生成
    - 列表推导: `[x**2 for x in range(10)]`
    - 使用 map: `list(map(lambda x: x**2, range(10)))`
12. 其他操作
    - 最大值和最小值: `max(lst)`, `min(lst)`
    - 求和: `sum(lst)`
13. 多维列表
    - 创建: `[[0 for _ in range(3)] for _ in range(3)]`
14. 列表作为栈或队列
    - 栈操作: `lst.append()` 和 `lst.pop()`
    - 队列操作: `lst.append()` 和 `lst.pop(0)` (不推荐用于大型列表)

接下来，我们可以逐一查找这些操作在 Lean 4 中的对应实现，并为它们编写测试。

### 1.初始化
关于lean4列表的初始化：
1. 在指定列表类型之后，我们可以使用一般的
`let nonEmptyList : List Int := [1, 2, 3]`来创建列表
如果我们要用范围，可以使用`List.range max_num`

### 2.访问元素

```lean
def example_index_access : IO Unit := do
  let lst : List Int := [1, 2, 3, 4, 5]

  -- 正向索引访问
  IO.println s!"First element: {lst.get? 0}"
  -- 使用 ! 操作符（不安全，可能会在越界时崩溃）
  IO.println s!"Second element: {lst[1]!}"

  -- Lean 没有内置的负索引支持，但我们可以自己实现
  let last := lst.get? (lst.length - 1)
  IO.println s!"Last element: {last}"

#eval example_index_access
```