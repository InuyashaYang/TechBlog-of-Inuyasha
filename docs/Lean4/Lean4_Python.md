

# Lean4 列表与数组操作教程

## 引言

Lean4 是一种强大的函数式编程语言，在处理列表和数组等数据结构时有其独特的方法。本教程将详细介绍 Lean4 中的列表和数组操作，并与 Python 的相应操作进行对比，帮助您更好地理解和使用 Lean4。

## 1. 基础工具

### 1.1 范围创建

Lean4 提供了几种创建范围的方法：

```lean
-- 创建从 0 到 9 的范围
#eval List.range 10

-- 创建从 5 到 9 的范围
#eval List.range 5 10
```

### 1.2 自定义步长范围函数

对于需要自定义步长的情况，我们可以实现以下函数：

```lean
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
```

### 1.3 let 和 def 的区别

1. `let` 声明：用于创建局部变量或局部函数。
2. `def` 声明：用于定义全局函数或常量。

例子：

```lean
def globalValue : Nat := 10  -- 全局常量

def exampleFunction : IO Unit := do
  let localValue := 20  -- 局部变量
  IO.println s!"Global value: {globalValue}"
  IO.println s!"Local value: {localValue}"

#eval globalValue
-- #eval localValue  -- 这会报错

#eval let x := 5; x * 2
```

## 2. 列表操作

### 2.1 创建和初始化

#### Python
```python
empty_list = []
non_empty_list = [1, 2, 3]
list_comprehension = [x for x in range(10)]
```

#### Lean4
```lean
def emptyList : List Int := []
def nonEmptyList : List Int := [1, 2, 3]
def listFromRange : List Nat := List.range 10
-- Lean4 使用 List.map 实现类似列表推导式的功能
def listComprehension : List Nat := (List.range 10).map (fun x => x * 2)
```

### 2.2 访问元素

#### Python
```python
lst = [1, 2, 3, 4, 5]
first = lst[0]
last = lst[-1]
slice = lst[1:3]
```

#### Lean4
```lean
def example_index_access : IO Unit := do
  let lst : List Int := [1, 2, 3, 4, 5]

  IO.println s!"First element: {lst.get? 0}"
  IO.println s!"Second element: {lst[1]!}"  -- 使用 ! 操作符（不安全）

  -- 获取最后一个元素
  let last := lst.get? (lst.length - 1)
  IO.println s!"Last element: {last}"

  -- 模拟切片操作
  let slice := (lst.drop 1).take 2
  IO.println s!"Slice: {slice}"

#eval example_index_access
```

### 2.3 修改元素

Lean4 的 List 是不可变的，需要创建新列表来"修改"元素。

```lean
def modifyList (lst : List Int) : List Int :=
  match lst with
  | [] => []
  | x :: xs => (x + 1) :: xs

#eval modifyList [1, 2, 3]  -- 输出: [2, 2, 3]
```

### 2.4 添加元素

#### Python
```python
lst = [1, 2, 3]
lst.append(4)
lst.insert(1, 5)
lst.extend([6, 7])
```

#### Lean4
```lean
def exampleAddElements : IO Unit := do
  let lst := [1, 2, 3]
  
  let lstAppended := lst.append [4]
  IO.println s!"After append: {lstAppended}"

  let lstInserted := lst.insertAt 1 5
  IO.println s!"After insert: {lstInserted}"

  let lstExtended := lst ++ [6, 7]
  IO.println s!"After extend: {lstExtended}"

#eval exampleAddElements
```

### 2.5 删除元素

Lean4 中，我们通过创建新列表来"删除"元素：

```lean
def removeElement (lst : List Int) (elem : Int) : List Int :=
  lst.filter (· ≠ elem)

#eval removeElement [1, 2, 3, 2, 4] 2  -- 输出: [1, 3, 4]
```

### 2.6 排序

```lean
def sortList (lst : List Int) : List Int :=
  lst.qsort (·<=·)

#eval sortList [3, 1, 4, 1, 5, 9, 2, 6]
```

## 3. 数组操作

Lean4 中的数组是可变的，但操作方式与列表有所不同。

### 3.1 创建和初始化

```lean
def emptyArray : Array Int := #[]
def nonEmptyArray : Array Int := #[1, 2, 3]
```

### 3.2 访问和修改元素

```lean
def arrayOperations : IO Unit := do
  let mut arr := #[1, 2, 3, 4, 5]
  
  IO.println s!"Original array: {arr}"
  
  -- 访问元素
  IO.println s!"First element: {arr[0]!}"
  
  -- 修改元素
  arr := arr.set! 1 10
  IO.println s!"After modification: {arr}"
  
  -- 添加元素
  arr := arr.push 6
  IO.println s!"After adding element: {arr}"
  
  -- 删除元素
  arr := arr.eraseIdx 2
  IO.println s!"After removing element: {arr}"

#eval arrayOperations
```

## 4. 高级操作

### 4.1 映射 (Map)

```lean
def mapExample (lst : List Int) : List Int :=
  lst.map (· * 2)

#eval mapExample [1, 2, 3, 4]  -- 输出: [2, 4, 6, 8]
```

### 4.2 过滤 (Filter)

```lean
def filterExample (lst : List Int) : List Int :=
  lst.filter (· % 2 == 0)

#eval filterExample [1, 2, 3, 4, 5, 6]  -- 输出: [2, 4, 6]
```

### 4.3 折叠 (Fold)

```lean
def sumList (lst : List Int) : Int :=
  lst.foldl (· + ·) 0

#eval sumList [1, 2, 3, 4, 5]  -- 输出: 15
```

## 5. 比较操作

Lean4 使用 `Ord` 类型类进行比较操作。以下是一些例子：

```lean
#eval compare 5 10        -- 结果: Ordering.lt
#eval compare "a" "b"     -- 结果: Ordering.lt
#eval compare [1, 2] [1, 3] -- 结果: Ordering.lt
```


## 6. 排序和 Ord 比较

在 Lean4 中，排序和比较操作通常依赖于 `Ord` 类型类。`Ord` 提供了一种统一的方式来比较不同类型的值。

### 6.1 Ord 类型类

`Ord` 类型类定义了以下主要方法：

- `compare : α → α → Ordering`：比较两个值，返回 `Ordering.lt`（小于）、`Ordering.eq`（等于）或 `Ordering.gt`（大于）。
- `(<) : α → α → Bool`：小于比较。
- `(≤) : α → α → Bool`：小于等于比较。
- `(>) : α → α → Bool`：大于比较。
- `(≥) : α → α → Bool`：大于等于比较。

### 6.2 基本类型的比较

```lean
#eval compare 5 10        -- 结果: Ordering.lt
#eval compare 3.14 3.14   -- 结果: Ordering.eq
#eval compare 'a' 'b'     -- 结果: Ordering.lt
#eval compare "hello" "world"  -- 结果: Ordering.lt
#eval compare true false  -- 结果: Ordering.gt
```

### 6.3 复合类型的比较

对于列表、数组等复合类型，比较操作通常是逐元素进行的：

```lean
#eval compare [1, 2, 3] [1, 2, 4]  -- 结果: Ordering.lt
#eval compare #[5, 6] #[5, 6, 7]   -- 结果: Ordering.lt
```

### 6.4 自定义类型的比较

对于自定义类型，我们可以实现 `Ord` 实例来定义比较行为：

```lean
structure Person where
  name : String
  age : Nat
deriving Repr

instance : Ord Person where
  compare p1 p2 :=
    match compare p1.age p2.age with
    | .eq => compare p1.name p2.name
    | ord => ord

def alice : Person := { name := "Alice", age := 30 }
def bob : Person := { name := "Bob", age := 25 }

#eval compare alice bob  -- 结果: Ordering.gt
```

### 6.5 排序操作

Lean4 提供了几种排序方法，它们都依赖于 `Ord` 类型类：

#### 6.5.1 列表排序

```lean
def sortList (lst : List Int) : List Int :=
  lst.qsort (·<=·)

#eval sortList [3, 1, 4, 1, 5, 9, 2, 6]  -- 输出: [1, 1, 2, 3, 4, 5, 6, 9]
```

#### 6.5.2 数组排序

```lean
def sortArray (arr : Array Int) : Array Int :=
  arr.qsort (·<=·)

#eval sortArray #[3, 1, 4, 1, 5, 9, 2, 6]  -- 输出: #[1, 1, 2, 3, 4, 5, 6, 9]
```

#### 6.5.3 自定义比较函数

我们可以提供自定义的比较函数来改变排序行为：

```lean
def sortDescending (lst : List Int) : List Int :=
  lst.qsort (·>=·)

#eval sortDescending [3, 1, 4, 1, 5, 9, 2, 6]  -- 输出: [9, 6, 5, 4, 3, 2, 1, 1]
```

### 6.6 高级排序示例

#### 6.6.1 按多个条件排序

```lean
def persons : List Person := [
  { name := "Alice", age := 30 },
  { name := "Bob", age := 25 },
  { name := "Charlie", age := 30 }
]

def sortPersons (ps : List Person) : List Person :=
  ps.qsort (fun p1 p2 => 
    match compare p1.age p2.age with
    | .eq => p1.name <= p2.name
    | .lt => true
    | .gt => false
  )

#eval sortPersons persons
-- 输出: [{name := "Bob", age := 25}, {name := "Alice", age := 30}, {name := "Charlie", age := 30}]
```

#### 6.6.2 稳定排序

Lean4 的 `qsort` 不保证稳定性。如果需要稳定排序，可以使用 `mergeSortBy`：

```lean
def stableSortPersons (ps : List Person) : List Person :=
  ps.mergeSortBy (fun p1 p2 => p1.age <= p2.age)

#eval stableSortPersons persons
```

### 6.7 性能考虑

- `qsort` 通常比 `mergeSortBy` 更快，但不稳定。
- 对于小列表，插入排序可能更高效。
- 对于大型数据集，考虑使用数组而不是列表，因为数组的随机访问更快。

## 总结

Lean4 中的排序和比较操作依赖于 `Ord` 类型类，这提供了一种统一和灵活的方式来处理不同类型的值。通过实现 `Ord` 实例，我们可以为自定义类型定义比较行为，从而支持排序操作。Lean4 提供了多种排序方法，如 `qsort` 和 `mergeSortBy`，可以根据需要选择合适的方法。理解这些概念对于高效处理和组织数据结构至关重要，尤其是在处理复杂的数据类型和大规模数据集时。