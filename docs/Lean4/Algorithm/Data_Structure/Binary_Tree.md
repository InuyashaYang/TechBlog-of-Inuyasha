```lean
import Std
-- 这就是说叶节点的类型是 BinaryTree alpha，节点的类型则是给定alpha BinaryTree alpha 和 BinaryTree alpha 之后得到的 BinaryTree alpha
inductive BinaryTree (α : Type)
  | leaf : BinaryTree α
  | node : α → BinaryTree α → BinaryTree α → BinaryTree α

-- 设置了一个命名空间，这就意味着我们在别的地方要靠open BinaryTree来启用这个库，我们也可以写完整缀
namespace BinaryTree

-- 空二叉树
def empty : BinaryTree α := leaf

-- 这里采用了递归的逻辑，lean4就是一门递归为主的语言
-- 如果当前节点是叶子，就将待插入的数据放在叶子上；如果当前是节点，就检查待插入值是否比节点值大，如果大就在右树中递归，如果小就在左树中递归
def insert [Ord α] (x : α) (t : BinaryTree α) : BinaryTree α :=
  match t with
  | leaf => node x leaf leaf
  | node y left right =>
    match compare x y with
    | .lt => node y (insert x left) right
    | .gt => node y left (insert x right)
    | .eq => t

-- 本质上就是遍历二叉树
def contains [Ord α] (x : α) (t : BinaryTree α) : Bool :=
  match t with
  | leaf => false
  | node y left right =>
    match compare x y with
    | .lt => contains x left
    | .gt => contains x right
    | .eq => true

-- 中缀表达化
def inorder (t : BinaryTree α) : List α :=
  match t with
  | leaf => []
  | node x left right => (inorder left) ++ [x] ++ (inorder right)

end BinaryTree
```