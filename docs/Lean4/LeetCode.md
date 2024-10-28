leetcode 9

```lean
def romanToInt (s : String) : Nat :=
  let romanValues : Char → Nat
    | 'I' => 1
    | 'V' => 5
    | 'X' => 10
    | 'L' => 50
    | 'C' => 100
    | 'D' => 500
    | 'M' => 1000
    | _ => 0

  let rec loop (chars : List Char) (total : Nat) (prevValue : Nat) : Nat :=
    match chars with
    | [] => total
    | c::rest =>
      let currentValue := romanValues c
      if currentValue >= prevValue then
        loop rest (total + currentValue) currentValue
      else
        loop rest (total - currentValue) currentValue

  loop (s.toList.reverse) 0 0

#eval romanToInt "III"       -- 应该返回 3
#eval romanToInt "IV"        -- 应该返回 4
#eval romanToInt "IX"        -- 应该返回 9
#eval romanToInt "LVIII"     -- 应该返回 58
#eval romanToInt "MCMXCIV"   -- 应该返回 1994
```
leetcode 14
```lean
partial def longestCommonPrefix (strs : List String) : String :=
  match strs with
  | [] => ""
  | first :: rest =>
    let rec findPrefix (pre : String) (strings : List String) : String :=
      match strings with
      | [] => pre
      | str :: rest' =>
        let newPre := pre.take (min pre.length str.length)
        if str.startsWith newPre then
          findPrefix newPre rest'
        else if newPre.length > 0 then
          findPrefix (newPre.dropRight 1) strings
        else
          ""
    
    findPrefix first rest

#eval longestCommonPrefix ["flower", "flow", "flight"]  -- 应该返回 "fl"
#eval longestCommonPrefix ["dog", "racecar", "car"]     -- 应该返回 ""
#eval longestCommonPrefix []                            -- 应该返回 ""
#eval longestCommonPrefix ["a"]                         -- 应该返回 "a"
```
leetcode 20
```lean
def isValid (s : String) : Bool :=
  let pair : Char → Option Char
    | '(' => some ')'
    | '[' => some ']'
    | '{' => some '}'
    | _ => none

  let rec check (stack : List Char) (chars : List Char) : Bool :=
    match chars with
    | [] => stack.isEmpty
    | c :: rest =>
      match pair c with
      | some _ => check (c :: stack) rest
      | none =>
        match stack with
        | [] => false
        | top :: stackRest =>
          if pair top == some c then
            check stackRest rest
          else
            false

  check [] s.toList

#eval isValid "()"        -- 应该返回 true
#eval isValid "()[]{}"    -- 应该返回 true
#eval isValid "(]"        -- 应该返回 false
#eval isValid "([)]"      -- 应该返回 false
#eval isValid "{[]}"      -- 应该返回 true
```
leetcode 26

```lean
partial def removeDuplicates (nums : Array Int) : Nat × Array Int :=
  if nums.isEmpty then
    (0, #[])
  else
    let rec loop (i : Nat) (uniqueCount : Nat) (result : Array Int) :=
      if i >= nums.size then
        (uniqueCount, result)
      else if i = 0 || nums[i]! ≠ nums[i-1]! then
        loop (i+1) (uniqueCount+1) (result.push nums[i]!)
      else
        loop (i+1) uniqueCount result
    loop 0 0 #[]

#eval removeDuplicates #[1, 1, 2]
#eval removeDuplicates #[0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
```

