# 1. 各个处理环节的提示词整理




<details>
<summary>Lean-NL命题对学科归类</summary>

```
prompt_template = '''
我们试图重构一个由lean4形式化语句和自然语句所共同表述的数学命题的数学领域结构，你需要给我这个数学命题的层级和具体的路径
我们提供下面的一些例子
  假如对象被归入"Mathematics / Number Theory / Diophantine Equations / Higher-Degree Diophantine Equations / Quintic Diophantine Equations / Quintic Diophantine Equations with Rational Coefficients / Symmetry and Group Theory in Quintic Diophantine Equations with Rational Coefficients",
    他是一个6层结构，因为他从Mathematics的节点往下沿生了6层，
    样例输出：{{"math_depth": 6,
    "math_field_path":"Mathematics / Number Theory / Diophantine Equations / Higher-Degree Diophantine Equations / Quintic Diophantine Equations / Quintic Diophantine Equations with Rational Coefficients / Symmetry and Group Theory in Quintic Diophantine Equations with Rational Coefficients"}}
    假如对象被归入"Mathematics / Combinatorics / Graph Theory / Graph Decompositions / Star Decompositions"
    那么其深度为4
    样例输出：{{"math_depth": 4,
    "math_field_path":"Mathematics / Combinatorics / Graph Theory / Graph Decompositions / Star Decompositions"}}
####################################################
   现在请你忽略以上的示例的内容:
   {data_template}
   生成以下命题的分析
   lean4格式：{pos1}
   自然语言格式:{pos2}
'''

data_template='''
    {"math_depth": math_depth_int,
    "math_field_path":field_path_str}
'''
```
- pos1 input_lean_statement 
- pos2 input_natural_language_statement 

</details>


<details>
<summary>生成数学学科树</summary>

```
prompt = f'''As a mathematics expert, focus exclusively on generating precise and specific subfields within pure and applied mathematics for the field of "{node_name}". 
Consider the following context: {node_field_info}

Guidelines:
1. Provide 3-5 direct subfields that are strictly within mathematics.
2. Ensure each subfield is more specific and narrower than "{node_name}".
3. Avoid any cross-disciplinary fields or applications outside of mathematics.
4. Focus on established mathematical concepts, not speculative or emerging ideas.
5. If a subfield seems too broad, break it down further into more specific areas.

Respond strictly in the following JSON format:
{data_template}

Ensure each subfield name is concise yet descriptive, using standard mathematical terminology.'''

data_template = "{'child1': 'subfield_name1', 'child2': 'subfield_name2', ...}"
```

- node_name:当前数学领域节点名称
- node_field_info:当前数学领域的相关信息

</details>


<details>
<summary>自然语言翻译为Lean4</summary>

```
prompt_template02 = '''
Given the natural language math statement {pos1}, translate it into Lean 4 theorem syntax. Please follow this data template:

{data_template}

Only provide the theorem statement without the proof. Use the appropriate Unicode symbols for mathematical notation where applicable.

Examples:

1. Natural language: If $r$ is rational $(r \neq 0)$ and $x$ is irrational, prove that $r+x$ is irrational.
   {{"lean_statement": "theorem exercise_1_1a (x : ℝ) (y : ℚ) : (irrational x) -> irrational (x + y) :="}}

2. Natural language: Prove that there is no rational number whose square is $12$.
   {{"lean_statement": "theorem exercise_1_2 : ¬ ∃ (x : ℚ), (x ^ 2 = 12) :="}}

3. Natural language: Let $A$ be a nonempty set of real numbers which is bounded below. Let $-A$ be the set of all numbers $-x$, where $x \in A$. Prove that $\inf A=-\sup (-A)$.
   {{"lean_statement": "theorem exercise_1_5 (A minus_A : set ℝ) (hA : A.nonempty) (hA_bdd_below : bdd_below A) (hminus_A : minus_A = {{x | -x ∈ A}}) : Inf A = Sup minus_A :="}}

4. Natural language: If $z$ is a complex number, prove that there exists an $r\geq 0$ and a complex number $w$ with $| w | = 1$ such that $z = rw$.
   {{"lean_statement": "theorem exercise_1_11a (z : ℂ) : ∃ (r : ℝ) (w : ℂ), abs w = 1 ∧ z = r * w :="}}
################################################################
Now, please translate the following statement into Lean 4:

{pos1}
'''

data_template02='''
    {"lean_statement":translated_lean_statement_str_here}
'''
```

- pos1:input_natural_language_statement

</details>


<details>
<summary>Lean语句翻译为自然语言</summary>

```
prompt_template ='''
    Here is a Lean statement: {pos1}.
    I want you to translate it into natural language and output it in the following format:
    {data_template}
'''

data_template='''
    {"lean":lean_str_here,"natural_language":nl_translation_here}
'''
```


- pos1:input_lean_statement

</details>