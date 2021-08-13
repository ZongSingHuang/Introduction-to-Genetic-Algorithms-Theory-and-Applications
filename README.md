# Introduction-to-Genetic-Algorithms-Theory-and-Applications
Python version

https://www.udemy.com/course/geneticalgorithm/

1. 這堂課總共有4個專案，我只展示專案的代碼

2. 我並不是原封不動的將MATLAB code 轉成 Python code，所以在參照時須留意

3. 這是一門不錯的GA入門課，但在菁英策略部分有點凌亂

4. GA適合用來解決組合問題、二進制問題、離散問題；雖然GA號稱萬能演算法，但連續空間問題的求解精度真的很爛，從許多paper在跑CEC 2005時都沒有把GA做為比較對象這點就可以佐證

------------------------------------

Lesson_17 使用的測試函數為Sphere，但講師把問題從連續空間[-5.12, 5.12]改為離散空間{0, 1}；維度從常見的30，改成20。

Lesson_21 使用的測試函數為Sphere，但講師把問題從連續空間[-5.12, 5.12]改為連續空間[-100, 100]；維度從常見的30，改成2。

Lesson_24 使用的測試函數為Gear train，講師採用組合問題進行求解，也就是染色體採二進制編碼，在適應值計算時又將染色體轉換為十進制。

Lesson_25 我沒有寫
