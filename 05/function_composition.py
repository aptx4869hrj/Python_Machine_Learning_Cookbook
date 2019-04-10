import numpy as np
from functools import reduce

# 定义一个函数，将数组每个数加二
def add3(input_array):
    return map(lambda x:x+3, input_array)

# 定义第二个函数,将数组每个数乘方
def mul2(input_array):
    return map(lambda x:x*x, input_array)

# 定义第三个函数，将数组每个数减5
def sub5(input_array):
    return map(lambda x:x-5, input_array)

# 定义一个函数组合器，将这些函数作为输入参数，返回一个组合函数。这个组合函数基本上是输入函数按序执行的一个函数
def function_composer(*args):
    return reduce(lambda f, g: lambda x: f(g(x)), args)

if __name__=='__main__':
    arr = np.array([2,5,4,7])
    print("\nOperation: add3(mul2(sub5(arr)))")
    # 常规方法执行
    arr1 = add3(arr)
    arr2 = mul2(arr1)
    arr3 = sub5(arr2)
    print("\nOutput using the lengthy way:", list(arr3))

    # 用函数组合器实现
    function_composed = function_composer(sub5, mul2, add3)
    print("\nOutput using the function composer:", list(function_composed(arr)))

    # 利用上面方法进行单行实现
    print("\nOperation: sub5(add3(mul2(sub5(mul2(arr)))))\nOutput:",
          list(function_composer(mul2, sub5, mul2, add3, sub5)(arr)))