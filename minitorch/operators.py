"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    """
    Multiply two numbers together

    Args:
        x: float
        y: float

    Returns:
        x * y
    """
    return x * y


def id(x: float) -> float:
    "$f(x) = x$"
    """
    Returns the value it takes in

    Args:
        x: float

    Returns:
        x
    """
    return x


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    """
    Adds to numbers together

    Args:
        x: float
        y: float

    Returns:
        x + y
    """
    return x + y


def neg(x: float) -> float:
    "$f(x) = -x$"
    """
    Negates a number

    Args:
        x: float

    Returns:
        -x
    """
    return -x


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    """
    Checks if first arg is less than second args

    Args:
        x: float
        y: float

    Returns:
        x < y as a float, 1.0/0 represting true/false
    """
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    """
    Checks if first arg is equal to the second arg

    Args:
        x: float
        y: float

    Returns:
        x == y as 1.0/0.0 representing true/false
    """
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    """
    Determines max of two values

    Args:
        x: float
        y: float

    Returns:
        x if x > y else y
    """
    if x > y:
        return x
    else:
        return y


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    """
    Checks almost equal of two floats

    Args:
        x: float
        y: float

    Returns:
        |x - y| < 1e-2
    """
    if abs(x - y) < 1e-2:
        return 1.0
    else:
        return 0.0


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =
    \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.

    Calculates sigmoid(x)

    Args:
        x: float

    Returns:
        sigmoid(x)
    """
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        return exp(x) / (1.0 + exp(x))


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)

    Calculates relu(x)

    Args:
        x: float

    Returns:
        relu(x)
    """
    if x > 0.0:
        return x
    else:
        return 0.0


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    """
    Calculates log(x)

    Args:
        x: float

    Returns:
        log(x)
    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    """
    Calculates exp(x)

    Args:
        x: float

    Returns:
        exp(x)
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    """
    Calculates the derivative of log scaled by d

    Args:
        x: float
        d: float

    Returns:
        f(x) = log(x)

        log_back(x, d) = d * f'(x)
    """
    return d * inv(x)


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    """
    Calculates 1/x

    Args:
        x: float

    Returns:
        1/x
    """
    return 1 / x


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    """
    Calculates derivative of inverse scaled by d

    Args:
        x: float
        d: float

    Returns:
        f(x) = 1/x

        inv_back(x, d) = d * f'(x)

    """
    return d * inv(x) ** 2


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    """
    Calculates derivative of relu(x) scaled by d

    Args:
        x: float
        d: float

    Returns:
        f(x) = relu(x)

        relu_back(x, d) = d * f'(x)
    """
    if x < 0:
        return 0.0
    else:
        return d


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and
         returns a new list
    """

    def apply(ls: Iterable[float]):
        new_ls = []
        for e in ls:
            e_prime = fn(e)
            new_ls.append(e_prime)
        return new_ls

    return apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    """
    Negates every element in a list

    Args:
        ls: a list of floats
    Returns:
        new_ls with new_ls[i] = -ls[i]
    """
    neg_map = map(neg)
    return neg_map(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`,
         produce a new list by applying fn(x, y) on each pair of elements.

    """

    def apply(ls1, ls2):
        assert len(ls1) == len(
            ls2
        ), f"Need to be same size, \
                                    {len(ls1)} != {len(ls2)}"
        new_ls = []

        for e1, e2 in zip(ls1, ls2):
            new_e = fn(e1, e2)
            new_ls.append(new_e)

        return new_ls

    return apply


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    """
    Element-wise add for two lists of integers

    Args:
        ls1: list of integers
        ls2: list of integers

    Returns:
        New list, ls, where ls[i] = ls1[i] + ls2[i]
    """
    list_add_map = zipWith(add)
    return list_add_map(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
        $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """

    def apply(ls):
        res = start
        for e in ls:
            res = fn(e, res)

        return res

    return apply


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    """
    Sum of list of floats

    Args:
        ls: list of floats

    Returns:
        A single value denoting the sum of ls
    """
    sum_fn = reduce(add, start=0)
    return sum_fn(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    """
    Calculates the product of a list of numbers

    Args:
        ls: list of numbers

    Returns:
        Single value representing product of all elements in ls
    """
    prod_fn = reduce(mul, start=1)
    return prod_fn(ls)
