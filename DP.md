# Fibonacci
Recursion Approach
```python
def fib_recursion(n):
    if n in [0, 1]:
        return n
    return fib_recursion( n - 1) + fib_recursion(n - 2)
```
Top Down Approach - Memoization
```python
def fib_top_down(n):
    F = [-1] * (n + 1)
    F[0] = 0
    F[1] = 1

    def fib(n):
        if F[n] != -1:
            return F[n]
        F[n] = fib(n - 1) + fib(n - 2)
        return F[n]

    return fib(n)
```
Bottom Up Approach - Tabulation
```python
def fib_bottom_up(n):
    F = [0] * (n+1)
    F[1] = 1

    for i in range(2, n+1):
        F[i] = F[i - 1] + F[i - 2]
    return F[n]
```

# 0-1 Knapsack
Top Down Approach - Memoization
```python
w = [2, 3, 4, 5]
v = [3, 4, 5, 6]
n = 4
W = 5

memo = [[-1] * (W + 1) for _ in range(n + 1)]

def knapsack(i, j):
    if i == 0 or j == 0:
        return 0

    if memo[i][j] != -1:
        return memo[i][j]

    if w[i - 1] > j:
        memo[i][j] = knapsack(i - 1, j)
    else:
        memo[i][j] = max(
            v[i - 1] + knapsack(i - 1, j - w[i - 1]),  # Include the item
            knapsack(i - 1, j)                         # Exclude the item
        )

    return memo[i][j]
```
Bottom Up Approach - Tabulation
```python
w = [2, 3, 4, 5]
v = [3, 4, 5, 6]
n = 4
W = 5

def knapsack01(w, v, n, W):
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, W + 1):
            if w[i - 1] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(v[i - 1] + dp[i - 1][j - w[i - 1]], dp[i - 1][j])
    return dp[n][W]
```

