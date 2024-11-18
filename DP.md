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
Construct Solution
```python
def item_selected(dp, w, n, W):
    i = n
    k = W
    selected_items = []

    while i > 0 and k > 0:
        if dp[i][k] != dp[i - 1][k]:
            selected_items.append(i)
            k -= w[i]
        i -= 1

    return list(reversed(selected_items))
```
# Least Common Subsequence
Recursion Approach
```python
def lcs(str1, str2):
    if str1 == "" or str2 == "":
        return 0

    n = len(str1)
    m = len(str2)

    if str1[n - 1] == str2[m - 1]:
        return 1 + lcs(str1[:n - 1], str2[:m - 1])
    return max(
        lcs(str1[:n - 1], str2[:m]),
        lcs(str1[:n], str2[:m - 1])
    )
```
Top Down Approach - Memoization
```python
n = len(str1)
m = len(str2)

memo = [[-1] * (m + 1) for _ in range(n + 1)]


def lcs(str1, str2, i, j):
    if i == 0 or j == 0:
        return 0

    if memo[i][j] != -1:
        return memo[i][j]

    if str1[i - 1] == str2[j - 1]:
        memo[i][j] = 1 + lcs(str1, str2, i - 1, j - 1)
    else:
        memo[i][j] = max(
            lcs(str1, str2, i - 1, j),
            lcs(str1, str2, i, j - 1)
        )
    return memo[i][j]
```
Bottom Up Approach - Tabulation
```python
def lcs(str1, str2):
    n = len(str1)
    m = len(str2)
    memo = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                memo[i][j] = 1 + memo[i - 1][j - 1]
            else:
                memo[i][j] = max(
                    memo[i][j - 1],
                    memo[i - 1][j]
                )
    return memo[i][j]
```
Construct Solution
```python
def reconstruct_lcs(str1, str2, memo):
    lcs_result = []
    i, j = n, m
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs_result.append(str1[i - 1])
            i -= 1
            j -= 1
        elif memo[i - 1][j] > memo[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return ''.join(reversed(lcs_result))
```

# Rod Cutting

Top Down Approach - Memoization
```python
prices = [1, 5, 8, 9, 10, 17, 17, 20]
memo = [-1] * (len(prices) + 1)

def rod_cutting(i):
    if i == 0:
        return 0

    if memo[i] != -1:
        return memo[i]

    val = 0
    for j in range(1, i + 1):
        val = max(val, prices[j - 1] + rod_cutting(i - j))
    memo[i] = val
    return memo[i]
```
Bottom Up Approach - Tabulation
```python
def rod_cutting_bottom_up(prices):
    n = len(prices)
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        max_val = 0
        for j in range(1, i + 1):
            max_val = max(max_val, prices[j - 1] + dp[i - j])
        dp[i] = max_val

    return dp[n]
```
