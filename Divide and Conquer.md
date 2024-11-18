# Divide and Conquer

## Merge Sort 
T(n) = 2T(n / 2) + n --> O(n log n)
```python
def mergesort(A, p, r):
    if p < r:
        q = (p + r) // 2
        mergesort(A, p, q)
        mergesort(A, q + 1, r)
        merge(A, p, q, r)

def merge(A, p, q, r):
    nl = q - p + 1
    nr = r - q
    L = [0] * nl
    R = [0] * nr

    for i in range(nl):
        L[i] = A[p + i]

    L.append(math.inf)

    for j in range(nr):
        R[j] = A[q + j + 1]

    R.append(math.inf)

    i = 0
    j = 0
    for k in range(p, r + 1):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
```

## Quick Sort
Average Case: T(n) = 2T(n / 2) + n --> O(n log n)<br>
Worst Case: T(n) = T(n -1) + T(0) + n --> O(n<sup>2</sup>)
```python
def quicksort(A, p, r):
    if p < r:
        q = random_partition(A, p, r)
        quicksort(A, p, q - 1)
        quicksort(A, q + 1, r)

def random_partition(A, p, r):
    q = random.randint(p, r)
    A[q], A[r] = A[r], A[q] # swap q and r values

    x = A[r]
    i = p - 1
    for j in range(p, r):
        if A[j] <= x:
            i += 1
            A[i], A[j] = A[j], A[i]

    A[i + 1], A[r] = A[r], A[i + 1]
    return i + 1
```
### Random Select
Average Case: T(n) = T(n / 2) + n --> O(n)<br>
Worst Case: T(n) = T(n -1) + T(0) + n --> O(n<sup>2</sup>)
```python
def select(A, p, r, i):
    # check if i in the range
    if i < 1 or i > r - p + 1:
        return math.inf

    if p == r:
        return A[p]

    q = random_partition(A, p, r)
    k = q - p + 1

    if i == k:
        return A[q]

    if i < k:
        return select(A, p, q - 1, i)
    else:
        return select(A, q + 1, r, i - k)
```
### Median of Medians
Group of 3 --> T(n) = T(n / 3) + T(2n / 3) + n --> O(n log n)

Group of 5 --> T(n) = T(n / 5) + T(3n / 4) + n --> O(n)

Group of 9 --> T(n) = T(n / 7) + T(5n / 7) + n --> O(n)

```python
def median_of_medians(A, p, r, i):

    if r - p + 1 <= 5:
        A[p:r + 1] = sorted(A[p:r + 1])
        return A[p + i - 1]

    medians = []
    for j in range(p, r + 1, 5):
        sub_right = min(j + 4, r)
        sublist = sorted(A[j:sub_right + 1])
        medians.append(sublist[len(sublist) // 2])

    medians_length = len(medians)
    median_of_medians_value = median_of_medians(medians, 0, medians_length - 1, (medians_length // 2) + 1)

    pivot_index = A[p:r+1].index(median_of_medians_value)

    # partitioning
    A[pivot_index], A[r] = A[r], A[pivot_index]
    x = A[r]
    q = p - 1
    for j in range(p, r):
        if A[j] <= x:
            q += 1
            A[q], A[j] = A[j], A[q]
    q += 1
    A[q], A[r] = A[r], A[q]

    k = q - p + 1
    if i == k:
        return A[q]
    elif i < k:
        return median_of_medians(A, p, q - 1, i)
    else:
        return median_of_medians(A, q + 1, r, i - k)
```
## Binary Search
T(n) = T(n / 2) + 1  --> O(log n)

### Sorted Array
```python
def binary_search(nums, low, high, target):
    if low > high:
        return -1

    mid = (low + high) // 2
    if nums[mid] == target:
        return mid
    elif nums[mid] > target:
        return binary_search(nums, low, mid - 1, target)
    else:
        return binary_search(nums, mid + 1, high, target)
```
### Rotated Sorted Array
```python
def rotated_binary_search(arr, low, high, key):
    if low > high:
        return -1
    mid = (low + high) // 2
    if key == arr[mid]:
        return mid
    elif key < arr[mid]:
        if key >= arr[low] or arr[mid] < arr[high]:
            return rotated_binary_search(arr, low, mid - 1, key)
        else:
            return rotated_binary_search(arr, mid + 1, high, key)
    else:
        if key <= arr[high] or arr[low] < arr[mid]:
            return rotated_binary_search(arr, mid + 1, high, key)
        else:
            return rotated_binary_search(arr, low, mid - 1, key)

```
### Find A[i] = i in sorted array
```python
def search(A, low, high):
    if low > high:
        return -1
    
    mid = (low + high) // 2 
    
    if A[mid] == mid:
        return mid
    elif A[mid] > mid:
        return search(A, low, mid - 1)
    else:
        return search(A, mid + 1, high) 

```

### Find Peak Element
```python
def find_peak(arr, low, high):
    if low == high:
        return low

    if low == high - 1:
        return low if arr[low] > arr[high] else high

    mid = (low + high) // 2

    if arr[mid] > arr[mid + 1] and arr[mid] > arr[mid - 1]:
        return mid
    elif arr[mid] > arr[mid + 1]:
        return find_peak(arr, low, mid - 1)
    else:
        return find_peak(arr, mid + 1, high)
```
## Power
T(n) = T(n / 2) + 1  --> O(log n)
```python
def power(x, n):
    if n == 0:
        return 1
    elif n < 0:
        return 1 / power(x, -n)
    else:
        y = power(x, n // 2)
        if n % 2 == 0:
            return y * y
        else:
            return y * y * x
```

## Reverse 32 bit Integer
```python
def reverse(n, length):
    if length == 1:
        return n  # Base case: if the segment length is 1, return the number as is

    half_length = length // 2

    # Divide the integer into two halves: left and right
    n_left = n >> half_length  # Right shift to get the left half
    n_right = n & ((1 << half_length) - 1)  # Mask to get the right half

    # Recursively reverse the left and right halves
    reversed_left = reverse(n_left, half_length)
    reversed_right = reverse(n_right, half_length)

    # Combine the reversed left and right halves
    return (reversed_right << half_length) | reversed_left

```
## Count Inversions
```python
def mergesort_count(A, p, r):
    inversion_count = 0
    if p < r:
        q = (p + r) // 2
        inversion_count += mergesort_count(A, p, q)
        inversion_count += mergesort_count(A, q + 1, r)
        inversion_count += merge_count(A, p, q, r)
    return inversion_count

def merge_count(A, p, q, r):
    nl = q - p + 1
    nr = r - q
    L = [0] * nl
    R = [0] * nr

    for i in range(nl):
        L[i] = A[p + i]
    L.append(math.inf)

    for j in range(nr):
        R[j] = A[q + j + 1]
    R.append(math.inf)

    i = 0
    j = 0
    inversions = 0
    for k in range(p, r + 1):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1
            inversions += (nl - i)
    return inversions
```

## Count Reverse Pairs where <mark>arr[i] > 2 * arr[j] and i < j</mark>
```python
def reverse_pairs(arr, low, high):
    pairs_count = 0
    if low < high:
        mid = (low + high) // 2

        pairs_count += reverse_pairs(arr, low, mid)
        pairs_count += reverse_pairs(arr, mid + 1, high)
        pairs_count += count_and_merge(arr, low, mid, high)

    return pairs_count


def count_and_merge(A, p, q, r):
    nl = q - p + 1
    nr = r - q

    L = [0] * nl
    R = [0] * nr

    for i in range(nl):
        L[i] = A[p + i]

    for j in range(nr):
        R[j] = A[q + j + 1]

    i = 0
    j = 0
    count = 0

    while i < nl and j < nr:
        if L[i] > 2 * R[j]:
            count += nl - i
            j += 1
        else:
            i += 1

    L.append(math.inf)
    R.append(math.inf)
    
    i = 0
    j = 0
    for k in range(p, r + 1):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1

    return count
```

## Max Subarray
Naive Solution O(n<sup>2</sup>)
```python
def max_subarray(nums):
    maxSum = -math.inf
    for i in range(len(nums)):
        currentSum = nums[i]
        for j in range(i + 1, len(nums)):
            currentSum += nums[j]
            maxSum = max(currentSum, maxSum)
    return maxSum
```

Divide and Conquer O(n log n)
```python
def max_subarray(nums, p, r):
    if p == r:
        return nums[p]

    q = (p + r) // 2
    left_sum = max_subarray(nums, p, q)
    right_sum = max_subarray(nums, q + 1, r)
    cross_sum = max_cross_subarray(nums, p, q, r)

    return max(left_sum, right_sum, cross_sum)

def max_cross_subarray(nums, p, q, r):
    left_sum = -math.inf
    sum = 0
    for i in range(q, p - 1, -1):
        sum += nums[i]
        left_sum = max(left_sum, sum)

    right_sum = -math.inf
    sum = 0
    for i in range(q + 1, r + 1):
        sum += nums[i]
        right_sum = max(right_sum, sum)

    return left_sum + right_sum
```
Return the indecies of the subarray
```python
def max_subarray(nums, p, r):
    if p == r:
        return (p, r, nums[p]) 

    q = (p + r) // 2 
    left_start, left_end, left_sum = max_subarray(nums, p, q)
    right_start, right_end, right_sum = max_subarray(nums, q + 1, r)
    cross_start, cross_end, cross_sum = max_cross_subarray(nums, p, q, r)

    if left_sum >= right_sum and left_sum >= cross_sum:
        return (left_start, left_end, left_sum)
    elif right_sum >= left_sum and right_sum >= cross_sum:
        return (right_start, right_end, right_sum)
    else:
        return (cross_start, cross_end, cross_sum)

def max_cross_subarray(nums, p, q, r):
    left_sum = -math.inf
    sum = 0
    left_start = q
    for i in range(q, p - 1, -1):
        sum += nums[i]
        if sum > left_sum:
            left_sum = sum
            left_start = i

    right_sum = -math.inf
    sum = 0
    right_end = q + 1
    for i in range(q + 1, r + 1):
        sum += nums[i]
        if sum > right_sum:
            right_sum = sum
            right_end = i

    return (left_start, right_end, left_sum + right_sum)
```

## Integer Multiplication
T(n) = 3T(n / 2) + n  --> O(n<sup>1.58</sup>)
```python
def int_mul(x, y):
    if x < 10 or y < 10:
        return x * y

    n = math.ceil(max(math.log10(x), math.log10(y))) # returns max number of digits in x and y
    half = n // 2

    a = x // (10 ** half)
    b = x % (10 ** half)

    c = y // (10 ** half)
    d = y % (10 ** half)

    ac = int_mul(a, c)
    bd = int_mul(b, d)
    term = int_mul(a + b, c + d) - ac - bd

    # ac * 10^(half * 2) + term * 10^half + bd
    return int(ac * (10 ** (half * 2)) + term * (10 ** half) + bd)
```
