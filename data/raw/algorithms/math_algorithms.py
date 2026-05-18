# algorithms/math_algorithms.py


# ── Matrix ──────────────────────────────────

class Matrix:
    def __init__(self, rows, cols, fill=0):
        self.rows = rows
        self.cols = cols
        self.data = [[fill] * cols for _ in range(rows)]

    @classmethod
    def from_list(cls, lst):
        m = cls(len(lst), len(lst[0]))
        m.data = [row[:] for row in lst]
        return m

    def __getitem__(self, pos):
        r, c = pos
        return self.data[r][c]

    def __setitem__(self, pos, val):
        r, c = pos
        self.data[r][c] = val

    def __add__(self, other):
        assert self.rows == other.rows and self.cols == other.cols
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = self[i, j] + other[i, j]
        return result

    def __mul__(self, other):
        assert self.cols == other.rows
        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i, j] += self[i, k] * other[k, j]
        return result

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j, i] = self[i, j]
        return result

    def __repr__(self):
        return "\n".join(str(row) for row in self.data)


def rotate_matrix_90(matrix):
    n = len(matrix)
    for i in range(n // 2):
        for j in range(i, n - i - 1):
            temp                           = matrix[i][j]
            matrix[i][j]                   = matrix[n - 1 - j][i]
            matrix[n - 1 - j][i]           = matrix[n - 1 - i][n - 1 - j]
            matrix[n - 1 - i][n - 1 - j]  = matrix[j][n - 1 - i]
            matrix[j][n - 1 - i]           = temp
    return matrix


def spiral_order(matrix):
    if not matrix:
        return []
    result              = []
    top, bottom         = 0, len(matrix) - 1
    left, right         = 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            result.append(matrix[top][c])
        top += 1
        for r in range(top, bottom + 1):
            result.append(matrix[r][right])
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                result.append(matrix[bottom][c])
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                result.append(matrix[r][left])
            left += 1
    return result


def set_matrix_zeroes(matrix):
    rows_zero = set()
    cols_zero = set()
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 0:
                rows_zero.add(i)
                cols_zero.add(j)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i in rows_zero or j in cols_zero:
                matrix[i][j] = 0
    return matrix


# ── Bit Manipulation ────────────────────────

def count_bits(n: int) -> int:
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def reverse_bits(n: int, bits: int = 32) -> int:
    result = 0
    for _ in range(bits):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


def single_number(nums) -> int:
    result = 0
    for n in nums:
        result ^= n
    return result


# ── Recursion ────────────────────────────────

def power(base: float, exp: int) -> float:
    if exp == 0:
        return 1
    if exp < 0:
        return 1 / power(base, -exp)
    if exp % 2 == 0:
        half = power(base, exp // 2)
        return half * half
    return base * power(base, exp - 1)


def permutations(arr):
    if len(arr) <= 1:
        return [arr[:]]
    result = []
    for i in range(len(arr)):
        arr[0], arr[i] = arr[i], arr[0]
        for perm in permutations(arr[1:]):
            result.append([arr[0]] + perm)
        arr[0], arr[i] = arr[i], arr[0]
    return result


def combinations(arr, k):
    if k == 0:
        return [[]]
    if not arr:
        return []
    with_first    = [[arr[0]] + rest for rest in combinations(arr[1:], k - 1)]
    without_first = combinations(arr[1:], k)
    return with_first + without_first


def subsets(arr):
    if not arr:
        return [[]]
    rest = subsets(arr[1:])
    return rest + [[arr[0]] + s for s in rest]