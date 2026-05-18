# algorithms/problems.py
from collections import defaultdict, Counter
from typing import List, Optional


# ── String Problems ─────────────────────────

def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {}
    for i, n in enumerate(nums):
        diff = target - n
        if diff in seen:
            return [seen[diff], i]
        seen[n] = i
    return []


def valid_parentheses(s: str) -> bool:
    stack = []
    pairs = {")": "(", "}": "{", "]": "["}
    for ch in s:
        if ch in "({[":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return len(stack) == 0


def longest_substring_no_repeat(s: str) -> int:
    char_idx = {}
    max_len = start = 0
    for i, ch in enumerate(s):
        if ch in char_idx and char_idx[ch] >= start:
            start = char_idx[ch] + 1
        char_idx[ch] = i
        max_len = max(max_len, i - start + 1)
    return max_len


def group_anagrams(words: List[str]) -> List[List[str]]:
    groups = defaultdict(list)
    for word in words:
        key = tuple(sorted(word))
        groups[key].append(word)
    return list(groups.values())


def is_anagram(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)


def reverse_string(s: List[str]) -> None:
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left  += 1
        right -= 1


def first_unique_char(s: str) -> int:
    count = Counter(s)
    for i, ch in enumerate(s):
        if count[ch] == 1:
            return i
    return -1


def roman_to_int(s: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50,
              "C": 100, "D": 500, "M": 1000}
    result = 0
    for i in range(len(s)):
        if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
            result -= values[s[i]]
        else:
            result += values[s[i]]
    return result


def is_valid_palindrome(s: str) -> bool:
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


def count_and_say(n: int) -> str:
    result = "1"
    for _ in range(n - 1):
        new_result = ""
        i = 0
        while i < len(result):
            ch    = result[i]
            count = 1
            while i + count < len(result) and result[i + count] == ch:
                count += 1
            new_result += str(count) + ch
            i += count
        result = new_result
    return result


# ── Array Problems ──────────────────────────

def max_profit(prices: List[int]) -> int:
    if not prices:
        return 0
    min_price = float("inf")
    max_prof  = 0
    for price in prices:
        min_price = min(min_price, price)
        max_prof  = max(max_prof, price - min_price)
    return max_prof


def contains_duplicate(nums: List[int]) -> bool:
    return len(nums) != len(set(nums))


def product_except_self(nums: List[int]) -> List[int]:
    n      = len(nums)
    result = [1] * n
    left   = 1
    for i in range(n):
        result[i] = left
        left     *= nums[i]
    right = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right
        right     *= nums[i]
    return result


def find_missing_number(nums: List[int]) -> int:
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)


def move_zeroes(nums: List[int]) -> None:
    insert = 0
    for num in nums:
        if num != 0:
            nums[insert] = num
            insert += 1
    while insert < len(nums):
        nums[insert] = 0
        insert += 1


def sorted_squares(nums: List[int]) -> List[int]:
    left, right = 0, len(nums) - 1
    result = []
    while left <= right:
        l2, r2 = nums[left] ** 2, nums[right] ** 2
        if l2 > r2:
            result.append(l2)
            left += 1
        else:
            result.append(r2)
            right -= 1
    return result[::-1]


def three_sum(nums: List[int]) -> List[List[int]]:
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left  += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result


def trap_rain_water(height: List[int]) -> int:
    left, right      = 0, len(height) - 1
    left_max = right_max = 0
    water    = 0
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
    return water