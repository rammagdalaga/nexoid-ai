# data_structures/advanced.py
import heapq


# ── Hash Map ────────────────────────────────

class HashMap:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.size     = 0
        self.buckets  = [[] for _ in range(capacity)]

    def _hash(self, key):
        return hash(key) % self.capacity

    def put(self, key, value):
        idx    = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self.size += 1
        if self.size / self.capacity > 0.75:
            self._resize()

    def get(self, key, default=None):
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        return default

    def remove(self, key):
        idx    = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return True
        return False

    def _resize(self):
        old_buckets   = self.buckets
        self.capacity *= 2
        self.buckets   = [[] for _ in range(self.capacity)]
        self.size      = 0
        for bucket in old_buckets:
            for k, v in bucket:
                self.put(k, v)

    def keys(self):
        return [k for bucket in self.buckets for k, _ in bucket]

    def values(self):
        return [v for bucket in self.buckets for _, v in bucket]

    def items(self):
        return [(k, v) for bucket in self.buckets for k, v in bucket]

    def __contains__(self, key):
        return self.get(key) is not None

    def __len__(self):
        return self.size


# ── Trie ────────────────────────────────────

class TrieNode:
    def __init__(self):
        self.children   = {}
        self.is_end     = False
        self.word_count = 0


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.word_count += 1
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

    def words_with_prefix(self, prefix: str):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        results = []
        self._collect(node, prefix, results)
        return results

    def _collect(self, node, path, results):
        if node.is_end:
            results.append(path)
        for ch, child in node.children.items():
            self._collect(child, path + ch, results)

    def delete(self, word: str) -> bool:
        return self._delete(self.root, word, 0)

    def _delete(self, node, word, depth):
        if depth == len(word):
            if not node.is_end:
                return False
            node.is_end = False
            return len(node.children) == 0
        ch = word[depth]
        if ch not in node.children:
            return False
        should_delete = self._delete(node.children[ch], word, depth + 1)
        if should_delete:
            del node.children[ch]
            return not node.is_end and len(node.children) == 0
        return False


# ── Min/Max Heap ────────────────────────────

class MinHeap:
    def __init__(self):
        self._data = []

    def push(self, val):
        heapq.heappush(self._data, val)

    def pop(self):
        if not self._data:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._data)

    def peek(self):
        if not self._data:
            raise IndexError("peek from empty heap")
        return self._data[0]

    def size(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0


class MaxHeap:
    def __init__(self):
        self._data = []

    def push(self, val):
        heapq.heappush(self._data, -val)

    def pop(self):
        if not self._data:
            raise IndexError("pop from empty heap")
        return -heapq.heappop(self._data)

    def peek(self):
        if not self._data:
            raise IndexError("peek from empty heap")
        return -self._data[0]

    def size(self):
        return len(self._data)


class MedianFinder:
    """Finds running median using two heaps."""
    def __init__(self):
        self.low  = MaxHeap()   # lower half
        self.high = MinHeap()   # upper half

    def add(self, num: float):
        self.low.push(num)
        self.high.push(self.low.pop())
        if self.high.size() > self.low.size():
            self.low.push(self.high.pop())

    def median(self) -> float:
        if self.low.size() > self.high.size():
            return float(self.low.peek())
        return (self.low.peek() + self.high.peek()) / 2.0