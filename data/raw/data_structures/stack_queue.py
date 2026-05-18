# data_structures/stack_queue.py
from collections import deque


class Stack:
    def __init__(self):
        self._data = []

    def push(self, item):
        self._data.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def is_empty(self):
        return len(self._data) == 0

    def size(self):
        return len(self._data)

    def __repr__(self):
        return f"Stack({self._data})"


class Queue:
    def __init__(self):
        self._data = deque()

    def enqueue(self, item):
        self._data.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self._data.popleft()

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty queue")
        return self._data[0]

    def is_empty(self):
        return len(self._data) == 0

    def size(self):
        return len(self._data)

    def __repr__(self):
        return f"Queue({list(self._data)})"


class PriorityQueue:
    def __init__(self):
        self._heap = []

    def push(self, item, priority):
        import heapq
        heapq.heappush(self._heap, (priority, item))

    def pop(self):
        import heapq
        if self.is_empty():
            raise IndexError("pop from empty priority queue")
        return heapq.heappop(self._heap)[1]

    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty priority queue")
        return self._heap[0][1]

    def is_empty(self):
        return len(self._heap) == 0

    def size(self):
        return len(self._heap)


class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self._data = [None] * capacity
        self.front = self.rear = -1
        self._size = 0

    def enqueue(self, item):
        if self._size == self.capacity:
            raise OverflowError("circular queue is full")
        if self.front == -1:
            self.front = 0
        self.rear = (self.rear + 1) % self.capacity
        self._data[self.rear] = item
        self._size += 1

    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from empty circular queue")
        item = self._data[self.front]
        self._size -= 1
        if self._size == 0:
            self.front = self.rear = -1
        else:
            self.front = (self.front + 1) % self.capacity
        return item

    def is_empty(self):
        return self._size == 0

    def is_full(self):
        return self._size == self.capacity

    def size(self):
        return self._size