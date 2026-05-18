# oop/classes.py
import math
from abc import ABC, abstractmethod


# ── Shapes ──────────────────────────────────

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(area={self.area():.2f})"


class Circle(Shape):
    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("Radius must be positive")
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def perimeter(self):
        return 2 * math.pi * self.radius


class Rectangle(Shape):
    def __init__(self, width: float, height: float):
        self.width  = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

    def is_square(self):
        return self.width == self.height


class Triangle(Shape):
    def __init__(self, a: float, b: float, c: float):
        if a + b <= c or a + c <= b or b + c <= a:
            raise ValueError("Invalid triangle sides")
        self.a, self.b, self.c = a, b, c

    def area(self):
        s = (self.a + self.b + self.c) / 2
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))

    def perimeter(self):
        return self.a + self.b + self.c


# ── Animals ─────────────────────────────────

class Animal(ABC):
    def __init__(self, name: str, age: int):
        self.name = name
        self.age  = age

    @abstractmethod
    def speak(self) -> str:
        pass

    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name}, age={self.age})"


class Dog(Animal):
    def __init__(self, name, age, breed):
        super().__init__(name, age)
        self.breed = breed

    def speak(self):
        return f"{self.name} says: Woof!"

    def fetch(self, item):
        return f"{self.name} fetches the {item}!"


class Cat(Animal):
    def __init__(self, name, age, indoor=True):
        super().__init__(name, age)
        self.indoor = indoor

    def speak(self):
        return f"{self.name} says: Meow!"

    def purr(self):
        return f"{self.name} purrs..."


class Bird(Animal):
    def __init__(self, name, age, can_fly=True):
        super().__init__(name, age)
        self.can_fly = can_fly

    def speak(self):
        return f"{self.name} says: Tweet!"

    def fly(self):
        if self.can_fly:
            return f"{self.name} flies away!"
        return f"{self.name} cannot fly."


# ── Vehicles ────────────────────────────────

class Vehicle:
    def __init__(self, make: str, model: str, year: int):
        self.make  = make
        self.model = model
        self.year  = year
        self.speed = 0

    def accelerate(self, amount: float):
        self.speed += amount
        return self.speed

    def brake(self, amount: float):
        self.speed = max(0, self.speed - amount)
        return self.speed

    def __repr__(self):
        return f"{self.year} {self.make} {self.model}"


class Car(Vehicle):
    def __init__(self, make, model, year, num_doors=4):
        super().__init__(make, model, year)
        self.num_doors = num_doors
        self.gear      = 1

    def shift_up(self):
        if self.gear < 6:
            self.gear += 1
        return self.gear

    def shift_down(self):
        if self.gear > 1:
            self.gear -= 1
        return self.gear


class ElectricCar(Car):
    def __init__(self, make, model, year, battery_capacity: float):
        super().__init__(make, model, year)
        self.battery_capacity = battery_capacity
        self.charge_level     = 100.0

    def charge(self, percent: float):
        self.charge_level = min(100.0, self.charge_level + percent)

    def drive(self, km: float):
        consumption = km * 0.2
        if consumption > self.charge_level:
            raise RuntimeError("Not enough charge!")
        self.charge_level -= consumption
        return self.charge_level


# ── Design Patterns ─────────────────────────

class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


class Observer(ABC):
    @abstractmethod
    def update(self, event: str, data):
        pass


class EventEmitter:
    def __init__(self):
        self._listeners = {}

    def on(self, event: str, listener: Observer):
        self._listeners.setdefault(event, []).append(listener)

    def emit(self, event: str, data=None):
        for listener in self._listeners.get(event, []):
            listener.update(event, data)