from kalman_filter import KalmanFilter


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):

        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Point(self.x / scalar, self.y / scalar)

    def __floordiv__(self, scalar):
        return Point(self.x // scalar, self.y // scalar)

    def __str__(self):
        return f"Point(x={self.x}, y={self.y})"


class PointKalmanFilter:
    def __init__(self, k=5) -> None:
        self.x_filter = KalmanFilter(k)
        self.y_filter = KalmanFilter(k)

    def update(self, *new_points: Point):
        self.x_filter.update(*[p.x for p in new_points])
        self.y_filter.update(*[p.y for p in new_points])

    def get(self) -> Point:
        return Point(self.x_filter.get(), self.y_filter.get())


class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_center(self) -> Point:
        return Point(self.x + self.w / 2, self.y + self.h / 2)

    def get_corners(self) -> list[Point]:
        return [
            Point(self.x, self.y),
            Point(self.x + self.w, self.y),
            Point(self.x + self.w, self.y + self.h),
            Point(self.x, self.y + self.h),
        ]

    def get_area(self) -> float:
        return self.w * self.h

    def get_aspect_ratio(self) -> float:
        return self.w / self.h if self.h != 0 else 0

    def __str__(self):
        return f"Rectangle(x={self.x}, y={self.y}, w={self.w}, h={self.h})"
