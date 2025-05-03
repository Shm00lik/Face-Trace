from kalman_filter import KalmanFilter, KalmanFilterConstants


class Point:
    def __init__(self, x: int | float, y: int | float):
        self.x = int(x)
        self.y = int(y)

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

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
    def __init__(self, constants: KalmanFilterConstants) -> None:
        self.constants = constants
        self.x_filter = KalmanFilter(self.constants)
        self.y_filter = KalmanFilter(self.constants)

    def update(self, *new_points: Point):
        self.x_filter.update(*[p.x for p in new_points])
        self.y_filter.update(*[p.y for p in new_points])

    def get(self) -> Point:
        return Point(self.x_filter.get(), self.y_filter.get())


class Rectangle:
    class RectangleCorners:
        def __init__(self, x: float, y: float, width: float, height: float):
            self.top_left = Point(x, y)
            self.top_right = Point(x + width, y)
            self.bottom_right = Point(x + width, y + height)
            self.bottom_left = Point(x, y + height)

    def __init__(self, x: float, y: float, width: float, height: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_center(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    def get_corners(self) -> "Rectangle.RectangleCorners":
        return Rectangle.RectangleCorners(self.x, self.y, self.width, self.height)

    def get_area(self) -> float:
        return self.width * self.height

    def get_aspect_ratio(self) -> float:
        return self.width / self.height if self.height != 0 else 0

    def __str__(self):
        return f"Rectangle(x={self.x}, y={self.y}, w={self.width}, h={self.height}, c={self.get_center()})"


class RectangleKalmanFilter:
    def __init__(self, constants: KalmanFilterConstants) -> None:
        self.constants = constants
        self.center_filter = PointKalmanFilter(self.constants)
        self.width_filter = KalmanFilter(self.constants)
        self.height_filter = KalmanFilter(self.constants)

    def update(self, *new_rects: Rectangle):
        self.center_filter.update(*[p.get_center() for p in new_rects])
        self.width_filter.update(*[r.width for r in new_rects])
        self.height_filter.update(*[r.height for r in new_rects])

    def get(self) -> Rectangle:
        center = self.center_filter.get()
        width = self.width_filter.get()
        height = self.height_filter.get()

        return Rectangle(center.x - width / 2, center.y - height / 2, width, height)


class Resolution:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.aspet_ratio = self.width / self.height

    def as_tuple(self):
        return (self.width, self.height)
