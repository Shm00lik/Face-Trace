class KalmanFilter:
    def __init__(self, k=5) -> None:
        self.k = k
        self.data: list[float] = []

    def update(self, *new_data: float) -> None:
        self.data.extend(new_data)

        if len(self.data) > self.k:
            self.data = self.data[-self.k :]

    def get(self) -> float:
        return sum(self.data) / len(self.data) if self.data else 0.0
