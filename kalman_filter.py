from dataclasses import dataclass


@dataclass
class KalmanFilterConstants:
    K: int = 5


class KalmanFilter:
    def __init__(self, constants: KalmanFilterConstants) -> None:
        self.constants = constants
        self.data: list[float] = []

    def update(self, *new_data: float) -> None:
        self.data.extend(new_data)

        if len(self.data) > self.constants.K:
            self.data = self.data[-self.constants.K :]

    def get(self) -> float:
        return sum(self.data) / len(self.data) if self.data else 0.0
