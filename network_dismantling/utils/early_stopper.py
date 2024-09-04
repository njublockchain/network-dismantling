from abc import ABC
import math


class EarlyStopper(ABC):
    def should_stop(self, validation_loss):
        pass


class SimpleEarlyStopper(EarlyStopper):
    def __init__(self):
        self.counter = 0
        self.min_validation_loss = float("inf")

    def should_stop(self, loss: float):
        if math.isnan(loss) or math.isinf(loss):
            return True

        if loss < self.min_validation_loss:
            self.min_validation_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= 2
