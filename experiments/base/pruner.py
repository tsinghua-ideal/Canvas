import math
import json


class Pruner:
    def __init__(self, epochs: int, theta: float = 0.8, min_limit: float = 0.05, milestone_file: str = ''):
        self.epochs = epochs
        self.theta = theta
        self.delta = (1 - self.theta) / self.epochs
        self.min_limit = min_limit
        if milestone_file:
            with open(milestone_file) as f:
                self.milestones = json.load(f)
            print(f'Milestones loaded: {self.milestones}')
        else:
            self.milestones = {}

        self.best_score = 0
        self.best_pattern = []
        self.current_pattern = []
        self.current_pruned_info = ''

    def prune_loss(self, v: float):
        if math.isnan(v):
            self.current_pruned_info = 'NaN occurs'
            return True
        return False

    def reset(self):
        self.current_pattern = []
        self.current_pruned_info = ''

    def update_epoch(self, score: float, current_score: float) -> bool:
        if score < self.min_limit or current_score < self.min_limit:
            self.current_pruned_info = f'The score is lower than the minimum limit ({self.min_limit})'
            return True
        current_epoch = len(self.current_pattern)
        self.current_pattern.append(score)
        if score >= self.best_score:
            self.best_score = score
            self.best_pattern = self.current_pattern
        current_theta = self.theta + current_epoch * self.delta
        to_compare = 0
        if len(self.best_pattern) != 0:
            to_compare = self.best_pattern[current_epoch] \
                if current_epoch < len(self.best_pattern) else self.best_pattern[-1]
        if score < to_compare * current_theta:
            self.current_pruned_info = f'The score reaches the pruning limit ' \
                                       f'(theta={current_theta}, max={self.best_pattern[current_epoch]})'
            return True

        # Check milestones
        current_epoch += 1
        current_epoch = str(current_epoch)
        if current_epoch in self.milestones:
            if score < self.milestones[current_epoch]:
                self.current_pruned_info = f'The score ({score}) does not reach ' \
                                           f'the milestone {current_epoch} ' \
                                           f'requirement ({self.milestones[current_epoch]})'
                return True
        return False
