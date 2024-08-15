from abc import ABC
from typing import Optional, Tuple

import numpy as np

from .legateboost import EvalResult, LBBase
from .metrics import metrics


class TrainingCallback(ABC):
    """Interface for training callback."""

    def __init__(self) -> None:
        pass

    def before_training(self, model: LBBase) -> None:
        """Run before training starts."""
        pass

    def after_training(self, model: LBBase) -> None:
        """Run after training is finished."""
        pass

    def before_iteration(
        self, model: LBBase, epoch: int, evals_result: EvalResult
    ) -> bool:
        """Run before each iteration.

        Returns True when training should stop. See
        :py:meth:`after_iteration` for details.
        """
        return False

    def after_iteration(
        self, model: LBBase, epoch: int, evals_result: EvalResult
    ) -> bool:
        """Run after each iteration.  Returns `True` when training should stop.

        Parameters
        ----------

        model :
            Either a :py:class:`~xgboost.Booster` object or a CVPack if the cv function
            in xgboost is being used.
        epoch :
            The current training iteration.
        evals_result :
            A dictionary containing the evaluation history:

            .. code-block:: python

                {"data_name": {"metric_name": [0.5, ...]}}
        """
        return False


class EarlyStopping(TrainingCallback):
    """Callback for early stopping during training. The last evaluation dataset
    is used for early stopping.

    Args:
        rounds (int): The number of rounds to wait for improvement before stopping.
        metric_name (Optional[str]): The name of the metric to monitor for improvement.
            If not provided, the last metric in the evaluation result will be used.
        min_delta (float): The minimum change in the monitored
            metric to be considered as improvement.
        prune_model (bool): Whether to prune the model after early stopping. If True,
            the model will be pruned to the best iteration.

    Raises:
        ValueError: If `min_delta` is less than 0.
    """

    def __init__(
        self,
        rounds: int,
        metric_name: Optional[str] = None,
        min_delta: float = 0.0,
        prune_model: bool = True,
        verbose: bool = False,
    ) -> None:
        self.metric_name = metric_name
        self.rounds = rounds
        self._min_delta = min_delta
        self.prune_model = prune_model
        if self._min_delta < 0:
            raise ValueError("min_delta must be greater or equal to 0.")

        self.current_rounds: int = 0
        self.best_score: Optional[Tuple[int, float]] = None
        self.verbose = verbose
        super().__init__()

    def before_training(self, model: LBBase) -> None:
        self.current_rounds = 0
        self.best_score = None

    def _update_rounds(
        self, score: float, metric: str, model: LBBase, epoch: int
    ) -> bool:
        def maximize(new: float, best: float) -> bool:
            """New score should be greater than the old one."""
            return np.greater(new - self._min_delta, best)

        def minimize(new: float, best: float) -> bool:
            """New score should be lesser than the old one."""
            return np.greater(best - self._min_delta, new)

        if metrics[metric].minimize():
            improve_op = minimize
        else:
            improve_op = maximize

        if self.best_score is None:  # First round
            self.best_score = (epoch, score)
            self.current_rounds = 0
        elif not improve_op(score, self.best_score[1]):
            # Not improved
            self.current_rounds += 1
        else:
            # Improved
            self.best_score = (epoch, score)
            self.current_rounds = 0  # reset

        if self.current_rounds >= self.rounds:
            # Should stop
            return True
        return False

    def after_iteration(
        self, model: LBBase, epoch: int, evals_result: EvalResult
    ) -> bool:
        msg = "Must have at least 1 validation dataset for early stopping."
        if len(evals_result.keys()) <= 1:
            raise ValueError(msg)

        # use last eval set
        data_log = list(evals_result.values())[-1]

        # Get metric name
        if self.metric_name:
            metric_name = self.metric_name
        else:
            # Use last metric by default.
            metric_name = list(data_log.keys())[-1]
        if metric_name not in data_log:
            raise ValueError(f"No metric named: {metric_name}")

        # The latest score
        score = data_log[metric_name][-1]
        return self._update_rounds(score, metric_name, model, epoch)

    def after_training(self, model: LBBase) -> None:
        if self.verbose and self.best_score is not None:
            print(
                f"Early stopping at round {self.best_score[0]}"
                + f" with score {self.best_score[1]}"
            )

        if self.prune_model and self.best_score is not None:
            model.models_ = model.models_[: self.best_score[0] + 1]
