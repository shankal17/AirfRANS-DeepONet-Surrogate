"""Experiment configuration: one dataclass per pipeline stage"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

STAGES = ("download", "extract", "validate", "sweep", "train", "evaluate", "plot")
TASKS = ("full", "scarce", "reynolds", "aoa") #AirRANS dataset splits


@dataclass
class Paths:
    """Input and output locations shared by all stages.

    Attributes:
        raw: Download root (Dataset/ inside).
        data: Extracted .npz files and cases.csv.
        runs: Directory to save model weights and evaluation figures
        validation_figs: Directory for validation figures.
        eval_figs: Directory for figures that compare runs (fig_extrapolation.png).
        results: Directory for metrics.json.
    """

    raw: str = "data/raw"
    data: str = "data/surface"
    runs: str = "runs"
    validation_figs: str = "figures/validation"
    eval_figs: str = "figures/eval"
    results: str = "results"

    def run_figs(self, task: str) -> str:
        """Figure directory of one trained run

        Args:
            task: AirfRANS task of the run.

        Returns:
            <runs>/<task>/figures.
        """

        return os.path.join(self.runs, task, "figures")


@dataclass
class ExtractConfig:
    """Settings for extracting surface data from the AirfRANS simulations.

    Attributes:
        workers: Num worker processes; defaults to one fewer than the CPU count.
        limit: Process only the first n cases (e.g. 10 for a smoke test); None = all.
    """

    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))
    limit: int | None = None


@dataclass
class ValidateConfig:
    """Settings for the extracted-data checks.

    Attributes:
        tolerance: Largest allowed |dCl| and relative |dCd| between our integrated forces and AirfRAMS
    """

    tolerance: float = 1e-6


@dataclass
class TrainConfig:
    """Model architecture, optimization and loss settings for the train stage.

    Attributes:
        hidden_layer_width: Hidden hidden_layer_width of the branch and trunk MLPs.
        num_hidden_layers: Num hidden layers in each MLP.
        n_basis: Number of basis functions per output channel.
        n_fourier: Number of fourier frequencies of the chord coordinate fed to the trunk.
        epochs: Number of training epochs.
        batch_size: Batch size
        learning_rate: Learning rate.
        weight_decay: AdamW weight decay.
        force_weight: Final weight of the integrated Cl/Cd loss term.
        force_start: Fraction of epochs trained on surface fields only, before the force ramp.
        length_weight: Blend of per-point (0) and per-arc-length (1) loss weighting.
        cf_tangential: Predict [Cp, Cf_t] instead of [Cp, Cf_x, Cf_y].
        trunk_aoa: Give the trunk the surface orientation relative to the freestream.
        model_selection_metric: Validation metric that picks the best checkpoint, one of `MODEL_SELECTION_METRICS`.
        tasks: AirfRANS tasks to train, one model each; each must be in `TASKS`.
        seed: Sets initialisation and batch order.
        task_overrides: Per-task settings that replace the ones above for that task's model, e.g. {"scarce": {"batch_size": 8}}. Keys are tasks, inner keys names in `HYPERPARAMETERS`.

    Raises:
        ValueError: If a task is not in `TASKS`, `model_selection_metric` is not in `MODEL_SELECTION_METRICS`, or `task_overrides` names an unknown task or setting.
    """

    # model
    hidden_layer_width: int = 256
    num_hidden_layers: int = 4
    n_basis: int = 128
    n_fourier: int = 16

    # optimization
    epochs: int = 300
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    force_weight: float = 0.03
    force_start: float = 0.3
    length_weight: float = 0.5
    cf_tangential: bool = True
    trunk_aoa: bool = True
    model_selection_metric: str = "case_rmse"
    tasks: tuple[str, ...] = ("full",)
    seed: int = 0
    task_overrides: dict[str, dict] = field(default_factory=dict)

    HYPERPARAMETERS = ("hidden_layer_width", "num_hidden_layers", "n_basis", "n_fourier", "epochs", "batch_size", "learning_rate", "weight_decay", "force_weight", "force_start", "length_weight", "cf_tangential", "trunk_aoa", "model_selection_metric")
    MODEL_SELECTION_METRICS = ("case_rmse", "loss")

    def __post_init__(self):
        """Coerce `tasks` to a tuple and validate `tasks` and `model_selection_metric`."""

        self.tasks = tuple(self.tasks)
        invalid_tasks = [task for task in self.tasks if task not in TASKS]
        if invalid_tasks:
            raise ValueError(f"Unknown task(s) {invalid_tasks}; choose from {TASKS}")
        
        if self.model_selection_metric not in self.MODEL_SELECTION_METRICS:
            raise ValueError(f"Unknown model_selection_metric {self.model_selection_metric!r}; choose from {self.MODEL_SELECTION_METRICS}")
        
        for task, overrides in self.task_overrides.items():
            if task not in TASKS:
                raise ValueError(f"Unknown task {task!r} in task_overrides; choose from {TASKS}")
            invalid_hyperparameter_overrides = [k for k in overrides if k not in self.HYPERPARAMETERS]

            if invalid_hyperparameter_overrides:
                raise ValueError(f"Unknown setting(s) {invalid_hyperparameter_overrides} in task_overrides[{task!r}]; choose from {self.HYPERPARAMETERS}")
            
            if overrides.get("model_selection_metric", self.model_selection_metric) not in self.MODEL_SELECTION_METRICS:
                raise ValueError(f"Unknown model_selection_metric in task_overrides[{task!r}]; choose from {self.MODEL_SELECTION_METRICS}")

    def hparams(self, task: str | None = None) -> dict:
        """The settings `train_model` takes and checkpoints store.

        Args:
            task: Apply `task_overrides[task]` over the shared settings; None = shared only.

        Returns:
            Mapping from each name in `HYPERPARAMETERS` to its value.
        """

        return {k: getattr(self, k) for k in self.HYPERPARAMETERS} | self.task_overrides.get(task, {})


@dataclass
class SweepConfig:
    """One model per value of one setting; every other setting comes from `Config.train`.

    The base settings are `Config.train.hparams(task)`, so a task's overrides apply.

    Attributes:
        task: AirfRANS task to train on.
        param: Setting to vary: a name in `TrainConfig.HYPERPARAMETERS`, or "seed" to measure how much the metrics move between seeds alone.
        values: Values of `param` to try, one model each.
        holdout: How validation is carved from the training split (see `data.split_names`): "random", or "edge" to hold out the extremes of Re ("reynolds") or AoA ("aoa")
        out: Directory for the sweep models

    Raises:
        ValueError: If `task` is not in `TASKS`, `param` is unknown, or `holdout` is not "random" or "edge".
    """

    task: str = "full"
    param: str = "force_weight"
    values: tuple = (0.0, 0.01, 0.03, 0.1)
    holdout: str = "random"
    out: str = "runs_sweep"

    def __post_init__(self):
        """Coerce `values` to a tuple and validate `task`, `param` and `holdout`."""

        self.values = tuple(self.values)
        if self.task not in TASKS:
            raise ValueError(f"Unknown task {self.task!r}; choose from {TASKS}")
        
        if self.param not in TrainConfig.HYPERPARAMETERS + ("seed",):
            raise ValueError(f"Unknown param {self.param!r}; choose 'seed' or one of {TrainConfig.HYPERPARAMETERS}")
        
        if self.holdout not in ("random", "edge"):
            raise ValueError(f"Unknown holdout {self.holdout!r}; choose 'random' or 'edge'")


@dataclass
class PlotConfig:
    """Surface-field figures (Cp, Cf) for every trained model (see pipeline/plot.py).

    Attributes:
        split: Cases to draw from, "test" or "val" of each run's task.
        cases: Case names to plot for every run; None = spread over AoA.
        n_gallery: Gallery panels, 4 per row.
        n_surface: Rows in the airfoil-outline figure.

    Raises:
        ValueError: If `split` is not "test" or "val".
    """

    split: str = "test"
    cases: tuple[str, ...] | None = None
    n_gallery: int = 12
    n_surface: int = 4

    def __post_init__(self):
        """Validate `split` and coerce `cases` to a tuple."""

        if self.split not in ("test", "val"):
            raise ValueError(f"Unknown split {self.split!r}; choose 'test' or 'val'")
        
        if self.cases is not None:
            self.cases = tuple(self.cases)


@dataclass
class Config:
    """A full experiment: its name, the stages to run and each stage's settings.

    Attributes:
        name: Experiment name.
        stages: Stages to run, a name or a tuple of names from `STAGES`
        paths: Input and output locations.
        extract: Extract stage settings.
        validate: Validate stage settings.
        train: Train stage settings.
        sweep: Sweep stage settings.
        plot: Plot stage settings.

    Raises:
        ValueError: If a stage is not in `STAGES`.
    """

    name: str
    stages: tuple[str, ...]
    paths: Paths = field(default_factory=Paths)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    validate: ValidateConfig = field(default_factory=ValidateConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    def __post_init__(self):
        """Coerce `stages` to a tuple and validate it."""

        if isinstance(self.stages, str):
            self.stages = (self.stages,)
        
        self.stages = tuple(self.stages)
        bad = [stage for stage in self.stages if stage not in STAGES]
        if bad:
            raise ValueError(f"Unknown stage(s) {bad}; choose from {STAGES}")

    def to_dict(self) -> dict:
        """Return the config as nested plain dicts.

        Returns:
            The `dataclasses.asdict` view of this config.
        """

        return asdict(self)
