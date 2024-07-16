from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class DatasetConfig:
    """
    A class used to configure the dataset for the model.

    Attributes
    ----------
    data_dir : str
        Specifies the directory where the dataset files are stored.  **Important**: The dataset uses a lot of disk space, so make sure you have enough storage available.
    dataset_name : str
        The name assigned to the dataset.
    hf_path : str
        The path to the dataset stored on HuggingFace.
    hf_name : str
        The name of the dataset on HuggingFace.
    seed : int
        A seed value for ensuring reproducibility across runs.
    n_workers : int
        The number of worker processes used for data loading.
    val_split : float
        The proportion of the dataset reserved for validation.
    task : str
        Defines the type of task (e.g., 'multilabel' or 'multiclass').
    subset : int, optional
        A subset of the dataset to use. If None, the entire dataset is used.
    sampling_rate : int
        The sampling rate for audio data processing.
    class_weights_loss : bool, optional
        (Deprecated) Previously used for applying class weights in loss calculation.
    class_weights_sampler : bool, optional
        Indicates whether to use class weights in the sampler for handling imbalanced datasets.
    classlimit : int, optional
        The maximum number of samples per class. If None, all samples are used.
    eventlimit : int, optional
        Defines the maximum number of audio events processed per audio file, capping the quantity to ensure balance across files. If None, all events are processed.
    direct_fingerprint: int, optional
        Only works with PretrainDatamodule. Path to a saved preprocessed dataset path
    """
    data_dir: str = "/workspace/data_birdset"
    dataset_name: str = "esc50"
    hf_path: str = "ashraq/esc50"
    hf_name: str = ""
    seed: int = 42
    n_workers: int = 1
    val_split: float = 0.2
    n_classes: int = 21
    task: Literal["multiclass", "multilabel"] = "multilabel"
    subset: Optional[int] = None
    sampling_rate: int = 32_000
    class_weights_loss: Optional[bool] = None
    class_weights_sampler: Optional[bool] = None
    classlimit: Optional[int] = None
    eventlimit: Optional[int] = None
    direct_fingerprint: Optional[str] = None  # TODO only supported in PretrainDatamodule

