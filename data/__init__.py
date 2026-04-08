from data.transforms import center_crop_arr, DatasetFactory
from data.web_dataset import WebDatasetDataset
from data.data_factory import WebDataLoader, OnlineFeatures
from data.wds_utils import (
    log_and_continue,
    pytorch_worker_info,
    is_multi_node_environment,
    split_data_by_node,
    get_dataset_size,
    WeightedRoundRobinSampler,
    StrictProportionalBatchSampler,
)


def get_dataset(name, **kwargs):
    if name == 'online_features':
        return OnlineFeatures(**kwargs)
    else:
        raise NotImplementedError(name)
