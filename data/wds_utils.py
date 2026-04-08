"""
WebDataset distributed utility functions, pipeline helper functions and sampler classes.
"""

from torch.utils.data import IterableDataset
import torch
import math
import random
import os
import logging

import braceexpand


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    if "No images in sample" in str(exn) or "Only one image in sample" in str(exn):
        return True
    
    if isinstance(exn, FileNotFoundError) or "FileNotFoundError" in str(type(exn)):
        if os.environ.get("RANK", "0") == "0":
            logging.warning(f"Handling webdataset FileNotFoundError: {exn}. Ignoring and continuing.")
        return True
    
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


# Distributed environment detection and shard allocation

def pytorch_worker_info(group=None):
    """Return node and worker info for PyTorch and some distributed environments."""
    rank = 0
    world_size = 1
    worker = 0
    num_workers = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        try:
            import torch.distributed
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = group or torch.distributed.group.WORLD
                rank = torch.distributed.get_rank(group=group)
                world_size = torch.distributed.get_world_size(group=group)
        except ModuleNotFoundError:
            pass
    if "WORKER" in os.environ and "NUM_WORKERS" in os.environ:
        worker = int(os.environ["WORKER"])
        num_workers = int(os.environ["NUM_WORKERS"])
    else:
        try:
            import torch.utils.data
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker = worker_info.id
                num_workers = worker_info.num_workers
        except ModuleNotFoundError:
            pass
    return rank, world_size, worker, num_workers


def is_multi_node_environment():
    """
    check if in a multi-process (world_size > 1) environment.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_world_size() > 1:
                return True
    except Exception:
        pass

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    nnodes = int(os.environ.get("NNODES", os.environ.get("SLURM_NNODES", "1")))
    if nnodes > 1:
        return True
    return world_size > 1


def split_data_by_node(urls, strategy="interleaved"):
    """split shards between nodes, even if the data is stored locally, it is recommended to use it to avoid duplicate training."""
    print('*'*80)
    print("split_data_by_node ing..................")
    gpus_per_node = torch.cuda.device_count()
    rank, world_size, worker, num_workers = pytorch_worker_info()
    print("rank: {}, world_size: {}, worker: {}, num_workers: {}, gpus_per_node: {}".format(
        rank, world_size, worker, num_workers, gpus_per_node))
    
    node_rank = rank // gpus_per_node
    node_world_size = world_size // gpus_per_node

    if len(urls) < node_world_size:
        print(f"Warning: Only {len(urls)} shards but {node_world_size} nodes. "
              f"All nodes will use all shards to avoid empty assignment.")
        print(f"Node {node_rank} has {len(urls)} URLs of {len(urls)} total.")
        print('*'*80)
        return urls

    if strategy == "chunk":
        urls_per_node = math.ceil(len(urls) / node_world_size)
        start_idx = node_rank * urls_per_node
        end_idx = min(start_idx + urls_per_node, len(urls))
        node_urls = urls[start_idx:end_idx]
    elif strategy == "interleaved":
        node_urls = urls[node_rank::node_world_size]
    elif strategy == "shuffled_chunk":
        shuffled_urls = random.sample(urls, len(urls))
        urls_per_node = math.ceil(len(shuffled_urls) / node_world_size)
        start_idx = node_rank * urls_per_node
        end_idx = min(start_idx + urls_per_node, len(urls))
        node_urls = shuffled_urls[start_idx:end_idx]
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    
    print(f"Node {node_rank} has {len(node_urls)} URLs of {len(urls)} total.")
    print('*'*80)
    return node_urls


def get_dataset_size(shards, estimated_sample_per_shard=1000):
    """estimate the dataset size, based on the number of shards."""
    if ',' in shards:
        shards_list = []
        for pattern in shards.split(','):
            pattern = pattern.strip()
            if not pattern:
                continue
            shards_list.extend(list(braceexpand.braceexpand(pattern)))
    else:
        shards_list = list(braceexpand.braceexpand(shards))
    num_shards = len(shards_list)
    
    total_size = num_shards * estimated_sample_per_shard
    print(f"Estimating dataset size: {total_size} samples ({num_shards} shards * {estimated_sample_per_shard} samples/shard)")
    return total_size, num_shards


# Pipeline helper functions (module level, supports pickle/spawn)

def nodesplitter_identity(urls):
    return urls


def handle_reconstruction_task(sample, handler=log_and_continue):
    in_key = None
    if "in.png" in sample:
        in_key = "in.png"
    elif "in.jpg" in sample:
        in_key = "in.jpg"
    
    out_key = None
    if "out.png" in sample:
        out_key = "out.png"
    elif "out.jpg" in sample:
        out_key = "out.jpg"
    
    if in_key and not out_key:
        if in_key == "in.png":
            sample["out.png"] = sample["in.png"]
        else:
            sample["out.jpg"] = sample["in.jpg"]
    
    return sample


def extract_fields_to_tuple(sample, handler=log_and_continue):
    in_img = sample.get("in.png") or sample.get("in.jpg")
    out_img = sample.get("out.png") or sample.get("out.jpg")
    if out_img is None and in_img is not None:
        out_img = in_img
    sample_type = sample.get("type", None)
    
    return (in_img, out_img, sample_type)


def identity_function(x, handler=log_and_continue):
    return x


def has_input_image(sample):
    return "in.png" in sample or "in.jpg" in sample


class WeightedRoundRobinSampler(IterableDataset):
    def __init__(self, pipelines, weights):
        super().__init__()
        if len(weights) != len(pipelines):
            raise ValueError(f"number of weights ({len(weights)}) must be equal to the number of pipelines ({len(pipelines)})")
        
        self.pipelines = pipelines
        self.weights = weights
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        max_decimal_places = max(len(str(w).split('.')[-1]) if '.' in str(w) else 0 for w in normalized_weights)
        scale_factor = 10 ** max_decimal_places
        int_weights = [int(w * scale_factor) for w in normalized_weights]
        
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        def gcd_list(nums):
            result = nums[0]
            for num in nums[1:]:
                result = gcd(result, num)
            return result
        
        common_divisor = gcd_list(int_weights)
        int_weights = [w // common_divisor for w in int_weights]
        
        self.sampling_sequence = []
        for i, weight in enumerate(int_weights):
            self.sampling_sequence.extend([i] * weight)
    
    def __iter__(self):
        import itertools
        
        iterators = [iter(p) for p in self.pipelines]
        sequence_iter = itertools.cycle(self.sampling_sequence)
        active = [True] * len(iterators)
        
        while True:
            if not any(active):
                break
            
            idx = next(sequence_iter)
            if active[idx]:
                try:
                    yield next(iterators[idx])
                except StopIteration:
                    active[idx] = False
                    if not any(active):
                        break
                    continue


class StrictProportionalBatchSampler(IterableDataset):
    """
    a strictly proportional batch sampler (适用于 resampled=True)
    ensure that the samples in each batch are strictly allocated according to the weight ratio
    """
    def __init__(self, pipelines, weights, batch_size):
        super().__init__()
        if len(weights) != len(pipelines):
            raise ValueError(f"number of weights ({len(weights)}) must be equal to the number of pipelines ({len(pipelines)})")

        self.pipelines = pipelines
        self.weights = weights
        self.batch_size = batch_size
        
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        self.samples_per_pipeline = []
        float_counts = [batch_size * w for w in normalized_weights]
        
        int_counts = [round(c) for c in float_counts]
        
        current_sum = sum(int_counts)
        diff = batch_size - current_sum
        
        if diff != 0:
            errors = [(float_counts[i] - int_counts[i], i) for i in range(len(int_counts))]
            errors.sort(reverse=(diff > 0))
            
            for _ in range(abs(diff)):
                _, idx = errors.pop(0)
                int_counts[idx] += 1 if diff > 0 else -1
        
        self.samples_per_pipeline = int_counts
        
        weight_strs = [f"{w*100:.1f}%" for w in normalized_weights]
        sample_strs = [f"{count}" for count in self.samples_per_pipeline]
        actual_ratios = [f"{count/batch_size*100:.1f}%" for count in self.samples_per_pipeline]
        print(f"Strict proportional batch sampling enabled:")
        print(f"  Target weights: {' : '.join(weight_strs)}")
        print(f"  Actual samples per batch: {' : '.join(sample_strs)} (total={batch_size})")
        print(f"  Actual ratios: {' : '.join(actual_ratios)}")
    
    def __iter__(self):
        import random as _random
        
        iterators = [iter(p) for p in self.pipelines]
        
        while True:
            batch_samples = []
            
            for idx, count in enumerate(self.samples_per_pipeline):
                for _ in range(count):
                    sample = next(iterators[idx])
                    batch_samples.append(sample)
            
            _random.shuffle(batch_samples)
            
            normalized_samples = []
            for sample in batch_samples:
                if len(sample) == 3:
                    normalized_samples.append((sample[0], sample[1], sample[2], None))
                elif len(sample) == 4:
                    normalized_samples.append(sample)
                else:
                    raise ValueError(f"Unexpected sample length: {len(sample)}")
            
            batch_transposed = list(zip(*normalized_samples))
            
            batch_results = []
            for idx, items in enumerate(batch_transposed):
                if idx < 3:
                    filtered_items = [item for item in items if item is not None]
                    if len(filtered_items) != len(items):
                        raise ValueError(f"Found None in tensor items at index {idx}")
                    batch_results.append(torch.stack(list(filtered_items)))
                else:
                    type_list = list(items)
                    batch_results.append(type_list)
            
            yield tuple(batch_results)
