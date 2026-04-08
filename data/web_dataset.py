"""
The core class for loading input/output image pairs using WebDataset.
"""

from torch.utils.data import IterableDataset
import numpy as np
import torch
import math
from PIL import Image
import einops

import webdataset as wds
import braceexpand

from data.transforms import center_crop_arr
from data.wds_utils import (
    log_and_continue,
    pytorch_worker_info,
    is_multi_node_environment,
    get_dataset_size,
    handle_reconstruction_task,
    extract_fields_to_tuple,
    identity_function,
    has_input_image,
    WeightedRoundRobinSampler,
    StrictProportionalBatchSampler,
)


class WebDatasetDataset(IterableDataset):
    """
    load input/output image pairs using WebDataset.
    """
    def __init__(self, tar_pattern, resolution=256, shuffle_buffer=300, 
                 resampled=True, handler=log_and_continue, 
                 estimated_samples_per_shard=1000,
                 split_data_by_node_flag=True,
                 allow_shared_shards=False,
                 vl_chat_processor=None,
                 device=None,
                 num_workers=None,
                 batch_size=None,
                 sampling_weights=None,
                 force_simple_mode=False,
                 enable_shuffle=True,
                 partial=False):
        super().__init__()
        self.resolution = resolution
        self.handler = handler
        self.vl_chat_processor = vl_chat_processor
        self.num_workers = num_workers if num_workers is not None else 1
        self.batch_size = batch_size
        self.sampling_weights = sampling_weights
        self.enable_shuffle = enable_shuffle
        self.partial = partial
        self.resampled = resampled

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device(f'cuda:{torch.cuda.current_device()}')
            else:
                self.device = torch.device('cpu')
        else:
            if isinstance(device, str):
                self.device = torch.device(device)
            else:
                self.device = device
        
        patterns = [p.strip() for p in tar_pattern.split(',') if p.strip()]
        self.use_proportional_sampling = len(patterns) > 1 and not force_simple_mode
        
        if self.use_proportional_sampling:
            self._init_proportional_sampling(
                patterns, tar_pattern, shuffle_buffer, resampled, handler,
                estimated_samples_per_shard, split_data_by_node_flag, allow_shared_shards
            )
        else:
            self._init_simple_mode(
                patterns, tar_pattern, shuffle_buffer, resampled, handler,
                estimated_samples_per_shard, split_data_by_node_flag, allow_shared_shards
            )


    def _init_proportional_sampling(self, patterns, tar_pattern, shuffle_buffer,
                                     resampled, handler, estimated_samples_per_shard,
                                     split_data_by_node_flag, allow_shared_shards):
        weights_str = None
        if self.sampling_weights is not None:
            if len(self.sampling_weights) != len(patterns):
                raise ValueError(f"number of sampling weights ({len(self.sampling_weights)}) must be equal to the number of path patterns ({len(patterns)})")
            if any(w <= 0 for w in self.sampling_weights):
                raise ValueError("all sampling weights must be positive")
            weights_str = " : ".join([f"{w*100:.1f}%" for w in self.sampling_weights])
            print(f"Detected {len(patterns)} path patterns, using weighted sampling with ratios ({weights_str})")
        else:
            print(f"Detected {len(patterns)} path patterns, using RoundRobin for proportional sampling (50:50)")
        
        pipelines = []
        total_shards = 0
        
        for i, pattern in enumerate(patterns):
            pattern_urls = list(braceexpand.braceexpand(pattern))
            total_shards += len(pattern_urls)

            if allow_shared_shards:
                need_nodesplitter = False
            elif split_data_by_node_flag and is_multi_node_environment():
                need_nodesplitter = True
            else:
                need_nodesplitter = False
            
            pipeline = self._create_single_pattern_pipeline(
                pattern_urls, shuffle_buffer, resampled, handler,
                need_nodesplitter, allow_shared_shards
            )
            pipelines.append(pipeline)
            print(f"Pattern {i+1}: {len(pattern_urls)} shards")
        
        self.num_shards = total_shards
        if split_data_by_node_flag:
            total_num_samples, _ = get_dataset_size(tar_pattern, estimated_samples_per_shard)
            self.total_num_samples = total_num_samples
        else:
            self.total_num_samples = total_shards * estimated_samples_per_shard
        
        self.num_samples = self.num_shards * estimated_samples_per_shard
        with_epoch_size = self._compute_epoch_size()
        
        if self.sampling_weights is not None:
            if self.batch_size is not None:
                if not self.resampled:
                    raise ValueError("StrictProportionalBatchSampler only used in resampled=True")
                merged_source = StrictProportionalBatchSampler(
                    pipelines, self.sampling_weights, self.batch_size
                )
                batch_already_done = True
            else:
                merged_source = WeightedRoundRobinSampler(pipelines, self.sampling_weights)
                batch_already_done = False
        else:
            merged_source = wds.RoundRobin(*pipelines)
            batch_already_done = False
        
        pipeline_stages = [merged_source]
        if self.batch_size is not None and not batch_already_done:
            pipeline_stages.append(wds.batched(self.batch_size, partial=self.partial))
        
        self._pipeline = wds.DataPipeline(*pipeline_stages).with_epoch(with_epoch_size)
        
        sampling_mode = "weighted sampling" if self.sampling_weights is not None else "RoundRobin"
        weights_info = f" ({weights_str})" if self.sampling_weights is not None else ""
        if self.batch_size is not None:
            print(f"{sampling_mode} mode{weights_info}: {len(patterns)} patterns, {total_shards} total shards, "
                  f"with_epoch({with_epoch_size} batches per worker), "
                  f"num_batches={self.num_batches}, num_samples={self.num_samples}")
        else:
            print(f"{sampling_mode} mode{weights_info}: {len(patterns)} patterns, {total_shards} total shards, "
                  f"with_epoch({with_epoch_size} samples per worker), num_samples={self.num_samples}")


    def _init_simple_mode(self, patterns, tar_pattern, shuffle_buffer,
                           resampled, handler, estimated_samples_per_shard,
                           split_data_by_node_flag, allow_shared_shards):
        if len(patterns) > 1:
            print(f"Simple mode: detected {len(patterns)} path patterns, merging all paths")
            all_urls = []
            for i, pattern in enumerate(patterns):
                pattern_urls = list(braceexpand.braceexpand(pattern))
                all_urls.extend(pattern_urls)
                print(f"  Pattern {i+1}: {len(pattern_urls)} shards")
            print(f"  Total merged: {len(all_urls)} shards")
        else:
            all_urls = list(braceexpand.braceexpand(tar_pattern))

        urls = all_urls
        
        need_nodesplitter = False
        if allow_shared_shards:
            print(f"Shared shards mode: all {len(urls)} shards accessible by all processes")
        elif split_data_by_node_flag and is_multi_node_environment():
            need_nodesplitter = True
            print(f"Multi-process mode: using {len(urls)} shards, will be split by nodesplitter")
        else:
            print(f"Single process mode: using all {len(urls)} shards")
        
        self.num_shards = len(urls)
        if split_data_by_node_flag:
            total_num_samples, _ = get_dataset_size(tar_pattern, estimated_samples_per_shard)
            self.num_samples = self.num_shards * estimated_samples_per_shard
            self.total_num_samples = total_num_samples
        else:
            self.num_samples = self.num_shards * estimated_samples_per_shard
            self.total_num_samples = self.num_samples
        
        if not resampled:
            _, world_size, _, _ = pytorch_worker_info()
            total_workers_needed = self.num_workers * world_size if world_size > 1 else self.num_workers
            if self.num_shards < total_workers_needed:
                print(f"Warning: Only {self.num_shards} shards but need {total_workers_needed} workers "
                      f"(num_workers={self.num_workers}, world_size={world_size}). "
                      f"Some workers may not have data. Consider using resampled=True or increasing shard count.")
        
        with_epoch_size = self._compute_epoch_size()
        
        if resampled:
            shard_source = wds.ResampledShards(urls)
        else:
            shard_source = wds.SimpleShardList(urls)
        
        pipeline_stages = [shard_source]
        
        if not allow_shared_shards and need_nodesplitter and hasattr(wds, "split_by_node"):
            pipeline_stages.append(wds.split_by_node)
            print(f"Using wds.split_by_node for multi-process training")
        
        if self.num_workers > 1:
            pipeline_stages.append(wds.split_by_worker)
            print(f"Added wds.split_by_worker for {self.num_workers} workers")
        
        pipeline_stages.append(wds.tarfile_to_samples(handler=handler))
        pipeline_stages.extend(self._build_processing_stages(shuffle_buffer, handler))
        
        if self.batch_size is not None:
            pipeline_stages.append(wds.batched(self.batch_size, partial=self.partial))
        
        self._pipeline = wds.DataPipeline(*pipeline_stages).with_epoch(with_epoch_size)
        
        if self.batch_size is not None:
            print(f"WebDataset initialized: {self.num_shards} shards for this node, "
                  f"with_epoch({with_epoch_size} batches per worker), "
                  f"num_batches={self.num_batches}, num_samples={self.num_samples} "
                  f"(num_workers={self.num_workers}, batch_size={self.batch_size})")
        else:
            print(f"WebDataset initialized: {self.num_shards} shards for this node, "
                  f"with_epoch({with_epoch_size} samples per worker), "
                  f"num_samples={self.num_samples} (num_workers={self.num_workers})")


    def _compute_epoch_size(self):
        if self.batch_size is not None:
            _, world_size, _, _ = pytorch_worker_info()
            num_worker_batches = math.ceil(self.total_num_samples / (world_size * self.batch_size * self.num_workers))
            self.num_batches = num_worker_batches * self.num_workers
            self.num_samples = self.num_batches * self.batch_size
            return num_worker_batches
        else:
            if self.num_workers > 1:
                epoch_size = math.ceil(self.num_samples / self.num_workers)
            else:
                epoch_size = self.num_samples
            self.num_samples = epoch_size
            return epoch_size

    def _build_processing_stages(self, shuffle_buffer, handler):
        stages = []
        if self.enable_shuffle:
            stages.append(wds.shuffle(shuffle_buffer, handler=handler))
        
        stages.extend([
            wds.decode("pil", handler=handler),
            wds.map(handle_reconstruction_task, handler=handler),
            wds.select(has_input_image),
            wds.map(extract_fields_to_tuple),
            wds.map_tuple(
                self._preprocess_input,
                self._preprocess_output,
                identity_function
            ),
            wds.map(self._unpack_input_tuple),
        ])
        return stages

    def _create_single_pattern_pipeline(self, urls, shuffle_buffer, resampled, handler,
                                        need_nodesplitter, allow_shared_shards):
        if resampled:
            shard_source = wds.ResampledShards(urls)
        else:
            shard_source = wds.SimpleShardList(urls)
        
        pipeline_stages = [shard_source]
        
        if not allow_shared_shards and need_nodesplitter and hasattr(wds, "split_by_node"):
            pipeline_stages.append(wds.split_by_node)
        
        if self.num_workers > 1:
            pipeline_stages.append(wds.split_by_worker)
        
        pipeline_stages.append(wds.tarfile_to_samples(handler=handler))
        pipeline_stages.extend(self._build_processing_stages(shuffle_buffer, handler))
        
        return wds.DataPipeline(*pipeline_stages)

    
    def _preprocess_input(self, pil_image):
        if not isinstance(pil_image, Image.Image):
            pil_image = Image.fromarray(pil_image)
        pil_image = pil_image.convert("RGB")
        
        input_arr = center_crop_arr(pil_image, image_size=self.resolution)
        input_arr = (input_arr / 127.5 - 1.0).astype(np.float32)
        input_tensor = torch.from_numpy(einops.rearrange(input_arr, 'h w c -> c h w'))
        
        if self.vl_chat_processor is not None:
            images_outputs = self.vl_chat_processor.image_processor(
                [pil_image], 
                return_tensors="pt"
            )
            pixel_values = images_outputs.pixel_values.squeeze(0)
            return pixel_values, input_tensor
        else:
            return np.array(pil_image, dtype=np.uint8), input_tensor
    
    def _preprocess_output(self, pil_image):
        if not isinstance(pil_image, Image.Image):
            pil_image = Image.fromarray(pil_image)
        pil_image = pil_image.convert("RGB")
        output_arr = center_crop_arr(pil_image, image_size=self.resolution)
        output_arr = (output_arr / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(einops.rearrange(output_arr, 'h w c -> c h w'))
    
    def _unpack_input_tuple(self, sample):
        input_tuple, output_tensor, sample_type = sample
        pixel_values, input_tensor = input_tuple
        return pixel_values, output_tensor, input_tensor, sample_type

    
    def __iter__(self):
        return iter(self._pipeline)
    
    def set_vl_chat_processor(self, vl_chat_processor):
        self.vl_chat_processor = vl_chat_processor
    
    def set_device(self, device):
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
