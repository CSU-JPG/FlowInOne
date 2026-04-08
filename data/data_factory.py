"""
Dataset factory class (OnlineFeatures) and DataLoader wrapper (WebDataLoader).
"""

from PIL import Image
import os

import webdataset as wds

from data.transforms import DatasetFactory
from data.web_dataset import WebDatasetDataset


class WebDataLoader:
    """
    encapsulates a unified interface for WebDataset and wds.WebLoader.
    
    """
    
    @staticmethod
    def create(dataset, vl_chat_processor=None, device=None,
               pin_memory=True, persistent_workers=True):
        
        if vl_chat_processor is not None:
            dataset.set_vl_chat_processor(vl_chat_processor)
        if device is not None:
            dataset.set_device(device)
        
        num_workers = dataset.num_workers
        
        return wds.WebLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )


class OnlineFeatures(DatasetFactory):
    """
    The dataset factory class uses WebDatasetDataset to load data from tar files.
    
    """
    def __init__(self, vis_image_root=None, train_tar_pattern=None, test_tar_pattern=None,
                 task='visual_instruction', cfg=False, resolution=256,
                 shuffle_buffer=300, resampled=True, split_data_by_node=True, estimated_samples_per_shard=1000,
                 vl_chat_processor=None, device=None, fid_stat_path=None,
                 num_workers=None, batch_size=None, test_batch_size=None, test_num_workers=None, sampling_weights=None,
                 **kwargs):
        """
        Args:
            vis_image_root: Root directory of the visualization image
            train_tar_pattern: Training set tar file path pattern, supports braceexpand
            test_tar_pattern: Test set tar file path pattern, supports braceexpand
            task: Task type
            cfg: Whether to use classifier-free guidance
            resolution: Image resolution
            shuffle_buffer: Size of the WebDataset shuffle buffer
            resampled: Whether to use resampled mode (for distributed training)
            split_data_by_node: Whether to distribute shards across multiple nodes
            estimated_samples_per_shard: Estimated number of samples per shard
            vl_chat_processor: VLChatProcessor instance (optional)
            device: torch.device or string (optional)
            num_workers: num_workers of the DataLoader (optional)
            batch_size: Training set batch size (optional)
            test_batch_size: Test set batch size (optional)
            test_num_workers: Test set num_workers (optional)
            sampling_weights: List of sampling weights (optional)
        """
        super().__init__()
        self.task = task
        self.vis_image_root = vis_image_root
        self.fid_stat_path = fid_stat_path
        
        if train_tar_pattern is None:
            raise ValueError("train_tar_pattern must be provided")
        
        print(f'Creating WebDataset with pattern: {train_tar_pattern}')
        
        self.train = WebDatasetDataset(
            tar_pattern=train_tar_pattern,
            resolution=resolution,
            shuffle_buffer=shuffle_buffer,
            resampled=resampled,
            split_data_by_node_flag=split_data_by_node,
            estimated_samples_per_shard=estimated_samples_per_shard,
            vl_chat_processor=vl_chat_processor,
            device=device,
            num_workers=num_workers,
            batch_size=batch_size,
            sampling_weights=sampling_weights
        )
        test_batch_size_to_use = test_batch_size if test_batch_size is not None else batch_size
        test_num_workers_to_use = test_num_workers if test_num_workers is not None else num_workers
        self.test = WebDatasetDataset(
            tar_pattern=test_tar_pattern,
            resolution=resolution,
            shuffle_buffer=100,
            resampled=False,
            split_data_by_node_flag=split_data_by_node,
            allow_shared_shards=False,
            estimated_samples_per_shard=estimated_samples_per_shard,
            vl_chat_processor=vl_chat_processor,
            device=device,
            num_workers=test_num_workers_to_use,
            batch_size=test_batch_size_to_use,
            force_simple_mode=True,
            enable_shuffle=False,
            partial=True
        )
        
        assert not cfg
        self.resolution = resolution

        self.vis_image_paths = []
        self.vis_output_paths = []
        self._scan_vis_images(vis_image_root)

        self._train_dataloader = None
        self._test_dataloader = None
    

    def set_vl_chat_processor(self, vl_chat_processor):
        """bind VLChatProcessor, and invalidate the cached DataLoader."""
        self.train.set_vl_chat_processor(vl_chat_processor)
        self.test.set_vl_chat_processor(vl_chat_processor)
        self._train_dataloader = None
        self._test_dataloader = None

    def set_device(self, device):
        """bind target device, and invalidate the cached DataLoader."""
        self.train.set_device(device)
        self.test.set_device(device)
        self._train_dataloader = None
        self._test_dataloader = None


    @property
    def train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = WebDataLoader.create(self.train)
        return self._train_dataloader

    @property
    def test_dataloader(self):
        if self._test_dataloader is None:
            self._test_dataloader = WebDataLoader.create(self.test)
        return self._test_dataloader

    def _scan_vis_images(self, vis_image_root):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}
        
        if not vis_image_root or not os.path.exists(vis_image_root):
            if vis_image_root:
                print(f"Warning: vis_image_root does not exist: {vis_image_root}")
            return
        
        input_dir = os.path.join(vis_image_root, 'input')
        output_dir = os.path.join(vis_image_root, 'output')
        
        if os.path.exists(input_dir):
            print(f"Scanning input images in: {input_dir}")
            for root, dirs, files in os.walk(input_dir):
                for filename in files:
                    if any(filename.endswith(ext) for ext in valid_extensions):
                        self.vis_image_paths.append(os.path.join(root, filename))
            self.vis_image_paths = sorted(self.vis_image_paths)
            print(f"Found {len(self.vis_image_paths)} input images")
        else:
            print(f"Warning: input directory does not exist: {input_dir}")
        
        if os.path.exists(output_dir):
            print(f"Scanning output images in: {output_dir}")
            for root, dirs, files in os.walk(output_dir):
                for filename in files:
                    if any(filename.endswith(ext) for ext in valid_extensions):
                        self.vis_output_paths.append(os.path.join(root, filename))
            self.vis_output_paths = sorted(self.vis_output_paths)
            print(f"Found {len(self.vis_output_paths)} output images")
        else:
            print(f"Warning: output directory does not exist: {output_dir}")
        
        if self.vis_image_paths and self.vis_output_paths:
            input_map = {os.path.splitext(os.path.basename(p))[0]: p for p in self.vis_image_paths}
            output_map = {os.path.splitext(os.path.basename(p))[0]: p for p in self.vis_output_paths}
            
            matched_keys = sorted(set(input_map.keys()) & set(output_map.keys()))
            self.vis_image_paths = [input_map[key] for key in matched_keys]
            self.vis_output_paths = [output_map[key] for key in matched_keys]
            print(f"Matched {len(self.vis_image_paths)} input-output image pairs")
        
        print(f"Images will be loaded on-demand when needed.")

    @staticmethod
    def _load_images_parallel(paths, max_workers=8):
        import concurrent.futures
        
        if not paths:
            return []
        
        def load_image(image_path):
            try:
                pil_img = Image.open(image_path)
                pil_img.load()
                return pil_img.convert("RGB")
            except Exception as e:
                print(f"Warning: Failed to load image {image_path}: {e}")
                return Image.new('RGB', (384, 384), color='black')
        
        workers = min(len(paths), max_workers)
        if workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                return list(executor.map(load_image, paths))
        return [load_image(path) for path in paths]

    def get_vis_images_as_pil(self, max_images=None):
        paths = self.vis_image_paths[:max_images] if max_images else self.vis_image_paths
        return self._load_images_parallel(paths)
    
    def get_vis_output_images_as_pil(self, max_images=None):
        paths = self.vis_output_paths[:max_images] if max_images else self.vis_output_paths
        return self._load_images_parallel(paths)

    @property
    def data_shape(self):
        if self.resolution == 512:
            return 4, 64, 64
        else:
            return 4, 32, 32
    
    @property
    def fid_stat(self):
        if self.fid_stat_path:
            return self.fid_stat_path
        return '/path/to/fid_stats_mscoco256_val.npz'
