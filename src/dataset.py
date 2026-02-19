from mmcv.transforms import Compose
from mmengine.registry import DATASETS
from pathlib import Path
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, get_worker_info
import torch
from collections import Counter
import random
from . import preprocessing

@DATASETS.register_module()
class MCIDataset(Dataset):
    
    def __init__(
        self,
        h5_filepath,
        patch_size,
        used_markers=None,
        used_indicies=None,
        pipeline=None,
        ignore_annotation=None,
        preprocess=None,
        mask_patch=None,
        annotation_map=None):
        
        self.h5_filepath = Path(h5_filepath)
        self.patch_size = patch_size
        self.used_markers = used_markers
        self.used_indicies = used_indicies
        self.pipeline = pipeline
        self.half = patch_size // 2
        self.ignore_annotation = ignore_annotation
        self.preprocess=preprocess
        self.mask_patch=mask_patch
        self.annotation_map = annotation_map or {}
        
        assert self.h5_filepath.exists(), f'{self.h5_filepath} does not exist!'
        
        # Don't open HDF5 here for multiprocessing - defer to worker
        self.h5f = None
        self.data_group = None
        self.sample_groups = None
        self._initialized = False
        
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = lambda x: x
        
        # Load metadata only (small, fast)
        with h5py.File(self.h5_filepath, 'r') as h5f_temp:
            coords = h5f_temp['coords']
            self.DIM1 = coords['DIM1'][:]
            self.DIM2 = coords['DIM2'][:]
            self.sample_id = coords['sample_id'][:].astype(str)
            self.annotation = h5f_temp['annotation'][()].astype(str)
            all_marker_names = h5f_temp['marker_names'][:].astype(str)
        
        if self.used_indicies is not None:
            if not isinstance(self.used_indicies, (str, Path)):
                raise TypeError('Please provide indicies as path to a comma separated .txt file!')
            
            indicies_path = Path(self.used_indicies)
            assert indicies_path.exists(), f'{indicies_path} does not exist!'
            
            mask = np.loadtxt(indicies_path, dtype=int)
            self.DIM1 = self.DIM1[mask]
            self.DIM2 = self.DIM2[mask]
            self.sample_id = self.sample_id[mask]
            self.annotation = self.annotation[mask]

        if self.annotation_map:
            self.annotation = np.array([self.annotation_map.get(a, a) for a in self.annotation])

        if self.ignore_annotation is not None:
            
            mask_keep = np.ones(len(self.annotation), dtype=bool)
            
            for annotation in self.ignore_annotation:
                keep = (self.annotation != annotation)
                print(f'Removing {(~keep).sum()} for {annotation}')
                mask_keep &= keep  # Keep where NOT equal
                
            # Apply in one operation (more efficient)
            self.DIM1 = self.DIM1[mask_keep]
            self.DIM2 = self.DIM2[mask_keep]
            self.sample_id = self.sample_id[mask_keep]
            self.annotation = self.annotation[mask_keep]
                
            
        if self.used_markers is not None:
            if isinstance(self.used_markers, (str, Path)):
                self.markers_path = Path(self.used_markers)
                assert self.markers_path.exists(), f'{self.markers_path} does not exist!'
                markers = np.loadtxt(self.markers_path, dtype=str, delimiter=',')
            
            __marker2idx = {name: i for i, name in enumerate(all_marker_names)}
            self.used_marker_indicies = np.array(sorted([__marker2idx[m] for m in markers]))

        else:
            self.markers_path = None
            self.used_marker_indicies = np.arange(len(all_marker_names))
            
        self.used_markers = all_marker_names[self.used_marker_indicies]
        self.marker2idx = {name:i for i,name in enumerate(self.used_markers)}
        self.idx2marker = {i:name for name,i in self.marker2idx.items()}
        
        if self.preprocess is not None:
            processor_type = self.preprocess.pop('type')
            with h5py.File(self.h5_filepath, 'r') as h5f_temp:
                self.preprocess_fn = getattr(preprocessing, processor_type)(**self.preprocess, h5f=h5f_temp, idx2marker=self.idx2marker)
                print(f'Using {self.preprocess_fn} for patch_normalization')

# -------------------------------------------------------------------------------------------------------------------------------------------------------
     
    def __getitem__(self, idx):
        # Initialize HDF5 on first access (per worker)
        if not self._initialized:
            self._init_worker()
        
        DIM1, DIM2, sample_id, annotation = self.DIM1[idx], self.DIM2[idx], self.sample_id[idx], self.annotation[idx]
        
        # Use cached group reference
        sample_group = self.sample_groups[sample_id]
        
        patch = self._get_patch(DIM1, DIM2, sample_group, key='image')
        mask = self._get_patch(DIM1, DIM2, sample_group, key='masks')
        
        # Center-based mask processing
        center = self.half
        mask[mask != mask[center, center]] = 0
        mask = np.expand_dims(mask.astype(bool).astype(float), -1)
        
        if self.preprocess is not None:
            patch = self.preprocess_fn(patch, sample_id)
            
        if self.mask_patch:
            patch = patch * mask
            
        patch = patch.astype(float)
        
        pipeline_dict = {
            'img': patch.astype(np.float32),
            'masks': mask,
            'sample_id': str(sample_id),
            'idx': idx,
            '(DIM1, DIM2)': (DIM1, DIM2),
            'annotation': str(annotation),
            'preprocessing': str(self.preprocess_fn.__class__.__name__) if self.preprocess is not None else 'NO PREPROCESSOR'
        }
        
        return self.pipeline(pipeline_dict)

# -------------------------------------------------------------------------------------------------------------------------------------------------------

    def _get_patch(self, DIM1, DIM2, sample_group, key):
        # Fast shape access
        dataset = sample_group[key]
        shape = dataset.shape
        sDIM1, sDIM2 = shape[0], shape[1]
        half = self.half
        
        # Vectorized boundary calculations
        raw_start_1, raw_end_1 = DIM1 - half, DIM1 + half
        raw_start_2, raw_end_2 = DIM2 - half, DIM2 + half
        
        # Clamp and compute padding in one pass
        start_1 = max(0, raw_start_1)
        end_1 = min(sDIM1, raw_end_1)
        start_2 = max(0, raw_start_2)
        end_2 = min(sDIM2, raw_end_2)
        
        # Simpler padding calculation
        pad_1_low = start_1 - raw_start_1
        pad_1_high = raw_end_1 - end_1
        pad_2_low = start_2 - raw_start_2
        pad_2_high = raw_end_2 - end_2
        
        # Extract with minimal indexing overhead
        if key == 'image':
            if self.used_marker_indicies is not None:
                patch = dataset[start_1:end_1, start_2:end_2, self.used_marker_indicies]
            else:
                patch = dataset[start_1:end_1, start_2:end_2, :]
            needs_pad = pad_1_low or pad_1_high or pad_2_low or pad_2_high
            padding = ((pad_1_low, pad_1_high), (pad_2_low, pad_2_high), (0, 0))
        else:
            patch = dataset[start_1:end_1, start_2:end_2]
            needs_pad = pad_1_low or pad_1_high or pad_2_low or pad_2_high
            padding = ((pad_1_low, pad_1_high), (pad_2_low, pad_2_high))
        
        # Fast path: no padding needed
        if not needs_pad:
            return patch
            
        # Apply padding
        return np.pad(patch, padding, mode='constant', constant_values=0)
        
# -------------------------------------------------------------------------------------------------------------------------------------------------------
        
    def _init_worker(self):
        """Lazy initialization for multiprocessing workers"""
        if self._initialized:
            return
            
        self.h5f = h5py.File(self.h5_filepath, 'r')
        self.data_group = self.h5f['data']
        
        # Cache sample groups to avoid repeated string lookups
        unique_sids = np.unique(self.sample_id)
        self.sample_groups = {sid: self.data_group[sid] for sid in unique_sids}
        
        self._initialized = True
        
# -------------------------------------------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.DIM1)
    
# -------------------------------------------------------------------------------------------------------------------------------------------------------

    def __getstate__(self):
        """Pickle state for multiprocessing"""
        state = self.__dict__.copy()
        # Remove unpicklable HDF5 objects
        state['h5f'] = None
        state['data_group'] = None
        state['sample_groups'] = None
        state['_initialized'] = False
        return state
    
# -------------------------------------------------------------------------------------------------------------------------------------------------------

    def __setstate__(self, state):
        """Restore state in worker process"""
        self.__dict__.update(state)
        # CRITICAL: Don't auto-open here - let __getitem__ do lazy init
        # This avoids pickling issues with h5py objects
        
# -------------------------------------------------------------------------------------------------------------------------------------------------------

    def close(self):
        if self.h5f is not None:
            self.h5f.close()
            self.h5f = None
            self._initialized = False
            
# -------------------------------------------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        
        lines = []
        lines.append(f"{self.__class__.__name__}(")
        lines.append(f"  h5_filepath: {self.h5_filepath.name}")
        lines.append(f"  patch_size: {self.patch_size}")
        lines.append(f"  num_samples: {len(self)}")
        
        # Indices file
        if self.used_indicies is not None:
            lines.append(f"  indices_file: {self.used_indicies}")
        else:
            lines.append(f"  indices_file: None (using all samples)")
        
        # Markers file
        if self.markers_path is not None:
            lines.append(f"  markers_file: {self.markers_path}")
            lines.append(f"  {len(self.used_markers)} markers used: {', '.join(self.used_markers[:5])}" + 
                        (", ..." if len(self.used_markers) > 5 else ""))
        else:
            lines.append(f"  markers_file: None (using all {len(self.used_markers)} markers)")
        
        # Pipeline placeholder
        lines.append(f"  pipeline: {self.pipeline if self.pipeline else 'None (not implemented yet)'}")
        
        # Sample ID examples
        unique_samples = np.unique(self.sample_id)
        lines.append(f"  unique_samples: {len(unique_samples)}")
        examples = random.sample(list(unique_samples), min(5, len(unique_samples)))
        lines.append(f"  sample_examples: {examples}")
        
        # Cell type distribution (if we can infer from sample_id or need to load annotations)
        # Since you don't have annotations loaded, we'll show sample_id distribution
        lines.append(f"  sample_distribution:")
        counts = Counter(self.sample_id)
        total = len(self.sample_id)
        max_count = max(counts.values()) if counts else 1
        
        for sid, count in counts.most_common(10):  # Top 10
            pct = 100 * count / total
            bar_len = int(20 * count / max_count) + 1
            bar = "█" * bar_len
            lines.append(f"    {sid:>20}: {count:>6} ({pct:>6.2f}%) {bar}")
        
        if len(counts) > 10:
            lines.append(f"    ... and {len(counts) - 10} more samples")
        
        lines.append(f"\n  celltype_distribution:")
        
        counts = Counter(self.annotation)
        total = len(self.annotation)
        max_count = max(counts.values()) if counts else 1
        
        for sid, count in counts.most_common(30):  # Top 10
            pct = 100 * count / total
            bar_len = int(20 * count / max_count) + 1
            bar = "█" * bar_len
            lines.append(f"         {sid:<25}: {count:>6} ({pct:>6.2f}%) {bar}")
        
        
        lines.append(")")
        
        return "\n".join(lines)