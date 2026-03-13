import torchvision.transforms as T
from torch import nn
import numpy as np
from scipy.ndimage import zoom, shift, rotate
import cv2

from mmengine.registry import TRANSFORMS, FUNCTIONS
from mmcv.transforms.base import BaseTransform

from typing import Union, List
from numbers import Number

from codetiming import Timer

from mmcv.transforms import Compose
from copy import deepcopy

ToTensor = T.ToTensor

@TRANSFORMS.register_module()
class C_RandomChannelDrop(BaseTransform):
    
    def __init__(self, drop_prob=0):
        super().__init__()
        
        self.drop_prob = drop_prob
        
    #@Timer(name='C_RandomChannelDrop', text='Function '{name}' took {seconds:.6f} seconds to execute.')        
    def transform(self, results: dict) -> dict:

        img = results['img']
        _, _, C = img.shape
        
        zero_channel = np.zeros_like(img[..., 0])

        for channel in range(C):
            drop = np.random.uniform() <= self.drop_prob
            if drop:
                img[..., channel] = zero_channel
        
        results['img'] = img.copy()
        
        return results
    
@TRANSFORMS.register_module()
class C_RandomChannelCopy(BaseTransform):

    def __init__(self, copy_prob=0.0):
        super().__init__()
        self.copy_prob = float(copy_prob)

    def transform(self, results: dict) -> dict:

        img = results['img']
        H, W, C = img.shape

        # Copy the image first
        new_img = img.copy()

        # Decide for each channel whether to replace it
        for c in range(C):
            if np.random.rand() <= self.copy_prob:
                # pick *any* other channel (including itself, harmless)
                src = np.random.randint(0, C)
                new_img[..., c] = img[..., src]

        results['img'] = new_img
        return results
     
@TRANSFORMS.register_module()
class C_RandomChannelShiftScale(BaseTransform):
    """
    Simulate batch-specific staining shift by applying
    channel-wise affine intensity transformation:
        x' = a * x + b
    where a and b are sampled per channel.
    """

    def __init__(self, scale, shift, clip):
        super().__init__()
        self.scale = scale
        self.shift = shift
        self.clip = clip

    def transform(self, results):
        img = results['img']
        H, W, C = img.shape

        scales = np.random.uniform(self.scale[0], self.scale[1], size=C)
        shifts = np.random.uniform(self.shift[0], self.shift[1], size=C)

        img = img * scales.reshape(1, 1, C) + shifts.reshape(1, 1, C)

        if self.clip:
            img = np.clip(img, 0.0, 1.0)

        results['img'] = img.astype(np.float32)
        results['channel_shift'] = shifts.tolist()
        results['channel_scale'] = scales.tolist()

        return results
    
@TRANSFORMS.register_module()
class C_RandomBackgroundGradient(BaseTransform):
    """
    Add smooth low-frequency gradient to simulate
    illumination bias / background fluorescence.
    """

    def __init__(self, strength, clip):
        super().__init__()
        self.strength = strength
        self.clip = clip

    def transform(self, results):
        img = results['img']
        H, W, C = img.shape

        alpha = np.random.uniform(*self.strength)

        # random 2D gradient
        gx = np.linspace(0, 1, W)
        gy = np.linspace(0, 1, H)
        gradient = np.outer(gy, gx)

        gradient = gradient[..., None]  # H,W,1
        img = img + alpha * gradient

        if self.clip:
            img = np.clip(img, 0.0, 1.0)

        results['img'] = img.astype(np.float32)
        results['bg_strength'] = alpha

        return results
    

@TRANSFORMS.register_module()
class C_RandomChannelMixup(BaseTransform):

    def __init__(self, mixup_prob=0.0, alpha=None):
        super().__init__()
        self.mixup_prob = float(mixup_prob)
        self.alpha=alpha

    def transform(self, results: dict) -> dict:

        img = results['img']
        H, W, C = img.shape

        new_img = img.copy()

        for c in range(C):

            if np.random.rand() <= self.mixup_prob:

                # select a different channel
                src = np.random.randint(0, C - 1)
                if src >= c:
                    src += 1

                # random mixup coefficient
                alpha = self.alpha if self.alpha is not None else np.random.rand()

                new_img[..., c] = alpha * img[..., c] + (1 - alpha) * img[..., src]

        results['img'] = new_img
        return results
    

@TRANSFORMS.register_module()
class C_RandomAffine(BaseTransform):
    
    def __init__(self, angle=(0,360), scale=(0.9, 1.1), shift=(-0.1,0.1), order=0):
        super().__init__()
        
        self.angle = angle
        self.scale = scale
        self.shift = shift
        self.order = order

        assert self.angle[0] <= self.angle[1], f'angle[0]: {angle[0]} must be smaller or equal to angle[1]: {angle[1]}'
        assert self.scale[0] <= self.scale[1], f'scale[0]: {scale[0]} must be smaller or equal to scale[1]: {scale[1]}'
        assert self.shift[0] <= self.shift[1], f'shift[0]: {shift[0]} must be smaller or equal to shift[1]: {shift[1]}'

    #@Timer(name='C_RandomAffine', text='Function '{name}' took {seconds:.6f} seconds to execute.')        
    def transform(self, results: dict) -> dict:
        '''Randomly crop the image and resize the image to the target size.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        '''
        
        img = results['img']
        masks = results['masks']
        height, width = img.shape[:2]

        # Random scaling
        curr_scale = np.random.uniform(self.scale[0], self.scale[1])

        # Random translation
        shift_y = np.random.uniform(self.shift[0], self.shift[1])
        shift_x = np.random.uniform(self.shift[0], self.shift[1])

        # Random rotation
        curr_angle = np.random.uniform(self.angle[0], self.angle[1])

        # Compute the combined transformation matrix
        center = (width / 2, height / 2)

        # Scaling matrix
        scale_matrix = cv2.getRotationMatrix2D(center, 0, curr_scale)

        # Translation matrix
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, curr_angle, 1)

        # Combine the transformation matrices
        transform_matrix = scale_matrix
        transform_matrix[0, 2] += translation_matrix[0, 2]
        transform_matrix[1, 2] += translation_matrix[1, 2]

        dtype = img.dtype
        # Apply the combined transformation matrix
        img = cv2.warpAffine(img.astype(np.float32), transform_matrix, (width, height), flags=self.order, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(dtype)
        masks = cv2.warpAffine(masks.astype(np.float32), transform_matrix, (width, height), flags=self.order, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Apply the rotation after scaling and translation
        img = cv2.warpAffine(img.astype(np.float32), rotation_matrix, (width, height), flags=self.order, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(dtype)
        masks = cv2.warpAffine(masks, rotation_matrix, (width, height), flags=self.order, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        results['img'] = img.copy()
        results['masks'] = masks
        results['angle'] = curr_angle
        results['scale'] = curr_scale
        results['shift'] = (shift_x, shift_y)
        
        return results
    
    
@TRANSFORMS.register_module()   
class C_CentralCutter(BaseTransform):
    
    def __init__(self, size: int):
        super().__init__()
        
        assert (size%2) == 0
        self.hsz = size // 2

    #@Timer(name='C_CentralCutter', text='Function '{name}' took {seconds:.6f} seconds to execute.')    
    def transform(self, results: dict) -> dict:
        '''Add random noise to the image with a mean and std deviation chosen randomly within the specified range.

        Args:
            results (dict): Result dict from the previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        '''
        
        img = results['img']
        masks = results['masks']
        
        c = img.shape[0] // 2, img.shape[1]//2
        
        #cut out the central part
        cropped_img = img[c[0]-self.hsz:c[0]+self.hsz, c[1]-self.hsz:c[1]+self.hsz]
        cropped_masks = masks[c[0]-self.hsz:c[0]+self.hsz, c[1]-self.hsz:c[1]+self.hsz]
        
        results['img'] = cropped_img
        results['masks'] = cropped_masks
        results['cut_size'] = 2*self.hsz
        
        return results


@TRANSFORMS.register_module()
class C_RandomCutter(BaseTransform):
    
    def __init__(self, size: int):
        super().__init__()
        
        assert (size % 2) == 0, "Crop size must be even"
        self.size = size

    def transform(self, results: dict) -> dict:
        img = results['img']
        masks = results['masks']

        H, W = img.shape[:2]

        # ensure crop fits fully inside image
        x = np.random.randint(0, H - self.size + 1)
        y = np.random.randint(0, W - self.size + 1)

        # crop image
        cropped_img = img[x:x+self.size, y:y+self.size]
        cropped_masks = masks[x:x+self.size, y:y+self.size]
        
        results['img'] = cropped_img
        results['masks'] = cropped_masks
        results['cut_size'] = self.size
        results['cut_origin'] = (x, y)

        return results

@TRANSFORMS.register_module()   
class C_RandomNoise(BaseTransform):
    
    def __init__(self, mean, std, clip):
        super().__init__()
        
        self.mean = mean
        assert self.mean[0] <= self.mean[1]
        self.std = std
        assert self.std[0] <= self.std[1]
        
        self.clip = clip

    #@Timer(name='RandomNoise', text='Function '{name}' took {seconds:.6f} seconds to execute.')    
    def transform(self, results: dict) -> dict:
        '''Add random noise to the image with a mean and std deviation chosen randomly within the specified range.

        Args:
            results (dict): Result dict from the previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        '''
        
        img = results['img']
        
        # Randomly choose mean and std within the given range
        curr_mean = np.random.uniform(self.mean[0], self.mean[1])
        curr_std = np.random.uniform(self.std[0], self.std[1])
        
        # Add Gaussian noise to the image
        noise = np.random.normal(curr_mean, curr_std, img.shape)
        img = img + noise
        
        # Clip the values to be in the valid range
        if self.clip:
            img = np.clip(img, 0.0, 1.0)
        
        results['img'] = img.copy()
        results['noise_level'] = (curr_mean, curr_std)
        
        return results
    
    
@TRANSFORMS.register_module()
class C_RandomIntensity(BaseTransform):
    
    def __init__(self, low, high, clip):
        super().__init__()
        self.low = low
        self.high = high
        if isinstance(self.low, (list, tuple)) and isinstance(self.high, (list, tuple)):
            assert len(self.low) == len(self.high), 'low and high must be of same length or a float'
            assert all([l <= h for l,h in zip(self.low, self.high)]), f'all values of low must be lower than the corresponding value of high! {self.low} - {self.high}'
            
            self.create_list = False
            
        if isinstance(self.low, Number) and isinstance(self.high, Number):
            assert self.low  <= self.high
            
            self.create_list = True
            
        self.clip = clip


    #@Timer(name='RandomIntensity', text='Function '{name}' took {seconds:.6f} seconds to execute.')                
    def transform(self, results: dict) -> dict:
        '''Randomly adjust the intensity of the image channels within the specified range.

        Args:
            results (dict): Result dict from the previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        '''
        
        img = results['img']
        n_channels = img.shape[-1]
        
        # add functionality that if l and h is just a scalar value that it checks how man chanels are in an image and just copies l and h that often
        if self.create_list:
            curr_low = [self.low for _ in range(n_channels)]
            curr_high = [self.high for _ in range(n_channels)]
            
        else:
            curr_low = self.low
            curr_high = self.high

        # Randomly choose scaling factors for each channel within the given range
        channel_scaling = np.array([np.random.uniform(l, h) for l, h in zip(curr_low, curr_high)])
        
        # Apply scaling to each channel
        img = img * channel_scaling.reshape(1, 1, n_channels)
        
        # Clip the values to be in the valid range
        if self.clip:
            img = np.clip(img, 0.0, 1.0)
        
        results['img'] = img.copy()
        results['channel_scaling'] = channel_scaling.tolist()  # Store the scaling factors used
        
        return results


@TRANSFORMS.register_module() 
class C_ToTensor(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.tt = ToTensor()
        
    def forward(self, input_dict):
        return {k: self.tt(v) if isinstance(v, np.ndarray) else v for k, v in input_dict.items()}
    
    
@TRANSFORMS.register_module()
class C_TensorCombiner(BaseTransform):
    
    def __init__(self):
        super().__init__()

    def transform(self, results) -> dict:
        """Concatenate image and mask tensors along the last dimension.

        Args:
            results (dict): Result dictionary containing 'img' and 'masks'.

        Returns:
            dict: Updated result dictionary with concatenated image and mask tensor.
        """
        if results['masks']:
            img = results['img']
            masks = np.atleast_3d(np.array(results['masks'])).transpose(1,2,0)  
            concat_tensor = np.concatenate((img, masks), axis=-1)
            results['img'] = concat_tensor
        
        return results
    
@TRANSFORMS.register_module()
class C_RandomFlip(BaseTransform):
    def __init__(self, prob=0.5, horizontal=True, vertical=True):
        """
        Randomly flip the image (and masks) horizontally and/or vertically.

        Args:
            prob (float): Probability of applying each enabled flip.
            horizontal (bool): Whether to allow horizontal flip.
            vertical (bool): Whether to allow vertical flip.
        """
        super().__init__()
        self.prob = float(prob)
        self.horizontal = bool(horizontal)
        self.vertical = bool(vertical)

        if not (self.horizontal or self.vertical):
            raise ValueError("At least one of horizontal or vertical must be True.")

    def transform(self, results: dict) -> dict:
        img = results['img']
        masks = results.get('masks', [])

        h_flip = np.random.rand() < self.prob
        v_flip = np.random.rand() < self.prob
        
        # Horizontal flip
        if self.horizontal and h_flip:
            img = np.flip(img, axis=1)  # flip width axis
            masks = np.flip(masks, axis=1)
        

        # Vertical flip
        if self.vertical and v_flip:
            img = np.flip(img, axis=0)  # flip height axis
            masks = np.flip(masks, axis=0)

        results['img'] = img.copy()
        results['masks'] = masks.copy()
        results['h_flip'] = h_flip
        results['v_flip'] = v_flip

        return results
    
@TRANSFORMS.register_module()
class C_PatientStyleTransfer(BaseTransform):
    """
    Augmentation that removes patient-level intensity batch effects by
    transferring the intensity profile of a randomly drawn target patient.

    For each marker channel c:
        x_c_new = (x_c - src_mean[c]) / (src_std[c] + eps) * tgt_std[c] + tgt_mean[c]

    This destroys the source patient's absolute intensity fingerprint while
    preserving within-patch relative structure (which marker is high vs low).
    Applied independently to each view in MultiView training, forcing VICReg's
    invariance loss to learn representations that are invariant to patient
    intensity profiles — analogous to color jitter in natural image SSL.

    Per-patient per-marker statistics are precomputed from the h5 file at init:
    reads from 'norm_stats' group if present, otherwise scans the full images.

    Args:
        h5_filepath:   path to the dataset h5 file
        used_markers:  path to used_markers.txt (comma-separated), or list of names
        eps:           floor for source std to avoid division by zero
        clip:          clip output to [0, 1] (recommended)
    """

    def __init__(self, h5_filepath, used_markers, eps=1e-6, clip=True):
        super().__init__()
        self.eps = eps
        self.clip = clip
        self._stats = self._load_stats(h5_filepath, used_markers)
        self.patient_ids = list(self._stats.keys())

    @staticmethod
    def _load_stats(h5_filepath, used_markers):
        import h5py
        from pathlib import Path

        h5_filepath = Path(h5_filepath)
        with h5py.File(h5_filepath, 'r') as f:
            all_markers = f['marker_names'][:].astype(str)

            if isinstance(used_markers, (str, Path)):
                markers = np.loadtxt(used_markers, dtype=str, delimiter=',')
            else:
                markers = np.array(used_markers)

            marker2idx = {name: i for i, name in enumerate(all_markers)}
            used_idx   = np.array(sorted([marker2idx[m] for m in markers]))
            used_names = all_markers[used_idx]

            stats = {}

            if 'norm_stats' in f:
                # Fast path: pre-computed stats already in h5
                for sid in f['norm_stats'].keys():
                    means = np.array([f['norm_stats'][sid][m]['mean'][()] for m in used_names],
                                     dtype=np.float32)
                    stds  = np.array([f['norm_stats'][sid][m]['std'][()]  for m in used_names],
                                     dtype=np.float32)
                    stats[sid] = {'mean': means, 'std': stds}
            else:
                # Compute from full patient images (one-time cost at init)
                for sid in f['data'].keys():
                    img  = f['data'][sid]['image'][:, :, used_idx]   # [H, W, C]
                    flat = img.reshape(-1, img.shape[-1])             # [H*W, C]
                    stats[sid] = {
                        'mean': flat.mean(axis=0).astype(np.float32),
                        'std':  flat.std(axis=0).astype(np.float32),
                    }

        print(f"  C_PatientStyleTransfer: loaded stats for {len(stats)} patients "
              f"({'norm_stats' if 'norm_stats' in open(str(h5_filepath), 'rb').read(200) else 'computed'})")
        return stats

    def transform(self, results):
        img    = results['img']                            # [H, W, C]
        src_id = results['sample_id']

        # pick a different target patient
        candidates = [p for p in self.patient_ids if p != src_id]
        tgt_id = candidates[np.random.randint(len(candidates))] if candidates else src_id

        src_mean = self._stats[src_id]['mean']   # [C]
        src_std  = self._stats[src_id]['std']    # [C]
        tgt_mean = self._stats[tgt_id]['mean']   # [C]
        tgt_std  = self._stats[tgt_id]['std']    # [C]

        # For channels with near-zero source std, shift only (avoid amplifying noise)
        has_signal = src_std > self.eps
        scale = np.where(has_signal, tgt_std / (src_std + self.eps), 1.0)
        shift = np.where(has_signal,
                         tgt_mean - src_mean * scale,
                         tgt_mean - src_mean)

        img = img * scale[None, None, :] + shift[None, None, :]

        if self.clip:
            img = np.clip(img, 0.0, 1.0)

        results['img'] = img.astype(np.float32)
        results['style_source'] = src_id
        results['style_target'] = tgt_id
        return results


from mmselfsup.structures import SelfSupDataSample

@TRANSFORMS.register_module()
class C_PackInputs(BaseTransform):
    
    def transform(self, results):
    
        return dict(
            inputs=results['img'],
            data_samples={k:v for k,v in results.items() if k!='img'}
        )
        
@TRANSFORMS.register_module()
class C_MultiView(BaseTransform):
    
    def __init__(self, n_views, transforms):

        assert len(n_views) == len(transforms)
        
        self.n_views = n_views
        self.transforms = [Compose(transform) for transform in transforms]

    def transform(self, results):
        
        return_dict = dict(img=[])
        for n, transform in zip(self.n_views, self.transforms):
            for _ in range(n):
                _results = deepcopy(results)
                for t in transform:
                    _results = t(_results)
                    #print(_results['img'].shape)
                    
                return_dict['img'].append(_results['img'])
                for k, v in _results.items():
                    if k == 'img':
                        continue
                    elif k in return_dict:
                        return_dict[k].append(v)
                    else:
                        return_dict[k] = [v]
                
        return return_dict
    
    

