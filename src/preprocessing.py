import numpy as np


class tanhNormalizer():
    
    def __init__(self, c, h5f, idx2marker, eps=1e-8, rescale=True):
        
        self.c = c
        self.idx2marker = idx2marker
        self.eps = 1e-8
        self.rescale = rescale
        
        self.norm_stats = {sample_id: dict(
            means=np.array([np.nan]*len(self.idx2marker)), 
            stds=np.array([np.nan]*len(self.idx2marker))) 
                      for sample_id in h5f['norm_stats'].keys()}
        
        for sample_id, stats in h5f['norm_stats'].items():
            for idx, marker in self.idx2marker.items():
                self.norm_stats[sample_id]['means'][idx] = stats[marker]['mean'][()]
                self.norm_stats[sample_id]['stds'][idx] = stats[marker]['std'][()]       
            
    def __call__(self, patch, *args, **kwargs):
        return self.transform(patch, *args, **kwargs)
        
    def transform(self, patch, sample_id, *args, **kwargs):
        
        means = self.norm_stats[sample_id]['means']
        stds = self.norm_stats[sample_id]['stds'] + 1e-8
        
        patch = np.tanh((patch - means) / (self.c * stds))
        
        if self.rescale:
            return (patch + 1)/2
        else:
            return patch