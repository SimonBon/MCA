from tqdm import tqdm
import numpy as np
import torch
import pandas as pd

def run_on_dataloader(dataloader, model, dataset, run_n=np.inf, get_relevance=False):

    model.eval()

    results_dict = dict(features=[], annotations=[], avg_expressions=[], dataset_idx=[])
    for key in dataset.additional_keys:
        results_dict[key]=[]
        
    if get_relevance:
        results_dict['spatial_relevance']=[]
        results_dict['channelwise_relevance']=[]

    for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    
        # if batch['masks']:
        #     imgs = batch['img'] * batch['masks'][..., None]
        #     results_dict['avg_expressions'].extend(np.array(imgs.sum(axis=(1,2)) / batch['masks'].sum(axis=(1,2))[..., None]))
    
        # else:
        if 'avg_expressions' in results_dict:
            del results_dict['avg_expressions']
            
        # batch['img']: (B, H, W, C)  -> convert to (B, C, H, W)
        imgs = [batch['inputs'][0].float().to(model.device)]

        if get_relevance:
            feats, spatial_relevance, channelwise_relevance = model.get_relevance(imgs)  # take 0th output
            results_dict['spatial_relevance'].extend(spatial_relevance)
            results_dict['channelwise_relevance'].extend(channelwise_relevance)
            if feats.ndim == 4:
                feats = feats.mean(axis=(2,3))
        else:
            with torch.no_grad():
                feats = model(imgs)[0].squeeze()  # take 0th output
                if feats.ndim == 4:
                    feats = feats.mean(axis=(2,3))
            
        feats = feats.detach().cpu().numpy()
        results_dict['features'].extend(feats)
        results_dict['annotations'].extend([b.annotation for b in batch['data_samples']])
        results_dict['dataset_idx'].extend([b.dataset_idx for b in batch['data_samples']])
        for key in dataset.additional_keys:
            results_dict[key].extend([getattr(b, key) for b in batch['data_samples']])
        
        if len(results_dict['features']) > run_n:
            return results_dict
            
    return results_dict

def add_module_scores_topk(df, module_definitions, k=2, simple=False):
    weight_df = pd.DataFrame(module_definitions).T.fillna(0)

    common_markers = [m for m in weight_df.columns if m in df.columns]
    weight_df = weight_df[common_markers]

    module_scores = {}
    for module, markers in module_definitions.items():
        markers_present = [m for m in markers.keys() if m in df.columns]
        sub = df[markers_present].values
        
        # Sort each row (cell) by expression descending
        topk_vals = np.sort(sub, axis=1)[:, -k:]
        
        # Score = mean of top-k marker expressions
        module_scores[module] = topk_vals.mean(axis=1)
        
    for module in module_scores:
        df[module] = module_scores[module]
    
    return df

def add_module_scores(df, module_definitions, simple=True):
    
    weight_df = pd.DataFrame(module_definitions).T.fillna(0)

    missing = [m for m in weight_df.columns if m not in df.columns]
    if missing:
        print("WARNING: markers missing in expression matrix:", missing)

    common_markers = [m for m in weight_df.columns if m in df.columns]
    if not common_markers:
        raise ValueError("No module markers found in dataframe columns.")

    weight_df = weight_df[common_markers]
    X = df[common_markers].values

    # --- simple vs weighted scoring ---
    if simple:
        wdf = (weight_df > 0).astype(float)
    else:
        wdf = weight_df.copy()

    W = wdf.T.values

    # module score calculation
    module_scores_matrix = X @ W
    norm_factors = wdf.sum(axis=1).values
    module_scores_matrix = module_scores_matrix / norm_factors

    module_score_df = pd.DataFrame(
        module_scores_matrix,
        index=df.index,
        columns=weight_df.index
    )
    
    for module in module_score_df:
        df[module] = module_score_df[module]

    # # ----------------------------
    # # UNCERTAINTY MEASURES
    # # ----------------------------
    # scores = module_scores_matrix

    # # (1) margin‐based uncertainty
    # best = np.max(scores, axis=1)
    # second = np.partition(scores, -2, axis=1)[:, -2]
    # uncertainty_margin = 1 - (best - second)

    # # (2) entropy‐based uncertainty
    # # normalize scores into probabilities
    # p = scores / (scores.sum(axis=1, keepdims=True) + 1e-12)
    # entropy = -np.sum(p * np.log(p + 1e-12), axis=1)
    # uncertainty_entropy = entropy / np.log(scores.shape[1])   # normalized [0,1]



    # df_out = pd.concat(
    #     [
    #         df,
    #         module_score_df,
    #         pd.DataFrame({
    #             "uncertainty_margin": uncertainty_margin,
    #             "uncertainty_entropy": uncertainty_entropy
    #         }, index=df.index)
    #     ],
    #     axis=1
    # )

    return df