import warnings
import os 
import xarray as xr
import torch 
import torch.nn as nn 
import torchmetrics.image as t_metrics 

metric_settings = {
    'l1Error': {},
    'l2Error': {},
    'Bias_time': {},
    'Bias_space': {},
    'Corr_time': {},
    'Corr_space': {},
    'RMSE_time': {},
    'RMSE_space': {},
    'PeakSignalNoiseRatio': {'torchmetric_settings': {'data_range': 366.35}},# max = 366.35 / p99 = 27.72 / p90 = 11.94
    'StructuralSimilarityIndexMeasure': {'torchmetric_settings': {}},
    'UniversalImageQualityIndex': {'torchmetric_settings': {}},#{'reset_real_features': True}
    'FrechetInceptionDistance': {'torchmetric_settings': {}},
    'KernelInceptionDistance': {'torchmetric_settings': {}}
    }

def check_metrics(metrics):

    metric_list = metric_settings.keys()

    invalid_metrics = [m for m in metrics if m not in metric_list]
    if invalid_metrics:
        raise ValueError('The given metric(s) ({}) not defined'.format(invalid_metrics))
    
def read_land_map(file_path):

    land_mask = xr.open_dataset(os.path.join(file_path))["mask"].values
    land_mask = torch.tensor(land_mask).to('cpu')

    return land_mask

def metrics(preds, targets, output_metrics_path, land_mask_bool, metrics, epoch):

    #### READ MASK!!!
    if land_mask_bool:
        path = os.path.join(os.getcwd(), 'data', 'CERRA', 'preprocessed_separate', 'aux_variables.nc')
        sl_mask = read_land_map(path)
    else:
        sl_mask = 0
    
    metric_dict = get_metrics(preds, targets, sl_mask, land_mask_bool, metrics)

    print_info = False

    if print_info:
        print()
        print('Metrics: ')
                
        for k, metric in enumerate(metric_dict):
            if 'map' not in metric:
                print(k, '.....', metric, '.....', metric_dict[metric])
        print()
                
    with open(os.path.join(output_metrics_path, 'metrics.csv'), 'a') as f:

        if epoch == 0:
            f.write('Epoch,'); 
            for k, metric in enumerate(metric_dict):
                if 'map' not in metric:
                    f.write(str(metric)); 
                    f.write(",") 
            f.write('\n')

        f.write(str(epoch))
        f.write(",")
        for k, metric in enumerate(metric_dict):
            if 'map' not in metric:
                f.write(str(metric_dict[metric]))
                f.write(",")
        f.write('\n')

    map_metric_dict = {k: v for k, v in metric_dict.items() if "map" in k}
    scalar_metric_dict = {k: v for k, v in metric_dict.items() if "map" not in k}
    
    return map_metric_dict, scalar_metric_dict

@torch.no_grad()
def get_metrics(preds, targets, sl_mask, land_mask_bool, metrics):
    
    metric_dict = {}

    for metric in metrics:

        print('Calculating {}...'.format(metric))

        settings = metric_settings[metric]

        land_mask_bool = False

        if 'l1Error' in metric:
            
            l1_loss_t, l1_loss_land = compute_loss(preds, targets, nn.L1Loss(reduction='mean'), sl_mask, land_mask_bool)

            metric_dict['l1Error_total'] = l1_loss_t
            if land_mask_bool:
                metric_dict['l1Error_land'] = l1_loss_land

        elif 'l2Error' in metric:
            
            l2_loss_t, l2_loss_land = compute_loss(preds, targets, nn.MSELoss(reduction='mean'), sl_mask, land_mask_bool)
            metric_dict['l2Error_total'] = l2_loss_t
            if land_mask_bool:
                metric_dict['l2Error_land'] = l2_loss_land

        elif 'Bias_time' in metric:
            
            bias_map_time, bias_map_time_land, bias_time, bias_time_land = compute_bias_time(preds, targets, sl_mask, land_mask_bool)
            metric_dict['Bias_map_time'] = bias_map_time
            metric_dict['Bias_time'] = bias_time
            if land_mask_bool:
                metric_dict['Bias_map_time_land'] = bias_map_time_land
                metric_dict['Bias_time_land'] = bias_time_land

        elif 'Bias_space' in metric:
            
            bias_map_space, bias_map_space_land, bias_space, bias_space_land = compute_bias_space(preds, targets, sl_mask, land_mask_bool)

            metric_dict['Bias_map_space'] = bias_map_space
            metric_dict['Bias_space'] = bias_space
            if land_mask_bool:
                metric_dict['Bias_map_space_land'] = bias_map_space_land
                metric_dict['Bias_space_land'] = bias_space_land

        elif 'Corr_time' in metric:
            
            correlation_map_time, correlation_map_time_land, correlation_time, correlation_time_land = compute_corr_time(preds, targets, sl_mask, land_mask_bool)

            metric_dict['Corr_map_time'] = correlation_map_time
            metric_dict['Corr_time'] = correlation_time
            if land_mask_bool:
                metric_dict['Corr_map_time_land'] = correlation_map_time_land
                metric_dict['Corr_time_land'] = correlation_time_land

        elif 'Corr_space' in metric:
            
            correlation_map_space, correlation_map_space_land, correlation_space, correlation_space_land = compute_corr_space(preds, targets, sl_mask, land_mask_bool)

            metric_dict['Corr_map_space'] = correlation_map_space
            metric_dict['Corr_space'] = correlation_space
            if land_mask_bool:
                metric_dict['Corr_map_space_land'] = correlation_map_space_land
                metric_dict['Corr_space_land'] = correlation_space_land


        elif 'RMSE_time' in metric:
            
            rmse_map_time, rmse_map_time_land, rmse_time, rmse_time_land = compute_rmse_time(preds, targets, sl_mask, land_mask_bool)

            metric_dict['RMSE_map_time'] = rmse_map_time
            metric_dict['RMSE_time'] = rmse_time
            if land_mask_bool:
                metric_dict['RMSE_map_time_land'] = rmse_map_time_land
                metric_dict['RMSE_time_land'] = rmse_time_land

        elif 'RMSE_space' in metric:
            
            rmse_map_space, rmse_map_space_land, rmse_space, rmse_space_land = compute_rmse_space(preds, targets, sl_mask, land_mask_bool)

            metric_dict['RMSE_map_space'] = rmse_map_space
            metric_dict['RMSE_space'] = rmse_space
            if land_mask_bool:
                metric_dict['RMSE_map_space_land'] = rmse_map_space_land
                metric_dict['RMSE_space_land'] = rmse_space_land


        else:
            metric_outputs = calculate_metric(metric, preds, targets, torchmetrics_settings=settings['torchmetric_settings'])
            metric_dict[f'{metric}'] = metric_outputs

            #if len(metric_outputs) > 1:
            #    for k, metric_name in enumerate(settings['outputs']):
            #        metric_dict['{metric}_{metric_name}'] = metric_outputs[k]
            #else:
            #    metric_dict['{metric}'] = metric_outputs[0]

    return metric_dict

def compute_loss(pred, target, loss_fn, sl_mask, land_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    loss_total = loss_fn(pred, target).item()

    if land_bool:
        # Create a mask to ignore NaN values
        mask = ~torch.isnan(sl_mask.unsqueeze(0).expand_as(pred))

        # Apply masks to vectors
        pred_land = pred[mask]
        target_land = target[mask]

        loss_land = loss_fn(pred_land, target_land).item()
    else:
        loss_land = 0

    return loss_total, loss_land

def compute_bias_time(pred, target, sl_mask, land_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    bias_map_time = torch.mean(pred - target, dim=0)

    # Máscara para España
    if land_bool:
        bias_map_time_land = torch.where(sl_mask.isnan(), float('nan'), bias_map_time)
    else:
        bias_map_time_land = torch.zeros_like(bias_map_time)

    return (bias_map_time, bias_map_time_land, torch.nanmean(bias_map_time).item(), torch.nanmean(bias_map_time_land).item())

def compute_bias_space(pred, target, sl_mask, land_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    # First, the spatial dimension (lat and lon) are planned. Dimensions now is time, lat * lon
    pred_flattened = pred.view(pred.shape[0], -1)
    target_flattened = target.view(pred.shape[0], -1)

    # Calculation of the bias
    bias_map_space = torch.mean(pred_flattened - target_flattened, dim=1)

    # sl_mask
    if land_bool:
        mask = ~torch.isnan(sl_mask.flatten())
        pred_land = pred_flattened[:, mask]
        target_land = target_flattened[:, mask]  

        bias_map_space_land = torch.mean(pred_land - target_land, dim=1)
    else:
        bias_map_space_land = torch.zeros_like(bias_map_space)
                
    return bias_map_space, bias_map_space_land, torch.nanmean(bias_map_space).item(), torch.nanmean(bias_map_space_land).item()
  
def compute_corr_time(pred, target, sl_mask, land_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    # Calculation of the Correlation: 
    # r = Cov(X, Y)/(sigma_x * sigma_y)
    
    # Calcule the means
    pred_mean  = torch.mean(pred, dim=0)
    target_mean = torch.mean(target, dim=0)

    # Calcule the Covariance and standard deviations
    covariance = torch.mean(pred * target, dim=0) - pred_mean * target_mean
    pred_var = torch.mean(pred ** 2, dim=0) - pred_mean ** 2
    target_var = torch.mean(target ** 2, dim=0) - target_mean ** 2

    # Correlation
    corr_map_time = covariance / (torch.sqrt(pred_var) * torch.sqrt(target_var))

    # Check for pixels with standard variance approximately 0, i.e., correlation values >1 or <-1
    corr_map_time[corr_map_time>1] = float('nan')
    corr_map_time[corr_map_time<-1] = float('nan')

    # sl_mask
    if land_bool:
        corr_map_time_land = torch.where(sl_mask.isnan(), float('nan'), corr_map_time)
    else:
        corr_map_time_land = torch.zeros_like(corr_map_time)

    return corr_map_time, corr_map_time_land, torch.nanmean(corr_map_time).item(), torch.nanmean(corr_map_time_land).item()

def compute_corr_space(pred, target, sl_mask, land_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    # First, the spatial dimension (lat and lon) are planned. Dimensions now is time, lat * lon
    pred_flattened = pred.view(pred.shape[0], -1)
    target_flattened = target.view(pred.shape[0], -1)

    # Calculation of the Means
    pred_mean = torch.mean(pred_flattened, dim=1)
    target_mean = torch.mean(target_flattened, dim=1)

    # Calculation of covariance and standard deviations
    covariance = torch.mean(pred_flattened * target_flattened, dim=1) - pred_mean * target_mean
    pred_var = torch.mean(pred_flattened ** 2, dim=1) - pred_mean ** 2
    target_var = torch.mean(target_flattened ** 2, dim=1) - target_mean ** 2

    # Correlation
    corr_map_space = covariance / (torch.sqrt(pred_var) * torch.sqrt(target_var))
    corr_map_space[torch.isnan(corr_map_space)] = 0

    # sl_mask
    if land_bool:
        mask = ~torch.isnan(sl_mask.flatten())
        pred_land = pred_flattened[:, mask]
        target_land = target_flattened[:, mask]

        # Calculation of the Means
        pred_mean_land = torch.mean(pred_land, dim=1)
        target_mean_land = torch.mean(target_land, dim=1)

        # Calculation of covariance and standard deviations
        covariance_land = torch.mean(pred_land * target_land, dim=1) - pred_mean_land * target_mean_land
        pred_var_land = torch.mean(pred_land ** 2, dim=1) - pred_mean_land ** 2
        target_var_land = torch.mean(target_land ** 2, dim=1) - target_mean_land ** 2

        # Correlation
        corr_map_space_land = covariance_land / (torch.sqrt(pred_var_land) * torch.sqrt(target_var_land))
        corr_map_space_land[torch.isnan(corr_map_space_land)] = 0    
    else:
        corr_map_space_land = torch.zeros_like(corr_map_space)
                
    return corr_map_space, corr_map_space_land, torch.nanmean(corr_map_space).item(), torch.nanmean(corr_map_space_land).item()

def compute_rmse_time(pred, target, sl_mask, land_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    # Calculation of the Full RMSE
    rmse_map_time = torch.sqrt(torch.mean((pred - target) ** 2, dim=0))

    # sl_mask
    if land_bool:
        rmse_map_time_land = torch.where(sl_mask.isnan(), float('nan'), rmse_map_time)
    else:
        rmse_map_time_land = torch.zeros_like(rmse_map_time)

    return rmse_map_time, rmse_map_time_land, torch.nanmean(rmse_map_time).item(), torch.nanmean(rmse_map_time_land).item() 

def compute_rmse_space(pred, target, sl_mask, land_bool):

    assert pred.shape == target.shape, "The prediction and target dimensions are different"

    # First, the spatial dimension (lat and lon) are planned. Dimensions now is time, lat * lon
    pred_flattened = pred.view(pred.shape[0], -1)
    target_flattened = target.view(pred.shape[0], -1)

    # Calculation of the RMSE
    rmse_map_space = torch.sqrt(torch.mean((pred_flattened - target_flattened) ** 2, dim=1))

    # sl_mask
    if land_bool:
        mask = ~torch.isnan(sl_mask.flatten())
        pred_land = pred_flattened[:, mask]
        target_land = target_flattened[:, mask]

        rmse_map_space_land = torch.sqrt(torch.mean((pred_land - target_land) ** 2, dim=1))
    else:
        rmse_map_space_land = torch.zeros_like(rmse_map_space)
                
    return rmse_map_space, rmse_map_space_land, torch.nanmean(rmse_map_space).item(), torch.nanmean(rmse_map_space_land).item()

def calculate_metric(name_expr, pred, target, torchmetrics_settings={}, part=5000):

    metric_str = [m for m in t_metrics.__dict__.keys() if (name_expr == m)]

    if len(metric_str) == 0:
        metric_str = [m for m in t_metrics.__dict__.keys() if (name_expr in m)]
        if len(metric_str) > 1:
            warnings.warn('found multiple hits for metric name {}. Will use {}'.format(name_expr, metric_str[0]))

    assert len(metric_str) > 0, 'metric {} not found in torchmetrics.image. Maybe torch-fidelity is missing.'.format(name_expr)

    metric = t_metrics.__dict__[metric_str[0]](**torchmetrics_settings)

    total_sum = 0.0
    total_batches = 0

    for i in range(0, pred.size(0), part):

        batch_preds = pred[i:min(i + part, pred.size(0))]
        batch_targets = target[i:min(i + part, pred.size(0))]
    
        with torch.no_grad():
            value = metric(batch_preds.unsqueeze(1), batch_targets.unsqueeze(1)).item()
            total_sum += value
            total_batches += 1
    
    del batch_preds, batch_targets

    return total_sum/total_batches
