import torch as tr
import numpy as np


import logging

log = logging.getLogger(__name__)


def convert_log_center_to_original(data, std, mean):
    data = data*std
    data = data + mean
    data = 10**data
    data = data - 1e-7
    return data


def convert_log_center_to_db(data, std, mean):
    data = data*std
    data = data + mean
    data = data*10
    data = data
    return data


def _prep_data(target, pred):
    """Given tensors with multiple viewing geometries, flatters viewing geometries into batch dimension to simplify computation

    Args:
        target (Tensor): target radar response. Can be single datapoint or batched
        pred (Tensor): predicted radar response. Can be single datapoint or batched

    """
    assert tr.is_tensor(target), 'Target is not a tensor'
    assert tr.is_tensor(pred), 'Pred is not a tensor'

    if len(target.size()) == 1:
        t = target[None, :].clone().detach()
    elif len(target.size()) == 2:
        t = target.clone().detach()
    elif len(target.size()) == 3:
        t = target.reshape((-1, target.size()[-1])).clone().detach()

    if len(pred.size()) == 1:
        p = pred[None, :].clone().detach()
    elif len(pred.size()) == 2:
        p = pred.clone().detach()
    elif len(pred.size()) == 3:
        p = pred.reshape((-1, pred.size()[-1])).clone().detach()

    return t, p


def calculated_base_matching_score(target, pred, num_low_points=30, reduction='mean'):
    """Calculates alignment between target and predicted response values that are close to base value of response.
    Works by averaging a set number of the lowest value points in the target and predicted response.

    Args:
        target (Tensor): target radar response. Can be single datapoint or batched
        pred (Tensor): predicted radar response. Can be single datapoint or batched
        num_low_points (int, optional): Number of low points
        reduction (str, optional): reduction method for scor. Defaults to 'mean'.
    """

    target, pred = _prep_data(target, pred)

    # get low values
    low_target_vals, _ = tr.topk(
        target, k=num_low_points, largest=False, dim=-1)
    low_pred_vals, _ = tr.topk(pred, k=num_low_points, largest=False, dim=-1)

    # average low values to get base for each response
    target_base_vals = low_target_vals.mean(dim=-1)
    pred_base_vals = low_pred_vals.mean(dim=-1)

    # compute metric
    base_val_diff = tr.nn.functional.l1_loss(
        target_base_vals, pred_base_vals, reduction=reduction)

    return base_val_diff


def peak_region_matching_score(target, pred, window_size=3, reduction='mean'):
    """Calculates alignment between target and predicted response values that are close to base value of response.
    Works by averaging a set number of the lowest value points in the target and predicted response.

    Args:
        target (Tensor): target radar response. Can be single datapoint or batched
        pred (Tensor): predicted radar response. Can be single datapoint or batched
        num_low_points (int, optional): Number of low points
        reduction (str, optional): reduction method for scor. Defaults to 'mean'.
    """

    target, pred = _prep_data(target, pred)
    batch_size, num_range_windows = target.size()[0], target.size()[1]

    peak_bins = tr.argmax(target, dim=-1)
    peak_windows_low = tr.clamp(
        peak_bins - window_size, 0, num_range_windows)[:, None]
    peak_windows_high = tr.clamp(
        peak_bins + window_size, 0, num_range_windows)[:, None]

    indices = tr.arange(num_range_windows).repeat(
        batch_size).reshape(batch_size, -1)
    response_mask = tr.where(tr.logical_and(
        indices >= peak_windows_low, indices <= peak_windows_high), 1, 0)
    num_comparisons = response_mask.sum()

    masked_target = response_mask*target
    masked_pred = response_mask*pred

    peak_region_error = tr.nn.functional.l1_loss(
        masked_target.float(), masked_pred.float(), reduction='sum')/num_comparisons

    return peak_region_error

# highest point matching metric


def maxima_matching_score(target, pred, k=1, reduction='mean'):
    """Calculates alignment between target and predicted response using a naive comparison of global maxima
    based on magnitude. Comparison is naive since it aligns peaks based on magnitude, not bin location or a more
    sophisticated method.

    Args:
        target (Tensor): target radar response. Can be single datapoint or batched
        pred (_type_): predicted radar response. Can be single datapoint or batched
        k (int, optional): Number of points to compare in each sample. As this increases, the chances of getting 
        non peak/undesirable points increases. Defaults to 1.
        reduction (str, optional): reduction method for scor. Defaults to 'mean'.
    """

    target, pred = _prep_data(target, pred)

    target_values, target_indices = tr.topk(target.detach(), k=k, dim=-1)
    pred_values, pred_indices = tr.topk(pred.detach(), k=k, dim=-1)

    val_diff = tr.nn.functional.l1_loss(
        target_values, pred_values, reduction=reduction)
    bin_diff = tr.nn.functional.l1_loss(
        target_indices.float(), pred_indices.float(), reduction=reduction)

    return val_diff, bin_diff


def calculated_peak_matching_score(target, pred, max_num_peaks=1, search_size=10, reduction='mean'):
    """Calculates alignment between target and predicted response using a naive comparison of local maxima (peaks)
    based on magnitude. Comparison is naive since it aligns peaks based on magnitude, not bin location or a more
    sophisticated method.

    Args:
        target (Tensor): target radar response. Can be single datapoint or batched
        pred (_type_): predicted radar response. Can be single datapoint or batched
        max_num_peaks (int, optional): Maximum number of peaks to compare in each sample. May compare fewer
        peaks a sample if the target or predicted response has fewer peaks than this value. Defaults to 1.
        search_size (int, optional): Number of points to consider when checking for peaks. Defaults to 10.
        reduction (str, optional): reduction method for scor. Defaults to 'mean'.
    """

    target, pred = _prep_data(target, pred)

    # get peak candidates from target and prediction
    target_values, target_indices = tr.topk(
        target.detach(), k=search_size, dim=-1)
    pred_values, pred_indices = tr.topk(pred.detach(), k=search_size, dim=-1)

    min_target_vals, _ = target.min(dim=-1)
    min_pred_vals, _ = pred.min(dim=-1)

    # get values to the left and right of each peak candidate
    target_lshift = tr.cat([target[:, 1:], min_target_vals[:, None]], dim=-1)
    target_rshift = tr.cat([min_target_vals[:, None], target[:, :-1]], dim=-1)

    pred_lshift = tr.cat([pred[:, 1:], min_pred_vals[:, None]], dim=-1)
    pred_rshift = tr.cat([min_pred_vals[:, None], pred[:, :-1]], dim=-1)

    i, j = np.diag_indices(target.shape[0], ndim=2)
    target_values_lshift = target_lshift[:, target_indices][i, j, :]
    target_values_rshift = target_rshift[:, target_indices][i, j, :]

    pred_values_lshift = pred_lshift[:, pred_indices][i, j, :]
    pred_values_rshift = pred_rshift[:, pred_indices][i, j, :]

    # determine which peak candidates are local maxima, which is how I'm defining peaks for now
    target_peak_masks = tr.where(tr.logical_and(
        target_values > target_values_lshift, target_values > target_values_rshift), 1, 0).bool()
    pred_peak_masks = tr.where(tr.logical_and(
        pred_values > pred_values_lshift, pred_values > pred_values_rshift), 1, 0).bool()

    all_target_peaks_bins = []
    all_target_peak_values = []
    all_pred_peaks_bins = []
    all_pred_peak_values = []

    # select peak values and bin locations using peak masks
    # have to loop since different rti pairs not guaranteed to have the same number of peaks
    for idx in range(target.shape[0]):

        target_peaks_bins = target_indices[idx][target_peak_masks[idx]]
        target_peak_values = target_values[idx][target_peak_masks[idx]]

        pred_peaks_bins = pred_indices[idx][pred_peak_masks[idx]]
        pred_peak_values = pred_values[idx][pred_peak_masks[idx]]

        # restrict number
        max_num_peaks = min(
            max_num_peaks, target_peaks_bins.shape[-1], pred_peaks_bins.shape[-1])

        all_target_peaks_bins.append(target_peaks_bins[:max_num_peaks])
        all_target_peak_values.append(target_peak_values[:max_num_peaks])
        all_pred_peaks_bins.append(pred_peaks_bins[:max_num_peaks])
        all_pred_peak_values.append(pred_peak_values[:max_num_peaks])

    target_peaks_bins = tr.cat(all_target_peaks_bins)
    target_peak_values = tr.cat(all_target_peak_values)
    pred_peaks_bins = tr.cat(all_pred_peaks_bins)
    pred_peak_values = tr.cat(all_pred_peak_values)

    # compute difference between
    val_diff = tr.nn.functional.l1_loss(
        target_peak_values, pred_peak_values, reduction=reduction)
    bin_diff = tr.nn.functional.l1_loss(
        target_peaks_bins.float(), pred_peaks_bins.float(), reduction=reduction)

    return val_diff, bin_diff
