import numpy as np

def calculate_ap_trapezoidal(recall, precision):
    """
    Calculate Average Precision (AP) using the Trapezoidal Rule for numerical integration.

    This method corresponds to Week 6 (Newton-Cotes Integration) of MMA3001.

    Args:
        recall (np.ndarray): 1D array of recall values (x-axis), sorted in ascending order. Must be within [0, 1].
        precision (np.ndarray): 1D array of precision values (y-axis) corresponding to each recall value. Must be within [0, 1].

    Returns:
        float: The calculated area under the PR curve (AP).

    Raises:
        ValueError: If input arrays have different lengths or contain values outside [0, 1].
        ValueError: If recall array is not sorted in ascending order.

    Examples:
        >>> r = np.array([0.0, 0.5, 1.0])
        >>> p = np.array([1.0, 0.8, 0.0])
        >>> calculate_ap_trapezoidal(r, p)
        0.65
    """
    r = np.asarray(recall)
    p = np.asarray(precision)

    if len(r) != len(p):
        raise ValueError("Recall and Precision arrays must have the same length.")
    if not np.all(np.diff(r) >= 0):
        raise ValueError("Recall array must be sorted in ascending order.")
    if np.any(r < 0) or np.any(r > 1) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Recall and Precision values must be between 0 and 1.")

    # 使用 NumPy 的梯形积分
    ap = float(np.trapezoid(p, r))
    return ap

def calculate_ap_riemann(recall, precision):
    """
    Calculate AP using the Left Riemann Sum (Rectangular rule).
    """
    r = np.asarray(recall)
    p = np.asarray(precision)

    if len(r) != len(p):
        raise ValueError("Recall and Precision arrays must have the same length.")

    # 黎曼和 (左矩形法则)
    ap = float(np.sum(np.diff(r) * p[:-1]))
    return ap