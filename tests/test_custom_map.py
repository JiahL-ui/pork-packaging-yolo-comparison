import pytest
import numpy as np
from src.custom_map import calculate_ap_trapezoidal, calculate_ap_riemann

def test_perfect_detection():
    """测试完美检测的情况（面积应为1.0）"""
    r = np.array([0.0, 0.5, 1.0])
    p = np.array([1.0, 1.0, 1.0])
    assert calculate_ap_trapezoidal(r, p) == pytest.approx(1.0)
    assert calculate_ap_riemann(r, p) == pytest.approx(1.0)

def test_zero_detection():
    """测试完全没检测到的情况（面积应为0.0）"""
    r = np.array([0.0, 0.5, 1.0])
    p = np.array([0.0, 0.0, 0.0])
    assert calculate_ap_trapezoidal(r, p) == pytest.approx(0.0)

def test_custom_values():
    """测试一组已知计算结果的数据"""
    r = np.array([0.0, 0.5, 1.0])
    p = np.array([1.0, 0.8, 0.0])
    # 梯形积分: (0.5-0)*(1.0+0.8)/2 + (1.0-0.5)*(0.8+0.0)/2 = 0.45 + 0.2 = 0.65
    assert calculate_ap_trapezoidal(r, p) == pytest.approx(0.65)
    # 矩形积分: 0.5*1.0 + 0.5*0.8 = 0.9
    assert calculate_ap_riemann(r, p) == pytest.approx(0.9)

def test_invalid_inputs():
    """测试错误输入是否会正确抛出异常"""
    with pytest.raises(ValueError):
        # 长度不匹配
        calculate_ap_trapezoidal(np.array([0.1, 0.2]), np.array([1.0]))

    with pytest.raises(ValueError):
        # recall 没有排序
        calculate_ap_trapezoidal(np.array([0.5, 0.0, 1.0]), np.array([0.8, 1.0, 0.0]))

    with pytest.raises(ValueError):
        # precision 超出合法范围
        calculate_ap_trapezoidal(np.array([0.0, 1.0]), np.array([1.1, 0.0]))