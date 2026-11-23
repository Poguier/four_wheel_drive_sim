"""轮毂/减速电机模型: 电流环 + 电气动力学 + 反馈传感器 (电流采样、编码器测速)。

信号流:
    控制器 --i_ref--> 电流环(PI) --电压--> 电机电气方程 --电流 i--> 力矩 = Kt*i*gear
                                              ^ 反电动势 Ke*ω_motor (来自 MuJoCo 车轮转速)
    反馈: measured_current (ADC: 噪声+偏置+量化), measured_speed (编码器: 计数量化)
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class MotorParams:
    R: float = 0.3            # 绕组电阻 [Ω]
    L: float = 1e-3           # 绕组电感 [H]
    Kt: float = 0.25          # 力矩常数 [Nm/A]
    Ke: float = 0.25          # 反电动势常数 [V/(rad/s)]
    gear: float = 5.0         # 减速比 (电机转速 = gear * 车轮转速)
    eta: float = 0.92         # 传动效率
    V_bus: float = 48.0       # 母线电压 [V]
    I_max: float = 40.0       # 电流限幅 [A]
    current_bw: float = 2 * math.pi * 300  # 电流环带宽 [rad/s]
    inner_dt: float = 2e-4    # 电流环/电气方程积分步长 [s]
    # 反馈传感器
    adc_lsb: float = 0.05     # 电流采样分辨率 [A]
    adc_noise: float = 0.15   # 电流采样噪声标准差 [A]
    adc_offset: float = 0.0   # 电流采样零偏 [A]
    encoder_cpr: int = 4096   # 电机轴编码器线数 (每转计数)


class Motor:
    def __init__(self, params: MotorParams | None = None, seed: int | None = None):
        self.p = params or MotorParams()
        self.rng = np.random.default_rng(seed)
        self.i = 0.0          # 真实相电流 (q 轴) [A]
        self._int = 0.0       # 电流环积分
        self._enc_prev = None
        self.measured_current = 0.0
        self.measured_speed = 0.0   # 车轮转速估计 [rad/s]
        self.torque = 0.0           # 施加在车轮上的力矩 [Nm]

    def reset(self):
        self.i = self._int = 0.0
        self._enc_prev = None

    def step(self, i_ref: float, wheel_speed: float, wheel_angle: float, dt: float) -> float:
        """推进一个物理步长, 返回应写入 MuJoCo 力矩执行器的车轮力矩 [Nm]。

        wheel_speed/wheel_angle 是 MuJoCo 中车轮关节的真实转速/转角。
        """
        p = self.p
        i_ref = float(np.clip(i_ref, -p.I_max, p.I_max))
        w_m = wheel_speed * p.gear
        kp = p.L * p.current_bw
        ki = p.R * p.current_bw

        n = max(1, round(dt / p.inner_dt))
        h = dt / n
        i_avg = 0.0
        for _ in range(n):
            err = i_ref - self.i
            v_cmd = kp * err + self._int + p.Ke * w_m  # 带反电动势前馈
            v = float(np.clip(v_cmd, -p.V_bus, p.V_bus))
            if v == v_cmd:  # 未饱和才积分 (抗积分饱和)
                self._int += ki * err * h
            # L di/dt = V - R i - Ke ω
            self.i += (v - p.R * self.i - p.Ke * w_m) / p.L * h
            i_avg += self.i / n

        self.torque = p.Kt * i_avg * p.gear * p.eta
        self._sample(i_avg, wheel_angle, dt)
        return self.torque

    def _sample(self, i_true: float, wheel_angle: float, dt: float):
        p = self.p
        raw = i_true + p.adc_offset + self.rng.normal(0.0, p.adc_noise)
        self.measured_current = round(raw / p.adc_lsb) * p.adc_lsb

        # 编码器在电机轴: 计数量化, M 法测速
        count = math.floor(wheel_angle * p.gear * p.encoder_cpr / (2 * math.pi))
        if self._enc_prev is not None:
            d_count = count - self._enc_prev
            self.measured_speed = d_count * 2 * math.pi / p.encoder_cpr / p.gear / dt
        self._enc_prev = count
