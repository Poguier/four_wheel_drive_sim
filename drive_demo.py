"""四轮车仿真: 直线加速, 中途右前轮突然失去抓地力。

用法:
    python drive_demo.py            # 默认自动打开 MuJoCo viewer, 结束后窗口保持, 空格键重播
    python drive_demo.py --headless # 无界面, 打印四轮电流/转速/滑移率反馈
    python drive_demo.py --headless --log run.csv  # 同时记录每步数据
"""
import argparse
import math
import os
import sys
import time

import mujoco
import numpy as np

from motor import MotorParams, Motor

I_MAX = MotorParams().I_max

WHEELBASE = 1.2
TRACK = 0.78
R_FRONT = 0.133
R_REAR = 0.170

HERE = os.path.dirname(os.path.abspath(__file__))


def wheel_commands(v, steer):
    """车体中心速度 v (m/s), 等效前轮转角 steer (rad, 左转为正)。

    返回 (转向角 fl, fr, 轮速 fl, fr, rl, rr [rad/s])。
    以后轴中心为参考点, 转弯半径 R = L / tan(steer)。
    """
    L, T = WHEELBASE, TRACK
    if abs(steer) < 1e-6:
        return 0.0, 0.0, v / R_FRONT, v / R_FRONT, v / R_REAR, v / R_REAR

    R = L / math.tan(steer)  # 后轴中心转弯半径 (左转为正)
    d_fl = math.atan(L / (R - T / 2))
    d_fr = math.atan(L / (R + T / 2))
    k = v / R  # 车体角速度
    # 各轮中心相对瞬心的距离 -> 线速度 (前轮沿滚动方向, 速度符号与 v 一致)
    v_rl = k * (R - T / 2)
    v_rr = k * (R + T / 2)
    v_fl = v * math.hypot(R - T / 2, L) / abs(R)
    v_fr = v * math.hypot(R + T / 2, L) / abs(R)
    return (d_fl, d_fr,
            v_fl / R_FRONT, v_fr / R_FRONT,
            v_rl / R_REAR, v_rr / R_REAR)


WHEELS = ("fl", "fr", "rl", "rr")


class Car:
    """把 MuJoCo 模型、四个电机模型和速度环绑在一起。"""

    def __init__(self, model, data, kp=5.0, ki=20.0):
        self.model, self.data = model, data
        self.motors = {w: Motor(seed=i) for i, w in enumerate(WHEELS)}
        self.kp, self.ki = kp, ki
        self.integ = dict.fromkeys(WHEELS, 0.0)
        self.i_ref = dict.fromkeys(WHEELS, 0.0)
        self.act = {n: model.actuator(n).id for n in (
            [f"{w}_drive_motor" for w in WHEELS] + ["fl_steer_motor", "fr_steer_motor"])}
        self.qadr = {w: model.joint(f"{w}_drive").qposadr[0] for w in WHEELS}
        self.dadr = {w: model.joint(f"{w}_drive").dofadr[0] for w in WHEELS}
        # 用于计算真实滑移率的车轮航向参考体 (前轮跟随转向节)
        self.ref_body = {w: model.body(f"{w}_knuckle" if w[0] == "f" else "chassis").id for w in WHEELS}
        self.hub_body = {w: model.body(f"{w}_wheel").id for w in WHEELS}
        self.radius = dict(fl=R_FRONT, fr=R_FRONT, rl=R_REAR, rr=R_REAR)

    # ---- 反馈量 (控制器只能用这些) ----
    def feedback(self):
        cur = {w: m.measured_current for w, m in self.motors.items()}
        spd = {w: m.measured_speed for w, m in self.motors.items()}
        return cur, spd

    # ---- 仅用于分析的真值 ----
    def slip_ratio(self):
        out, res = {}, np.zeros(6)
        for w in WHEELS:
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                     self.hub_body[w], res, 0)
            heading = self.data.xmat[self.ref_body[w]].reshape(3, 3)[:, 0]
            v_long = float(res[3:] @ heading)
            v_wheel = self.data.qvel[self.dadr[w]] * self.radius[w]
            out[w] = (v_wheel - v_long) / max(abs(v_wheel), abs(v_long), 0.1)
        return out

    def speed_controller(self, w, target_speed, measured_speed, dt):
        """车轮转速 PI -> 电流指令。牵引力控制算法可在此处替换/叠加。"""
        err = target_speed - measured_speed
        i_unsat = self.kp * err + self.integ[w]
        i_ref = float(np.clip(i_unsat, -I_MAX, I_MAX))
        if i_ref == i_unsat:
            self.integ[w] += self.ki * err * dt
        return i_ref

    def step(self, v, steer, i_cmd=None):
        """i_cmd 不为 None 时, 四轮直接按该电流指令 (开环油门) 驱动, 否则走转速环。"""
        model, data = self.model, self.data
        dt = model.opt.timestep
        d_fl, d_fr, *w_target = wheel_commands(v, steer)
        data.ctrl[self.act["fl_steer_motor"]] = d_fl
        data.ctrl[self.act["fr_steer_motor"]] = d_fr
        for w, target in zip(WHEELS, w_target):
            m = self.motors[w]
            if i_cmd is None:
                self.i_ref[w] = self.speed_controller(w, target, m.measured_speed, dt)
            else:
                self.i_ref[w] = i_cmd
            torque = m.step(self.i_ref[w], data.qvel[self.dadr[w]], data.qpos[self.qadr[w]], dt)
            data.ctrl[self.act[f"{w}_drive_motor"]] = torque
        mujoco.mj_step(model, data)


GRIP_LOSS_TIME = 1.0   # 右前轮失去抓地力的时刻 [s]
GRIP_LOSS_MU = 0.00    # 失去抓地力后右前轮的摩擦系数 (正常为 1.0)
DRIVE_CURRENT = 4.0    # 每个电机的电流指令 [A]


def schedule(t):
    """(速度 m/s, 转角 rad, 电流指令 A): 直行, 四轮相同的开环电流指令 (相当于固定油门)。

    返回的速度仅在 i_cmd=None 时 (转速环模式) 使用。
    """
    return 0.0, 0.0, DRIVE_CURRENT


def lose_grip(model, geom_name, mu):
    """让指定轮胎骤然打滑 (如踩到冰面)。

    MuJoCo 接触摩擦默认取两个几何体中较大的值, 地面 mu=1 会盖过轮胎,
    所以同时提高该轮胎的 priority, 让它的摩擦参数优先生效。
    """
    gid = model.geom(geom_name).id
    model.geom_priority[gid] = 1
    model.geom_friction[gid, 0] = mu
    model.geom_rgba[gid] = [0.9, 0.1, 0.1, 1]  # 变红, 便于在 viewer 中观察


class OffsetOverlay:
    """在 viewer 中绘制行驶轨迹、到中心线的偏移线, 并叠加偏移量文字。"""

    TRAIL_DT = 0.05  # 轨迹采样间隔 [s]

    def __init__(self, viewer, model, data):
        self.viewer, self.model, self.data = viewer, model, data
        self.trail = []
        self.next_t = 0.0

    def reset(self):
        self.trail.clear()
        self.next_t = 0.0

    def update(self):
        d = self.data
        x, y = d.qpos[0], d.qpos[1]
        if d.time >= self.next_t:
            self.trail.append((x, y))
            self.next_t = d.time + self.TRAIL_DT

        w, qx, qy, qz = d.qpos[3:7]
        yaw = math.degrees(math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2)))
        side = "left" if y > 0 else "right"
        text = (f"{y * 100:+7.1f} cm ({side})", f"{yaw:+6.2f} deg")

        with self.viewer.lock():
            scn = self.viewer.user_scn
            scn.ngeom = 0
            pts = self.trail + [(x, y)]
            for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                self._line(scn, (x0, y0), (x1, y1), 0.008, (0.1, 0.4, 1.0, 1.0))
            if abs(y) > 1e-3:  # 偏移线画在车头前方 (否则被车身挡住): 中心线 -> 车所在横向位置
                xa = x + 1.5
                self._line(scn, (xa, y), (xa, 0.0), 0.012, (1.0, 0.6, 0.0, 1.0))
        # 必须在释放锁之后再调用: set_texts 需要 UI 线程配合, 持锁调用会与 UI 线程互相等待而死锁
        self.viewer.set_texts((
            mujoco.mjtFontScale.mjFONTSCALE_200, mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "Lateral offset\nYaw", "\n".join(text)))

    @staticmethod
    def _line(scn, p0, p1, width, rgba):
        if scn.ngeom >= scn.maxgeom or (p0 == p1):
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.zeros(9), np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.array([p0[0], p0[1], 0.004]), np.array([p1[0], p1[1], 0.004]))
        scn.ngeom += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="不打开 viewer")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--log", help="把每步反馈数据写入 CSV 文件")
    args = parser.parse_args()

    # macOS 的 viewer 必须在 mjpython 下运行, 这里自动切换
    if not args.headless and sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
        mjpython = os.path.join(os.path.dirname(sys.executable), "mjpython")
        if os.path.exists(mjpython):
            os.execv(mjpython, [mjpython] + sys.argv)
        sys.exit("找不到 mjpython, 请用 mjpython 运行, 或加 --headless")

    model = mujoco.MjModel.from_xml_path(os.path.join(HERE, "four_wheel_car.xml"))
    data = mujoco.MjData(model)
    car = Car(model, data)
    log = []

    grip_lost = False

    def tick():
        nonlocal grip_lost
        if not grip_lost and data.time >= GRIP_LOSS_TIME:
            lose_grip(model, "fr_tire", GRIP_LOSS_MU)
            grip_lost = True
            print(f"t={data.time:.1f}s: 右前轮失去抓地力 (mu -> {GRIP_LOSS_MU})")
        car.step(*schedule(data.time))
        if args.log:
            cur, spd = car.feedback()
            slip = car.slip_ratio()
            log.append([data.time, *(cur[w] for w in WHEELS), *(spd[w] for w in WHEELS),
                        *(car.i_ref[w] for w in WHEELS), *(slip[w] for w in WHEELS)])

    if not args.headless:
        from mujoco import viewer as mj_viewer

        tire = model.geom("fr_tire").id
        orig = (model.geom_priority[tire], model.geom_friction[tire].copy(), model.geom_rgba[tire].copy())
        restart_requested = False

        def on_key(keycode):
            nonlocal restart_requested
            if keycode == 32:  # 空格
                restart_requested = True

        def restart():
            nonlocal car, grip_lost
            mujoco.mj_resetData(model, data)
            model.geom_priority[tire], model.geom_friction[tire], model.geom_rgba[tire] = orig
            car = Car(model, data)
            grip_lost = False

        with mj_viewer.launch_passive(model, data, key_callback=on_key) as viewer:
            overlay = OffsetOverlay(viewer, model, data)
            # 车后方追踪视角, 方便观察偏离中心线的情况
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body("chassis").id
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = 0, -50, 2.8
            start = time.perf_counter()
            announced = False
            while viewer.is_running():
                if restart_requested:
                    restart_requested = announced = False
                    with viewer.lock():  # 重置数据时不能与 UI 渲染并发
                        restart()
                        overlay.reset()
                    start = time.perf_counter()
                if data.time < args.duration:
                    tick()
                    if round(data.time / model.opt.timestep) % 10 == 0:
                        overlay.update()
                        viewer.sync()
                    ahead = data.time - (time.perf_counter() - start)
                    if ahead > 0:
                        time.sleep(ahead)
                else:
                    # 仿真结束后保持窗口, 可自由旋转/缩放查看; 空格重播, 关闭窗口退出
                    if not announced:
                        overlay.update()
                        print(f"仿真结束 (最终横向偏移 {data.qpos[1] * 100:+.1f} cm): 按空格键重播, 关闭窗口退出")
                        announced = True
                    viewer.sync()
                    time.sleep(0.02)
        return

    while data.time < args.duration:
        tick()
        if round(data.time / model.opt.timestep) % 500 == 0:
            report(car, data)

    save_log(args.log, log)


def report(car, data):
    cur, spd = car.feedback()
    slip = car.slip_ratio()
    w, qx, qy, qz = data.qpos[3:7]
    yaw = math.degrees(math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2)))
    f = lambda d, fmt: " ".join(fmt.format(d[w]) for w in WHEELS)
    print(f"t={data.time:4.1f}s yaw={yaw:6.2f}° y={data.qpos[1]:6.3f}m  I[A]={f(cur, '{:6.2f}')} | w[rad/s]={f(spd, '{:6.2f}')} | slip={f(slip, '{:6.3f}')}")


def save_log(path, rows):
    if not path or not rows:
        return
    header = ["t"] + [f"{k}_{w}" for k in ("i_meas", "w_meas", "i_ref", "slip") for w in WHEELS]
    np.savetxt(path, np.array(rows), delimiter=",", header=",".join(header), comments="")
    print(f"已保存 {len(rows)} 行到 {path}")


if __name__ == "__main__":
    main()
