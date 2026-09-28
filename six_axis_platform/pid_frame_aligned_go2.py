import numpy as np
import argparse
import math
import serial
import time
import csv
from datetime import datetime
from pathlib import Path


class PIDController:
    def __init__(self, kp, ki, kd, output_limit=None, integral_limit=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit

        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, target, current):
        now = time.time()
        dt = now - self.last_time
        if dt <= 1e-6:
            dt = 0.01

        error = target - current
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        self.last_time = now

        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        return output


class StewartPlatform:
    def __init__(self, params):
        self.a = float(params['a'])
        self.s = float(params['s'])

        self.base_angles_rad = self._get_corrected_angles(params['base_bias'])
        self.b = self._generate_coords(params['base_radius'], self.base_angles_rad)

        self.platform_angles_rad = self._get_corrected_angles(params['top_bias'])
        self.p = self._generate_coords(params['top_radius'], self.platform_angles_rad)

        self.beta = np.array([
            a + (math.pi / 2 if i % 2 == 0 else -math.pi / 2)
            for i, a in enumerate(self.base_angles_rad)
        ])

        self.us_per_rad = params.get('us_per_rad', 636.6)
        self.offsets = params.get('servo_offsets', [0] * 6)

        dist_h = params['base_radius'] - params['top_radius'] - self.a
        self.h0 = math.sqrt(max(0, self.s ** 2 - dist_h ** 2))

        self.alpha0 = self.solve_inverse_kinematics([0, 0, 0], [0, 0, 0], is_setup=True)

    def _get_corrected_angles(self, bias_deg):
        offset = 60.0 - bias_deg
        angles_deg = [
            30 + offset, 150 - offset, 150 + offset,
            270 - offset, 270 + offset, 30 - offset
        ]
        return [math.radians(a) for a in angles_deg]

    def _generate_coords(self, radius, angles_rad):
        return np.array([
            [radius * math.cos(a), radius * math.sin(a), 0]
            for a in angles_rad
        ])

    def solve_inverse_kinematics(self, translation, rotation, is_setup=False):
        """
        rotation = [yaw, pitch, roll]
        当前这里只用了 pitch(Y) 和 roll(X)
        """
        T = np.array([translation[0], translation[1], self.h0 + translation[2]])

        pitch = rotation[1]
        roll = rotation[2]

        c_p, s_p = math.cos(pitch), math.sin(pitch)
        c_r, s_r = math.cos(roll), math.sin(roll)

        R = np.array([
            [c_p,     s_p * s_r,  s_p * c_r],
            [0,       c_r,       -s_r],
            [-s_p,    c_p * s_r,  c_p * c_r]
        ])

        alphas = []
        for i in range(6):
            q_i = T + R @ self.p[i]
            l_vec = q_i - self.b[i]
            x, y, z = l_vec

            L = (x**2 + y**2 + z**2) - (self.s**2 - self.a**2)
            M = 2 * self.a * z
            N = 2 * self.a * (math.cos(self.beta[i]) * x + math.sin(self.beta[i]) * y)

            denom = math.sqrt(M**2 + N**2)

            if denom < 1e-9:
                alphas.append(self.alpha0[i] if not is_setup else 0.0)
                continue

            ratio = L / denom
            if abs(ratio) > 1:
                alphas.append(self.alpha0[i] if not is_setup else 0.0)
            else:
                alphas.append(math.asin(ratio) - math.atan2(N, M))

        return np.array(alphas)

    def get_pulse_widths(self, target_alphas):
        pwms = []
        for i in range(6):
            direction = 1 if i % 2 == 0 else -1
            pwm = 1500 + self.offsets[i] + direction * (target_alphas[i] - self.alpha0[i]) * self.us_per_rad
            pwm = int(np.clip(pwm, 500, 2500))
            pwms.append(pwm)
        return pwms


def parse_esp32_line(line):
    """
    ESP32 输出格式:
    T,时间戳,下Roll,下Pitch,上Roll,上Pitch

    返回:
    esp_ms, bot_roll_deg, bot_pitch_deg, top_roll_deg, top_pitch_deg
    """
    parts = line.strip().split(',')
    if len(parts) != 6:
        return None
    if parts[0] != 'T':
        return None

    try:
        esp_ms = int(parts[1])
        bot_roll_deg = float(parts[2])
        bot_pitch_deg = float(parts[3])
        top_roll_deg = float(parts[4])
        top_pitch_deg = float(parts[5])
        return esp_ms, bot_roll_deg, bot_pitch_deg, top_roll_deg, top_pitch_deg
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description='Six-axis platform PID controller')
    parser.add_argument('--port', required=True, help='ESP32 serial port, e.g. /dev/ttyUSB0')
    parser.add_argument('--baudrate', type=int, default=921600)
    parser.add_argument('--calibration-seconds', type=float, default=5.0)
    parser.add_argument('--log-dir', default='~/go2_six_axis_platform/logs')
    args = parser.parse_args()

    config = {
        'a': 35.0,
        's': 117.5,
        'base_radius': 83.07,
        'base_bias': 48.12,
        'top_radius': 59.16,
        'top_bias': 54.18,
        'us_per_rad': 636.6,
        'servo_offsets': [0, 0, 0, 0, 0, 0]
    }

    stewart = StewartPlatform(config)

    # ========== 串口参数 ==========
    ser = serial.Serial(args.port, args.baudrate, timeout=0.02)
    time.sleep(2.0)
    ser.reset_input_buffer()

    # ========== 控制参数 ==========
    # 单轴测试模式: "roll" | "pitch" | "both"
    AXIS_TUNE_MODE = "both"

    POLARITY_ROLL = -1.0
    POLARITY_PITCH = -1.0

    pid_roll = PIDController(
        kp=1.05, ki=1.3, kd=0,
        output_limit=math.radians(30),
        integral_limit=0.2
    )
    pid_pitch = PIDController(
        kp=1.05, ki=1.3, kd=0,
        output_limit=math.radians(30),
        integral_limit=0.2
    )

    # 上平台测量的一阶低通
    ALPHA_TOP = 0.3
    ANGLE_DEADBAND = math.radians(0.15)
    MAX_PWM_STEP = 25

    filt_top_r = None
    filt_top_p = None

    # 下平台数据先也滤一下，主要为了 CSV 更平滑
    ALPHA_BOT = 0.3
    filt_bot_r = None
    filt_bot_p = None

    DEBUG_PRINT_INTERVAL = 0.2
    last_debug_time = 0.0

    last_pwms = [1500 + o for o in config['servo_offsets']]

    target_roll = 0.0
    target_pitch = 0.0

    cal_data_r = []
    cal_data_p = []
    is_calibrated = False
    calibration_duration = args.calibration_seconds

    # ========== CSV ==========
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = log_dir / f"jy901_dual_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_filename, 'w', newline='', encoding='utf-8-sig')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'pc_time_s',
        'esp_time_ms',
        'bot_roll_deg', 'bot_pitch_deg',
        'top_roll_deg', 'top_pitch_deg',
        'filt_bot_roll_deg', 'filt_bot_pitch_deg',
        'filt_top_roll_deg', 'filt_top_pitch_deg',
        'target_roll_deg', 'target_pitch_deg',
        'adj_roll_deg', 'adj_pitch_deg',
        'pwm1', 'pwm2', 'pwm3', 'pwm4', 'pwm5', 'pwm6'
    ])

    print(">>> 5秒校准开始（以上平台传感器作为控制反馈）...")
    start_time = time.time()

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            parsed = parse_esp32_line(line)
            if parsed is None:
                continue

            esp_ms, bot_roll_deg, bot_pitch_deg, top_roll_deg, top_pitch_deg = parsed

            # 转为弧度
            raw_bot_r = math.radians(bot_roll_deg)
            raw_bot_p = math.radians(bot_pitch_deg)
            raw_top_r = math.radians(top_roll_deg)
            raw_top_p = math.radians(top_pitch_deg)

            # ========== 滤波 ==========
            if filt_top_r is None:
                filt_top_r, filt_top_p = raw_top_r, raw_top_p
                filt_bot_r, filt_bot_p = raw_bot_r, raw_bot_p
            else:
                filt_top_r = (1 - ALPHA_TOP) * filt_top_r + ALPHA_TOP * raw_top_r
                filt_top_p = (1 - ALPHA_TOP) * filt_top_p + ALPHA_TOP * raw_top_p

                filt_bot_r = (1 - ALPHA_BOT) * filt_bot_r + ALPHA_BOT * raw_bot_r
                filt_bot_p = (1 - ALPHA_BOT) * filt_bot_p + ALPHA_BOT * raw_bot_p

            # ========== 校准 ==========
            if not is_calibrated:
                cal_data_r.append(filt_top_r)
                cal_data_p.append(filt_top_p)

                if time.time() - start_time > calibration_duration:
                    target_roll = float(np.mean(cal_data_r))
                    target_pitch = float(np.mean(cal_data_p))
                    pid_roll.reset()
                    pid_pitch.reset()
                    is_calibrated = True

                    print(
                        f"校准成功! "
                        f"目标 Top Roll={math.degrees(target_roll):.2f} deg, "
                        f"Top Pitch={math.degrees(target_pitch):.2f} deg"
                    )

                pwms = [1500 + o for o in config['servo_offsets']]
                adj_r = 0.0
                adj_p = 0.0

            else:
                # ========== 以上平台 PID 闭环（支持单轴测试） ==========
                top_r_for_pid = target_roll if abs(filt_top_r - target_roll) < ANGLE_DEADBAND else filt_top_r
                top_p_for_pid = target_pitch if abs(filt_top_p - target_pitch) < ANGLE_DEADBAND else filt_top_p

                if AXIS_TUNE_MODE == "roll":
                    adj_r = pid_roll.compute(target_roll, top_r_for_pid) * POLARITY_ROLL
                    adj_p = 0.0
                elif AXIS_TUNE_MODE == "pitch":
                    adj_r = 0.0
                    adj_p = pid_pitch.compute(target_pitch, top_p_for_pid) * POLARITY_PITCH
                else:
                    adj_r = pid_roll.compute(target_roll, top_r_for_pid) * POLARITY_ROLL
                    adj_p = pid_pitch.compute(target_pitch, top_p_for_pid) * POLARITY_PITCH

                # 这里先不加前馈，只做 PID
                target_alphas = stewart.solve_inverse_kinematics(
                    translation=[0, 0, 0],
                    rotation=[0, adj_p, adj_r]
                )

                current_target_pwms = stewart.get_pulse_widths(target_alphas)

                # 平滑步进
                pwms = [
                    int(last_pwms[i] + np.clip(current_target_pwms[i] - last_pwms[i], -MAX_PWM_STEP, MAX_PWM_STEP))
                    for i in range(6)
                ]

                now = time.time()
                if now - last_debug_time >= DEBUG_PRINT_INTERVAL:
                    print(
                        "MODE:{} | BOT raw R:{:+.2f} P:{:+.2f} | "
                        "TOP raw R:{:+.2f} P:{:+.2f} | "
                        "TOP filt R:{:+.2f} P:{:+.2f} | "
                        "ADJ R:{:+.2f} P:{:+.2f}".format(
                            AXIS_TUNE_MODE,
                            math.degrees(raw_bot_r),
                            math.degrees(raw_bot_p),
                            math.degrees(raw_top_r),
                            math.degrees(raw_top_p),
                            math.degrees(filt_top_r),
                            math.degrees(filt_top_p),
                            math.degrees(adj_r),
                            math.degrees(adj_p)
                        )
                    )
                    last_debug_time = now

            # ========== 发给 ESP32 ==========
            cmd = "<{}>\n".format(",".join(map(str, pwms)))
            ser.write(cmd.encode('utf-8'))
            last_pwms = pwms

            # ========== 写 CSV ==========
            csv_writer.writerow([
                time.time(),
                esp_ms,
                bot_roll_deg, bot_pitch_deg,
                top_roll_deg, top_pitch_deg,
                math.degrees(filt_bot_r), math.degrees(filt_bot_p),
                math.degrees(filt_top_r), math.degrees(filt_top_p),
                math.degrees(target_roll), math.degrees(target_pitch),
                math.degrees(adj_r), math.degrees(adj_p),
                *pwms
            ])

            # 降低缓存丢失风险
            csv_file.flush()

    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
    finally:
        try:
            # 停机时回中
            safe_pwms = [1500 + o for o in config['servo_offsets']]
            ser.write(f"<{','.join(map(str, safe_pwms))}>\n".encode('utf-8'))
            time.sleep(0.2)
        except Exception:
            pass

        csv_file.close()
        ser.close()
        print(f"CSV 已保存: {csv_filename}")


if __name__ == "__main__":
    main()
