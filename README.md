# Go2 移动停机坪与 Crazyflie 着陆实验

本仓库整理 Go2 背部六自由度并联平台与 Crazyflie 无人机协同实验的软件：Crazyflie ROS1 跟随/着陆控制、Crazyswarm1 Arm/Supervisor 接口补丁，以及平台姿态自稳上位机程序。

> 这是研究与实验代码，不是即插即用的飞行产品。公开前已排除专利初稿、本机日志、缓存和开发审计备份。代码仍需按实际硬件、无线电配置和实验场地检查。

## 目录

```text
crazyswarm_overlay/
  patches/                         # 针对 Crazyswarm1 的 Arm/Supervisor 补丁
ros1/
  crazyflie_test_ros1/             # 无刷 Crazyflie 跟随/着陆控制
  crazyflie_test_ros1_brushed/     # 有刷 Crazyflie 替代实现
six_axis_platform/
  pid_frame_aligned_go2.py         # 六自由度平台姿态自稳上位机
```

## 软件组成与完整性

| 子系统 | 本仓库包含 | 外部依赖/未包含内容 |
|---|---|---|
| Crazyflie 无刷控制 | ROS1 控制节点、launch、机群配置、离线自检入口 | ROS Noetic、Crazyswarm1、Crazyflie 与 Lighthouse 硬件；Arm/Supervisor 补丁需应用到指定 Crazyswarm1 版本 |
| Crazyflie 有刷控制 | 独立 ROS1 控制节点、launch、机群配置、离线自检入口 | ROS Noetic、Crazyswarm1、旧版有刷 Crazyflie 固件及 Lighthouse；与无刷方案不能同时运行 |
| 六自由度平台自稳 | PID、Stewart 逆运动学、串口协议、CSV 记录 | ESP32/传感器/舵机硬件和对应固件未包含；目前只能确认上位机代码，不能单独驱动完整平台 |
| 无线充电 | 无 | 未找到充电控制、充电状态检测或充电通信程序；这里只记录平台与无人机控制软件 |

两套 ROS1 包有完整的主要源文件、catkin 元数据、launch 和配置，脚本也提供离线自检入口。但这不等同于已在干净环境重新编译或完成真机验证。本次整理未改动飞行/自稳算法参数，也未运行飞行测试。

## 环境

- Ubuntu 20.04、ROS1 Noetic
- 与本仓库补丁匹配的 Crazyswarm1 基线：`beb05492eaf226462631545bf5972199099267c2`
- Python 3、NumPy、PySerial（仅六自由度平台脚本需要 NumPy/PySerial）
- Crazyflie、Lighthouse、Crazyradio，以及与本机一致的机体 ID、无线电信道和坐标系配置

## 安装 Crazyflie ROS1 包

先克隆与补丁基线一致的 Crazyswarm1，然后安装其递归子模块：

```bash
git clone https://github.com/USC-ACTLab/crazyswarm.git --recursive
cd crazyswarm
git checkout beb0549
git submodule update --init --recursive
```

假设本仓库位于 `$REPO`，在 Crazyswarm1 根目录应用本项目扩展：

```bash
cp "$REPO/crazyswarm_overlay/patches/Arm.srv" ros_ws/src/crazyswarm/srv/Arm.srv
git apply "$REPO/crazyswarm_overlay/patches/crazyswarm-arm-supervisor.patch"
cp -a "$REPO/ros1/crazyflie_test_ros1" ros_ws/src/
cp -a "$REPO/ros1/crazyflie_test_ros1_brushed" ros_ws/src/
./build.sh
```

如果 `git apply` 提示上下文不匹配，先停止，不要强制套用；确认当前 Crazyswarm1 commit 与 `beb0549` 一致。构建前请检查 `ros_ws/src/crazyswarm/launch/crazyflies.yaml`、两个包的 `config/`、机体类型、radio/channel、初始位置和 Lighthouse 坐标系。配置里的 ID `17` 与 `231` 是原实验角色编号示例，控制节点也使用对应的 `/cf17`、`/cf231` 名称；改 ID 时必须同步修改脚本中的 ID、ROS topic 与配置。

启动流程（先无桨/地面检查，终端 1 启动连接与位姿发布，终端 2 再启动控制节点）：

```bash
source /opt/ros/noetic/setup.bash
source ros_ws/devel/setup.bash
roslaunch crazyflie_test_ros1 lighthouse_connect.launch
```

另开终端并 source 同一工作区后，**只选与实际机型匹配的一种**：

```bash
rosrun crazyflie_test_ros1 dog_tracker_flight.py
# 或：有刷机型使用独立替代包，绝不要与上面的方案同时运行
rosrun crazyflie_test_ros1_brushed dog_tracker_flight_brushed.py
```

两个控制脚本均提供 `--self-test` 与 `--offline-state-machine-test`，可在接电机/螺旋桨前按下列命令检查纯软件逻辑：

```bash
python3 ros_ws/src/crazyflie_test_ros1/scripts/dog_tracker_flight.py --self-test
python3 ros_ws/src/crazyflie_test_ros1/scripts/dog_tracker_flight.py --offline-state-machine-test
python3 ros_ws/src/crazyflie_test_ros1_brushed/scripts/dog_tracker_flight_brushed.py --self-test
python3 ros_ws/src/crazyflie_test_ros1_brushed/scripts/dog_tracker_flight_brushed.py --offline-state-machine-test
```

按键行为与安全状态机以各脚本当前实现和实验硬件为准；首次通电/飞行前必须拆桨检查无线电连接、位姿、坐标轴、解锁状态及急停/降落路径。离线自检通过不代表真机安全。

## 六自由度平台自稳程序

安装 Python 依赖：

```bash
python3 -m pip install -r "$REPO/six_axis_platform/requirements.txt"
```

确认平台已机械固定、执行器供电安全、串口设备和固件协议一致后，再运行：

```bash
python3 "$REPO/six_axis_platform/pid_frame_aligned_go2.py" --port /dev/ttyUSB0
```

可用 `--baudrate`、`--calibration-seconds` 和 `--log-dir` 指定串口速率、校准时长和 CSV 目录。默认串口速率为 `921600`，默认记录到 `~/go2_six_axis_platform/logs`。程序期望 ESP32 每行输出：

```text
T,时间戳毫秒,下平台Roll度,下平台Pitch度,上平台Roll度,上平台Pitch度
```

上位机发送六路舵机 PWM：`<pwm1,pwm2,pwm3,pwm4,pwm5,pwm6>\n`。串口/字段格式必须与下位机固件完全一致。当前脚本没有串口数据看门狗；串口数据停止后不会保证平台回到安全姿态，执行器可能保持最后一次指令。按 `Ctrl+C` 会尝试发送中位 PWM 后退出；这不是硬件急停，若平台动作异常请使用独立断电/急停装置。

脚本内的 Stewart 几何参数、PID 增益、极性、限幅和舵机偏置保留自本地实验版本。不同机构尺寸、舵机连杆方向和传感器安装姿态可能要求重新标定；不要直接把示例参数视为通用参数。

## 安全与限制

- 无刷和有刷两套飞控配置/程序针对不同硬件，不能同时连接或运行。
- 先拆除螺旋桨验证连接、坐标、位置反馈和状态机，再进行任何带桨测试。
- 跟随/着陆控制依赖 Lighthouse 位姿；遮挡、坐标系不一致或位姿陈旧时不得继续飞行。
- 平台程序控制真实舵机。首次测试应使用机械限位/安全支撑，设置独立断电手段，并验证中位 PWM 与舵机方向。
- 本仓库未包含六自由度平台 ESP32 固件、充电控制程序、完整机械/电气设计或专利文件。

## 来源与许可

Crazyswarm1 上游项目及其许可见 [USC-ACTLab/crazyswarm](https://github.com/USC-ACTLab/crazyswarm)。两个本地 ROS 包的 `package.xml` 声明 MIT。六自由度自稳脚本未找到独立许可声明，因此本仓库不替它添加许可；仓库公开可见不代表该脚本已获得再利用许可。第三方依赖仍遵循各自的许可证。

专利说明初稿没有纳入仓库；公开前请另行核对专利申请状态、发明人/权利人授权以及需要公开的技术细节。
