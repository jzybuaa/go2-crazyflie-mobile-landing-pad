# 有刷 Crazyflie ROS1 跟随控制

这是针对旧版有刷 Crazyflie 的独立替代包，不是无刷包的附加节点。只适用于相应的有刷机型与旧版固件。

**无刷和有刷方案不能同时上电或运行。** 依赖 ROS Noetic、Crazyswarm1。安装、配置、运行和安全说明见仓库根目录 README。

离线检查：

```bash
python3 scripts/dog_tracker_flight_brushed.py --self-test
python3 scripts/dog_tracker_flight_brushed.py --offline-state-machine-test
```

此实现不依赖显式 Arm/Supervisor 服务。机体 ID/topic 必须与配置一致；首次检查必须拆桨并确认位姿、降落路径和硬件状态。
