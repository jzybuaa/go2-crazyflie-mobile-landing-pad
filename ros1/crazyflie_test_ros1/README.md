# 无刷 Crazyflie ROS1 跟随控制

本包以 `cf17` 为唯一飞行控制对象，以 `cf231` 的 Lighthouse 位姿作为移动平台参考。`cf231` 仅发布位姿，不向它发送飞行命令。

依赖 ROS Noetic、Crazyswarm1，以及仓库 `crazyswarm_overlay/patches/` 中的 Arm/Supervisor 扩展。安装、配置、运行和安全说明见仓库根目录 README。

离线检查：

```bash
python3 scripts/dog_tracker_flight.py --self-test
python3 scripts/dog_tracker_flight.py --offline-state-machine-test
```

配置中的机体 ID 和脚本中的 ID/topic 必须保持一致。修改前先拆桨验证，并确认 Lighthouse 坐标和无线电信道。
