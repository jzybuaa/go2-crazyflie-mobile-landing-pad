#!/usr/bin/env python3
"""Legacy brushed cf17 follows cf231 using Lighthouse pose only.

This node deliberately has no Supervisor or explicit arming dependency.
"""

import copy
import math
import select
import sys
import termios
import threading
import time
import tty
import types
from pathlib import Path
from collections import deque

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped

from pycrazyswarm import Crazyswarm


# Pose-health thresholds. The measured pose rate is about 9-10 Hz. cf231 had
# one 0.31 s packet gap, so 0.55 s tolerates one such gap while still rejecting
# stale data promptly.
EXPECTED_FRAME = "world"
POSE_TIMEOUT_S = 0.55
HEADER_MAX_AGE_S = 1.0
RATE_WINDOW_S = 3.0
MIN_POSE_RATE_HZ = 7.0
MIN_VALID_SAMPLES = 15
MIN_STABLE_DURATION_S = 2.0
MAX_POSE_STEP_M = 0.15
# A 0.15 m absolute floor rejects a one-frame estimator discontinuity. For
# missing frames, allow displacement proportional to elapsed message time so
# normal motion up to 1.5 m/s is not compared as if every sample were at 10 Hz.
MAX_POSE_SPEED_MPS = 1.50
RESYNC_MIN_SAMPLES = 8
RESYNC_MIN_DURATION_S = 0.50
RESYNC_CLUSTER_RADIUS_M = 0.05
RESYNC_MAX_SAMPLE_GAP_S = 0.40
QUAT_NORM_MIN = 0.80
QUAT_NORM_MAX = 1.20

# Command-target limits. The normal target remains platform z + 0.30 m.
TARGET_HEIGHT_MIN_M = 0.15
TARGET_HEIGHT_MAX_M = 1.00
MAX_TARGET_STEP_M = 0.20
MAX_TARGET_HORIZONTAL_FROM_CF_M = 0.80
MAX_TARGET_VERTICAL_FROM_CF_M = 0.45
# Normal FOLLOW remains constrained to the existing 0.80 m working radius.
# A separate, bounded acquisition radius lets cf17 safely close a modest
# initial separation using intermediate targets; it is not a global target
# limit and does not affect MANUAL or cf17 pose validation.
FOLLOW_NORMAL_DISTANCE_M = 0.80
FOLLOW_ACQUIRE_MAX_DISTANCE_M = 1.20
FOLLOW_ACQUIRE_RECOVERY_DISTANCE_M = 1.10

# Takeoff confirmation is based solely on consecutive healthy pose samples.
TAKEOFF_DURATION_S = 2.0
TAKEOFF_SETTLE_MARGIN_S = 0.25
TAKEOFF_HEIGHT_TOLERANCE_M = 0.05
# Must prove a real lift from the healthy pre-takeoff pose, not a single jump.
TAKEOFF_MIN_RISE_M = 0.12
TAKEOFF_MIN_SAFE_HEIGHT_M = 0.12
TAKEOFF_STABLE_SAMPLES = 5  # 5 x 5 Hz timer samples = about one stable second.
TAKEOFF_COMPLETION_TIMEOUT_S = 5.0
LAND_DURATION_S = 2.0
LAND_CONFIRM_TIMEOUT_S = 5.0
# The Lighthouse origin is not necessarily the brushed vehicle's physical
# ground plane.  Land against the healthy cf17 height observed immediately
# before this flight's takeoff, then accept a small measurement tolerance.
LAND_GROUND_HEIGHT_TOLERANCE_M = 0.04
LAND_STABLE_SAMPLES = 5
LAND_AUDIT_LOG_PERIOD_S = 0.5

# FOLLOW acquisition is intentionally much smaller/slower than the hard 0.20 m
# target-step limit.  0.05 m and 0.10 m/s give observable, conservative motion;
# duration is distance / speed, never a fixed maximum-step duration.
FOLLOW_ACQUIRE_MAX_STEP_M = 0.05
FOLLOW_ACQUIRE_MAX_SPEED_MPS = 0.10
FOLLOW_ACQUIRE_MIN_DURATION_S = 0.30
FOLLOW_ACQUIRE_COMPLETE_TOLERANCE_M = 0.025

# Normal FOLLOW is a streamed, absolute-goTo trajectory.  These limits are
# intentionally independent of the platform-motion classifier below: cf17
# always receives a bounded 0.40 m/s trajectory even if cf231 is observed
# moving faster on the ground.
FOLLOW_MAX_COMMAND_SPEED_MPS = 0.40
FOLLOW_MAX_COMMAND_ACCEL_MPS2 = 0.60
FOLLOW_COMMAND_MIN_DURATION_S = 0.30
FOLLOW_COMMAND_DT_MAX_S = 0.30

# cf231 is an input-only moving platform.  A sample above the generic pose
# jump gate is first quarantined.  It may be committed only after this bounded
# sequence test; it never relaxes the corresponding cf17 pose gate.
PLATFORM_MOTION_CONFIRM_SAMPLES = 4
PLATFORM_MOTION_CONFIRM_DURATION_S = 0.30
PLATFORM_MOTION_MAX_SPEED_MPS = 0.80
PLATFORM_MOTION_MAX_ACCEL_MPS2 = 1.50
PLATFORM_MOTION_MIN_DIRECTION_COS = 0.80
PLATFORM_MOTION_MAX_PREDICTION_DEVIATION_M = 0.50
PLATFORM_MOTION_HOLD_MAX_DURATION_S = 0.75

# A fresh but temporarily slow pose stream is a command-hold condition, not a
# relaxation of either the 7 Hz rate floor or the 0.55 s stale-pose hard limit.
HOLD_RECOVERY_RATE_HZ = 7.5
HOLD_RECOVERY_STABLE_S = 1.0
HOLD_MAX_DURATION_S = 4.0

# A FOLLOW hold has one public mode and an explicit, ordered internal reason.
# Target bounds below are recovery hysteresis only; the hard target bounds above
# are never widened or clamped.
HOLD_RATE = "rate"
HOLD_TARGET = "target"
HOLD_TARGET_DISTANCE = "target_distance"
HOLD_OUTLIER_CF17 = "outlier_cf17"
HOLD_OUTLIER_CF231 = "outlier_cf231"
HOLD_PLATFORM_MOTION = "platform_motion"
HOLD_PRIORITY = {HOLD_RATE: 1, HOLD_TARGET: 2, HOLD_TARGET_DISTANCE: 2,
                 HOLD_PLATFORM_MOTION: 3,
                 HOLD_OUTLIER_CF231: 4, HOLD_OUTLIER_CF17: 5}
TARGET_HOLD_RECOVERY_HEIGHT_MIN_M = 0.17
TARGET_HOLD_RECOVERY_HEIGHT_MAX_M = 0.98
TARGET_HOLD_RECOVERY_HORIZONTAL_M = 0.75
TARGET_HOLD_RECOVERY_VERTICAL_M = 0.40
TARGET_HOLD_STABLE_S = 1.0
TARGET_HOLD_MAX_DURATION_S = 4.0
OUTLIER_RULES = {
    "cf17": (5, 0.30, 0.04, 0.08, 0.50),
    "cf231": (5, 0.30, 0.05, 0.12, 0.75),
}
OUTLIER_PREDICTION_MAX_SPEED_MPS = 0.50


def finite_pose_values(pose):
    values = (
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    return all(math.isfinite(value) for value in values)


def position_distance(a, b):
    return math.sqrt(
        (a.position.x - b.position.x) ** 2
        + (a.position.y - b.position.y) ** 2
        + (a.position.z - b.position.z) ** 2
    )


def normalize_pose_message(msg):
    """Return a normalized copy, or (None, reason) for invalid input."""
    if msg.header.frame_id != EXPECTED_FRAME:
        return None, "frame_id=%r，不是 %r" % (msg.header.frame_id, EXPECTED_FRAME)
    if not finite_pose_values(msg.pose):
        return None, "位置或四元数含 NaN/Inf"
    q = msg.pose.orientation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if not math.isfinite(norm) or norm < QUAT_NORM_MIN or norm > QUAT_NORM_MAX:
        return None, "四元数范数 %.6f 超出 [%.2f, %.2f]" % (
            norm,
            QUAT_NORM_MIN,
            QUAT_NORM_MAX,
        )
    result = copy.deepcopy(msg)
    result.pose.orientation.x /= norm
    result.pose.orientation.y /= norm
    result.pose.orientation.z /= norm
    result.pose.orientation.w /= norm
    return result, None


class PoseHealth:
    """Thread-safe pose validation and rolling health state for one vehicle."""

    def __init__(self, name):
        self.name = name
        self.lock = threading.RLock()
        self.raw_message = None
        self.raw_header_stamp = None
        self.raw_received_monotonic = None
        self.raw_times = deque()
        self.message = None
        self.header_stamp = None
        self.received_monotonic = None
        self.valid_times = deque()
        self.accepted_history = deque(maxlen=2)
        self.consecutive_valid = 0
        self.stable_since = None
        self.last_invalid_reason = "尚未收到位姿"
        self.fault_latched = False
        self.fault_reason = None
        self.resync_candidates = deque()
        self.last_event = None

    @staticmethod
    def _prune_times(samples, now_monotonic):
        while samples and now_monotonic - samples[0] > RATE_WINDOW_S:
            samples.popleft()

    def _reset_valid_run(self, reason):
        self.consecutive_valid = 0
        self.stable_since = None
        self.last_invalid_reason = reason

    def _clear_resync(self):
        self.resync_candidates.clear()

    def _accept(self, normalized, now_monotonic):
        self.message = normalized
        self.header_stamp = normalized.header.stamp
        self.received_monotonic = now_monotonic
        self.valid_times.append(now_monotonic)
        self.accepted_history.append((copy.deepcopy(normalized), now_monotonic))
        self._prune_times(self.valid_times, now_monotonic)
        self.consecutive_valid += 1
        if self.stable_since is None:
            self.stable_since = now_monotonic
        self.last_invalid_reason = None
        self._clear_resync()

    def _resync_with_candidate(
        self, normalized, now_monotonic, distance, dt, sample_dt,
        allowed_distance
    ):
        if (
            self.resync_candidates
            and now_monotonic - self.resync_candidates[-1][1]
            > RESYNC_MAX_SAMPLE_GAP_S
        ):
            self.resync_candidates.clear()

        self.resync_candidates.append((normalized, now_monotonic))
        positions = np.array(
            [
                [
                    candidate.pose.position.x,
                    candidate.pose.position.y,
                    candidate.pose.position.z,
                ]
                for candidate, _ in self.resync_candidates
            ],
            dtype=float,
        )
        centroid = positions.mean(axis=0)
        if np.max(np.linalg.norm(positions - centroid, axis=1)) \
                > RESYNC_CLUSTER_RADIUS_M:
            self.resync_candidates.clear()
            self.resync_candidates.append((normalized, now_monotonic))

        count = len(self.resync_candidates)
        duration = (
            0.0
            if count < 2
            else now_monotonic - self.resync_candidates[0][1]
        )
        detail = (
            "位置差 %.3f m，距最后有效点 dt %.3f s，相邻原始 dt %.3f s，"
            "等效速度 %.2f m/s，允许 %.3f m"
            % (
                distance,
                dt,
                sample_dt,
                distance / dt if dt > 0.0 else float("inf"),
                allowed_distance,
            )
        )
        if count < RESYNC_MIN_SAMPLES or duration < RESYNC_MIN_DURATION_S:
            reason = (
                "单帧离群：%s；等待地面重同步 %d/%d，聚类时长 %.2f/%.2f s"
                % (
                    detail,
                    count,
                    RESYNC_MIN_SAMPLES,
                    duration,
                    RESYNC_MIN_DURATION_S,
                )
            )
            self._reset_valid_run(reason)
            return False, reason

        candidate_times = [
            candidate_time for _, candidate_time in self.resync_candidates
        ]
        self.message = normalized
        self.header_stamp = normalized.header.stamp
        self.received_monotonic = now_monotonic
        self.valid_times.clear()
        self.valid_times.extend(candidate_times)
        self._prune_times(self.valid_times, now_monotonic)
        self.consecutive_valid = count
        self.stable_since = self.resync_candidates[0][1]
        self.last_invalid_reason = None
        self.last_event = (
            "地面重同步成功：%d 个连续样本在 %.2f m 半径内，持续 %.2f s；%s"
            % (count, RESYNC_CLUSTER_RADIUS_M, duration, detail)
        )
        self._clear_resync()
        return True, None

    def update(
        self, msg, now_monotonic=None, ros_now_sec=None,
        allow_ground_resync=False, quarantine_airborne=False
    ):
        now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        if ros_now_sec is None:
            ros_now_sec = rospy.Time.now().to_sec()
        with self.lock:
            previous_raw_header_stamp = self.raw_header_stamp
            self.raw_message = copy.deepcopy(msg)
            self.raw_header_stamp = msg.header.stamp
            self.raw_received_monotonic = now_monotonic
            self.raw_times.append(now_monotonic)
            self._prune_times(self.raw_times, now_monotonic)
            self._prune_times(self.valid_times, now_monotonic)

        normalized, reason = normalize_pose_message(msg)
        with self.lock:
            if reason is None:
                stamp_sec = msg.header.stamp.to_sec()
                if stamp_sec <= 0.0:
                    reason = "header.stamp 为零"
                elif ros_now_sec - stamp_sec > HEADER_MAX_AGE_S:
                    reason = "ROS 时间戳过期 %.3f s" % (ros_now_sec - stamp_sec)
                elif stamp_sec - ros_now_sec > 0.25:
                    reason = "ROS 时间戳超前 %.3f s" % (stamp_sec - ros_now_sec)
            if reason is not None:
                self._clear_resync()
                self._reset_valid_run(reason)
                return False, reason

            # Validation above proves the raw message itself is structurally
            # sound.  During airborne recovery it remains quarantined until
            # TrackerFlight validates the complete return-to-trajectory set.
            if quarantine_airborne:
                return None, "飞行中离群恢复候选"

            if self.message is not None:
                dt = normalized.header.stamp.to_sec() - self.header_stamp.to_sec()
                if dt <= 0.0:
                    reason = "header.stamp 未递增，dt=%.6f s" % dt
                    self._clear_resync()
                    self._reset_valid_run(reason)
                    return False, reason
                sample_dt = dt
                if previous_raw_header_stamp is not None:
                    raw_dt = (
                        normalized.header.stamp.to_sec()
                        - previous_raw_header_stamp.to_sec()
                    )
                    if raw_dt > 0.0:
                        sample_dt = raw_dt
                distance = position_distance(normalized.pose, self.message.pose)
                allowed_distance = max(
                    MAX_POSE_STEP_M, MAX_POSE_SPEED_MPS * sample_dt
                )
                if distance > allowed_distance:
                    if allow_ground_resync:
                        return self._resync_with_candidate(
                            normalized,
                            now_monotonic,
                            distance,
                            dt,
                            sample_dt,
                            allowed_distance,
                        )
                    reason = (
                        "飞行中故障锁定：位置差 %.3f m，距最后有效点 "
                        "dt %.3f s，相邻原始 dt %.3f s，等效速度 %.2f m/s，"
                        "允许 %.3f m"
                        % (
                            distance,
                            dt,
                            sample_dt,
                            distance / dt,
                            allowed_distance,
                        )
                    )
                    self._clear_resync()
                    self._reset_valid_run(reason)
                    return False, reason

            self._accept(normalized, now_monotonic)
            return True, None

    def consume_event(self):
        with self.lock:
            event = self.last_event
            self.last_event = None
            return event

    def latch_fault(self, reason):
        with self.lock:
            self.fault_latched = True
            self.fault_reason = reason

    def clear_fault_if_healthy(self, now_monotonic=None):
        reasons = self.health_reasons(now_monotonic, include_latch=False)
        with self.lock:
            if reasons:
                return False, reasons
            self.fault_latched = False
            self.fault_reason = None
            return True, []

    def _rate_hz(self, samples, now_monotonic):
        self._prune_times(samples, now_monotonic)
        if len(samples) < 2:
            return 0.0
        span = samples[-1] - samples[0]
        return (len(samples) - 1) / span if span > 0.0 else 0.0

    def pose_rate_hz(self, now_monotonic=None):
        now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        with self.lock:
            return self._rate_hz(self.valid_times, now_monotonic)

    def raw_rate_hz(self, now_monotonic=None):
        now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        with self.lock:
            return self._rate_hz(self.raw_times, now_monotonic)

    def health_reasons(self, now_monotonic=None, include_latch=True):
        reasons = self.hard_reasons(now_monotonic, include_latch=include_latch)
        rate = self.pose_rate_hz(now_monotonic)
        if rate < MIN_POSE_RATE_HZ:
            reasons.append("位姿频率 %.2f Hz < %.2f Hz" % (rate, MIN_POSE_RATE_HZ))
        return reasons

    def hard_reasons(self, now_monotonic=None, include_latch=True):
        """All safety failures except a fresh, valid low-rate stream."""
        now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        with self.lock:
            reasons = []
            if include_latch and self.fault_latched:
                reasons.append("故障锁定: %s" % self.fault_reason)
            if self.resync_candidates:
                reasons.append(
                    "等待地面重同步 %d/%d"
                    % (len(self.resync_candidates), RESYNC_MIN_SAMPLES)
                )
            if self.message is None or self.received_monotonic is None:
                reasons.append(self.last_invalid_reason or "尚未收到有效位姿")
                return reasons
            age = now_monotonic - self.received_monotonic
            if age > POSE_TIMEOUT_S:
                reasons.append("位姿过期 %.3f s > %.2f s" % (age, POSE_TIMEOUT_S))
            if self.consecutive_valid < MIN_VALID_SAMPLES:
                reasons.append(
                    "连续有效样本 %d < %d"
                    % (self.consecutive_valid, MIN_VALID_SAMPLES)
                )
            stable_duration = (
                0.0 if self.stable_since is None else now_monotonic - self.stable_since
            )
            if stable_duration < MIN_STABLE_DURATION_S:
                reasons.append(
                    "连续稳定时间 %.2f s < %.2f s"
                    % (stable_duration, MIN_STABLE_DURATION_S)
                )
            return reasons

    def soft_rate_state(self, now_monotonic=None):
        """Return (rate, age) only when this health object has no hard fault."""
        now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        if self.hard_reasons(now_monotonic):
            return None
        with self.lock:
            return (
                self._rate_hz(self.valid_times, now_monotonic),
                now_monotonic - self.received_monotonic,
            )

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.message)

    def raw_snapshot(self):
        with self.lock:
            return copy.deepcopy(self.raw_message)

    def raw_age(self, now_monotonic=None):
        now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        with self.lock:
            return float("inf") if self.raw_received_monotonic is None else (
                now_monotonic - self.raw_received_monotonic
            )

    def commit_airborne_recovery(self, candidates):
        """Commit only a pre-validated return-to-old-trajectory candidate set."""
        with self.lock:
            for candidate, candidate_time in candidates:
                self.valid_times.append(candidate_time)
            self._prune_times(self.valid_times, candidates[-1][1])
            self.message = copy.deepcopy(candidates[-1][0])
            self.header_stamp = self.message.header.stamp
            self.received_monotonic = candidates[-1][1]
            self.consecutive_valid += len(candidates)
            self.accepted_history.append((copy.deepcopy(self.message), candidates[-1][1]))
            self.last_invalid_reason = None
            self._clear_resync()

    def summary(self):
        with self.lock:
            now_monotonic = time.monotonic()
            raw_age = (
                float("inf")
                if self.raw_received_monotonic is None
                else now_monotonic - self.raw_received_monotonic
            )
            valid_age = (
                float("inf")
                if self.received_monotonic is None
                else now_monotonic - self.received_monotonic
            )
            return (
                "%s: raw_rate=%.2f Hz, valid_rate=%.2f Hz, valid=%d, "
                "raw_age=%s, valid_age=%s, resync=%d/%d, latched=%s"
            ) % (
                self.name,
                self.raw_rate_hz(now_monotonic),
                self.pose_rate_hz(now_monotonic),
                self.consecutive_valid,
                "n/a" if not math.isfinite(raw_age) else "%.3f s" % raw_age,
                "n/a" if not math.isfinite(valid_age) else "%.3f s" % valid_age,
                len(self.resync_candidates),
                RESYNC_MIN_SAMPLES,
                self.fault_latched,
            )


def target_reasons(target, yaw, cf_pose, previous_target, horizontal_limit=None):
    reasons = []
    target = np.asarray(target, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)) or not math.isfinite(yaw):
        return ["目标位置或 yaw 含 NaN/Inf"]
    if not TARGET_HEIGHT_MIN_M <= target[2] <= TARGET_HEIGHT_MAX_M:
        reasons.append(
            "目标高度 %.3f m 超出 [%.2f, %.2f] m"
            % (target[2], TARGET_HEIGHT_MIN_M, TARGET_HEIGHT_MAX_M)
        )
    current = np.array(
        [cf_pose.position.x, cf_pose.position.y, cf_pose.position.z], dtype=float
    )
    horizontal = np.linalg.norm(target[:2] - current[:2])
    vertical = abs(target[2] - current[2])
    horizontal_limit = (MAX_TARGET_HORIZONTAL_FROM_CF_M if horizontal_limit is None
                        else horizontal_limit)
    if horizontal > horizontal_limit:
        reasons.append(
            "目标水平距离 %.3f m > %.2f m"
            % (horizontal, horizontal_limit)
        )
    if vertical > MAX_TARGET_VERTICAL_FROM_CF_M:
        reasons.append(
            "目标垂直距离 %.3f m > %.2f m"
            % (vertical, MAX_TARGET_VERTICAL_FROM_CF_M)
        )
    if previous_target is not None:
        step = np.linalg.norm(target - previous_target)
        if step > MAX_TARGET_STEP_M:
            reasons.append("单次目标跳变 %.3f m > %.2f m" % (step, MAX_TARGET_STEP_M))
    return reasons


class TrackerFlight:
    MODE_IDLE = "idle"
    MODE_TAKEOFF_WAIT_BRUSHED = "takeoff_wait_brushed"
    MODE_TAKEOFF_UNCONFIRMED = "takeoff_unconfirmed"
    MODE_FOLLOW = "follow"
    MODE_FOLLOW_ACQUIRE = "follow_acquire"
    MODE_FOLLOW_HOLD = "follow_hold"
    MODE_LANDING = "landing"
    MODE_LANDING_UNCONFIRMED = "landing_unconfirmed"
    MODE_MANUAL = "manual"
    MODE_FAULT = "fault"

    def __init__(self, crazyflies_yaml):
        self.swarm = Crazyswarm(crazyflies_yaml=crazyflies_yaml)
        self.time_helper = self.swarm.timeHelper
        self.cf17 = self.swarm.allcfs.crazyfliesById[17]
        if 231 not in self.swarm.allcfs.crazyfliesById:
            raise RuntimeError("cf231 is missing from the shared configuration")

        self.cf17_health = PoseHealth("cf17")
        self.cf231_health = PoseHealth("cf231")
        self.is_flying = False
        self.flight_mode = self.MODE_IDLE
        # Only _send_target() writes this: it is the last successfully sent
        # absolute goTo target, never a pose, candidate, or MANUAL reference.
        self.previous_target = None
        self.takeoff_target_height = None
        self.takeoff_start_height = None
        self.takeoff_valid_samples = 0
        self.landing_started_monotonic = None
        self.landing_start_height = None
        self.landing_target_height = None
        self.landing_stable_samples = 0
        self.landing_start_position = None
        self.landing_start_yaw = None
        self.landing_last_audit_log_monotonic = None
        self.takeoff_started_monotonic = None
        self.takeoff_height_stable_samples = 0
        self.takeoff_low_rate_last_log_monotonic = None
        self.acquire_anchor = None
        self.follow_hold_started_monotonic = None
        self.follow_hold_deadline_monotonic = None
        self.follow_hold_recovery_started_monotonic = None
        self.follow_hold_reason = None
        self.follow_hold_detail = None
        self.outlier_context = None
        self.platform_motion_context = None
        self.platform_motion_observations = deque(maxlen=PLATFORM_MOTION_CONFIRM_SAMPLES)
        self.follow_command_speed_mps = 0.0
        self.follow_command_last_sent_monotonic = None
        self.follow_command_last_log_monotonic = None
        # Audit-only metadata for the last successful absolute command.  It
        # never participates in target generation or landing behavior.
        self.last_successful_target = None
        self.last_successful_target_yaw = None
        self.last_successful_target_duration = None
        self.last_successful_target_monotonic = None
        # rospy.Timer callbacks normally serialize, but protect the target
        # generator as the goTo service may release the interpreter lock.
        self.follow_command_lock = threading.Lock()
        # Serializes every cf17 goTo RPC with the LANDING state transition and
        # land RPC.  A Timer callback that has already entered a goTo is
        # allowed to finish; once land owns this lock, no new target can pass.
        self.cf17_command_lock = threading.RLock()

        self.manual_x = 0.0
        self.manual_y = 0.0
        self.manual_z = 0.0
        # MANUAL owns an absolute target that is intentionally independent of
        # cf231.  Its horizontal axes are frozen from cf17's yaw at the moment
        # MANUAL is entered, so the keyboard directions do not rotate as the
        # platform moves.
        self.manual_reference_target = None
        self.manual_target = None
        self.manual_yaw = None
        self.xy_step = 0.020
        self.z_step = 0.020
        self.max_xy_offset = 0.40
        self.max_z_offset = 0.30
        self.manual_command_period = 0.10
        self.last_manual_command_time = 0.0

        self.cf17_pose_subscriber = rospy.Subscriber(
            "/cf17/pose", PoseStamped, self.cf17_pose_callback, queue_size=10
        )
        self.cf231_pose_subscriber = rospy.Subscriber(
            "/cf231/pose", PoseStamped, self.cf231_pose_callback, queue_size=10
        )
        self.track_timer = rospy.Timer(rospy.Duration(0.2), self.track_loop)
        self.status_timer = rospy.Timer(rospy.Duration(2.0), self.status_loop)

    def _pose_callback(self, health, msg):
        # Legacy brushed firmware has no reliable Supervisor topic.  A confirmed
        # on-ground/idle state is the only time resynchronization is allowed.
        allow_ground_resync = (
            not self.is_flying
            and self.flight_mode in (self.MODE_IDLE, self.MODE_TAKEOFF_UNCONFIRMED,
                                     self.MODE_LANDING_UNCONFIRMED)
        )
        quarantining = (
            self.flight_mode == self.MODE_FOLLOW_HOLD
            and (
                (self.outlier_context is not None
                 and self.outlier_context["name"] == health.name)
                or (health.name == "cf231"
                    and self.follow_hold_reason == HOLD_PLATFORM_MOTION
                    and self.platform_motion_context is not None)
            )
        )
        valid, reason = health.update(
            msg, allow_ground_resync=allow_ground_resync,
            quarantine_airborne=quarantining,
        )
        if valid is None:
            if health.name == "cf231" and self.follow_hold_reason == HOLD_PLATFORM_MOTION:
                self._update_platform_motion_confirmation()
                return
            self._handle_airborne_outlier(health, msg, reason)
            return
        if valid:
            if health.name == "cf231":
                self._observe_platform_motion(health.snapshot())
            event = health.consume_event()
            if event is not None:
                rospy.loginfo("%s %s", health.name, event)
            return
        rospy.logwarn_throttle(1.0, "%s 位姿无效: %s", health.name, reason)
        if self.flight_mode == self.MODE_FAULT:
            return
        if self.is_flying and reason.startswith("飞行中故障锁定：位置差"):
            if health.name == "cf231":
                self._start_platform_motion_confirmation(reason)
                return
            self._handle_airborne_outlier(health, msg, reason)
        elif self.is_flying and not health.fault_latched:
            health.latch_fault(reason)
            self.enter_fault("%s 位姿无效: %s" % (health.name, reason))

    def cf17_pose_callback(self, msg):
        self._pose_callback(self.cf17_health, msg)

    def cf231_pose_callback(self, msg):
        self._pose_callback(self.cf231_health, msg)

    def localization_reasons(self):
        reasons = []
        for health in (self.cf17_health, self.cf231_health):
            reasons.extend("%s: %s" % (health.name, item) for item in health.health_reasons())
        return reasons

    def hard_localization_reasons(self):
        reasons = []
        for health in (self.cf17_health, self.cf231_health):
            reasons.extend("%s: %s" % (health.name, item) for item in health.hard_reasons())
        return reasons

    def soft_rate_states(self):
        states = []
        for health in (self.cf17_health, self.cf231_health):
            state = health.soft_rate_state()
            if state is not None and state[0] < MIN_POSE_RATE_HZ:
                states.append((health.name, state[0], state[1]))
        return states

    def legacy_status(self):
        """State label only: old brushed firmware exposes no Supervisor proof."""
        return "unsupported/legacy brushed"

    def enter_fault(self, reason, category="定位故障锁定"):
        self.cf17_health.latch_fault(reason)
        self.flight_mode = self.MODE_FAULT
        rospy.logerr(
            "%s：%s。已停止产生新的 goTo；不会自动 cmd_stop 或自动恢复。"
            "操作员应评估后按 s 请求正常降落，或在未飞行时按 c 人工清除锁定。",
            category, reason,
        )

    def confirm_recovery(self):
        if self.is_flying:
            rospy.logwarn("飞行状态下不允许清除定位故障锁定；请先评估并按 s 降落。")
            return
        reasons = []
        for health in (self.cf17_health, self.cf231_health):
            ok, current = health.clear_fault_if_healthy()
            if not ok:
                reasons.extend("%s: %s" % (health.name, item) for item in current)
        if reasons:
            rospy.logwarn("无法清除故障锁定：%s", "; ".join(reasons))
            return
        self.flight_mode = self.MODE_IDLE
        self.previous_target = None
        rospy.loginfo("操作员已确认，两机定位当前健康；故障锁定已清除。")

    def status_loop(self, _event):
        reasons = self.localization_reasons()
        rospy.loginfo_throttle(
            4.0,
            "状态=%s, flying=%s, supervisor=%s | %s | %s | 定位=%s",
            self.flight_mode,
            self.is_flying,
            self.legacy_status(),
            self.cf17_health.summary(),
            self.cf231_health.summary(),
            "健康" if not reasons else "未就绪",
        )

    def get_pad_global_target(self, platform_pose, dx, dy, dz):
        qx = platform_pose.orientation.x
        qy = platform_pose.orientation.y
        qz = platform_pose.orientation.z
        qw = platform_pose.orientation.w
        cx = platform_pose.position.x
        cy = platform_pose.position.y
        cz = platform_pose.position.z
        rx = (
            (1 - 2 * qy**2 - 2 * qz**2) * dx
            + (2 * qx * qy - 2 * qz * qw) * dy
            + (2 * qx * qz + 2 * qy * qw) * dz
        )
        ry = (
            (2 * qx * qy + 2 * qz * qw) * dx
            + (1 - 2 * qx**2 - 2 * qz**2) * dy
            + (2 * qy * qz - 2 * qx * qw) * dz
        )
        rz = (
            (2 * qx * qz - 2 * qy * qw) * dx
            + (2 * qy * qz + 2 * qx * qw) * dy
            + (1 - 2 * qx**2 - 2 * qy**2) * dz
        )
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2)
        )
        return np.array([cx + rx, cy + ry, cz + rz + 0.30]), yaw

    @staticmethod
    def clamp(value, limit):
        return max(-limit, min(limit, value))

    @staticmethod
    def pose_yaw(pose):
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        qw = pose.orientation.w
        return math.atan2(
            2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2)
        )

    def _command_reasons(self):
        health_reasons = self.hard_localization_reasons()
        return health_reasons

    def _hold_deadline_for(self, reason, now):
        if reason == HOLD_OUTLIER_CF17:
            return now + OUTLIER_RULES["cf17"][4]
        if reason == HOLD_OUTLIER_CF231:
            return now + OUTLIER_RULES["cf231"][4]
        if reason == HOLD_PLATFORM_MOTION:
            return now + PLATFORM_MOTION_HOLD_MAX_DURATION_S
        return now + (TARGET_HOLD_MAX_DURATION_S
                      if reason in (HOLD_TARGET, HOLD_TARGET_DISTANCE) else HOLD_MAX_DURATION_S)

    def _enter_follow_hold(self, reason, detail):
        """Enter/escalate one bounded HOLD transaction; never resets its clock."""
        now = time.monotonic()
        new_deadline = self._hold_deadline_for(reason, now)
        if self.flight_mode == self.MODE_FOLLOW_HOLD:
            old_reason, old_deadline = self.follow_hold_reason, self.follow_hold_deadline_monotonic
            if HOLD_PRIORITY[reason] <= HOLD_PRIORITY[old_reason]:
                return
            self.follow_hold_reason = reason
            self.follow_hold_detail = detail
            self.follow_hold_deadline_monotonic = min(old_deadline, new_deadline)
            self.follow_hold_recovery_started_monotonic = None
            rospy.logwarn("FOLLOW_HOLD 原因升级：%s -> %s；deadline %.3f -> %.3f。",
                          old_reason, reason, old_deadline, self.follow_hold_deadline_monotonic)
            return
        self.flight_mode = self.MODE_FOLLOW_HOLD
        self.follow_hold_reason = reason
        self.follow_hold_detail = detail
        self.follow_hold_started_monotonic = now
        self.follow_hold_deadline_monotonic = new_deadline
        self.follow_hold_recovery_started_monotonic = None
        rospy.logwarn("进入 FOLLOW_HOLD[%s]：%s；暂停新的 goTo，保持最后成功目标；deadline=%.3f。",
                      reason, detail, new_deadline)

    def _hold_hard_reasons(self, now):
        reasons = []
        quarantined = None
        if self.follow_hold_reason == HOLD_OUTLIER_CF17:
            quarantined = self.cf17_health
        elif self.follow_hold_reason in (HOLD_OUTLIER_CF231, HOLD_PLATFORM_MOTION):
            quarantined = self.cf231_health
        for health in (self.cf17_health, self.cf231_health):
            if health is quarantined:
                if health.fault_latched:
                    reasons.append("%s: 故障锁定: %s" % (health.name, health.fault_reason))
                if health.raw_age(now) > POSE_TIMEOUT_S:
                    reasons.append("%s: 原始位姿过期 %.3f s > %.2f s" % (
                        health.name, health.raw_age(now), POSE_TIMEOUT_S))
            else:
                reasons.extend("%s: %s" % (health.name, item)
                               for item in health.hard_reasons(now))
        return reasons

    def _clear_follow_hold(self):
        self.follow_hold_started_monotonic = None
        self.follow_hold_deadline_monotonic = None
        self.follow_hold_recovery_started_monotonic = None
        self.follow_hold_reason = None
        self.follow_hold_detail = None
        self.outlier_context = None
        self.platform_motion_context = None

    def _follow_command_allowed(self):
        hard_reasons = self._command_reasons()
        if hard_reasons:
            self.enter_fault("; ".join(hard_reasons))
            return False
        rate_states = self.soft_rate_states()
        if rate_states:
            detail = "; ".join("%s=%.2f Hz, age=%.3f s" % state for state in rate_states)
            self._enter_follow_hold(HOLD_RATE, detail)
            return False
        return True

    def update_follow_hold(self):
        """Monitor only; HOLD never generates a target or a radio command."""
        now = time.monotonic()
        hard_reasons = self._hold_hard_reasons(now)
        if hard_reasons:
            self.enter_fault("FOLLOW_HOLD: " + "; ".join(hard_reasons))
            return
        started, deadline = self.follow_hold_started_monotonic, self.follow_hold_deadline_monotonic
        if started is None or deadline is None or self.follow_hold_reason is None:
            self.enter_fault("FOLLOW_HOLD 缺少进入时间")
            return
        if now > deadline:
            rospy.logerr("FOLLOW_HOLD[%s] 超时：持续 %.3f s，deadline=%.3f，转入 FAULT。",
                         self.follow_hold_reason, now - started, deadline)
            if self.follow_hold_reason == HOLD_TARGET_DISTANCE:
                self.enter_fault("目标距离 HOLD 超时；未进入 %.2f m 恢复范围" %
                                 FOLLOW_ACQUIRE_RECOVERY_DISTANCE_M,
                                 category="目标范围安全锁定")
            elif self.follow_hold_reason == HOLD_TARGET:
                self.enter_fault("目标暂时不可行 HOLD 超时", category="目标范围安全锁定")
            else:
                self.enter_fault("FOLLOW_HOLD 超时")
            return
        if self.follow_hold_reason in (HOLD_OUTLIER_CF17, HOLD_OUTLIER_CF231,
                                       HOLD_PLATFORM_MOTION):
            return
        if self.follow_hold_reason == HOLD_TARGET:
            target, yaw = self._follow_target_from_latest_platform()
            if self._target_recovery_ok(target, yaw):
                self._hold_recovery_timer(now, "目标回到恢复滞回范围")
            else:
                self.follow_hold_recovery_started_monotonic = None
            return
        if self.follow_hold_reason == HOLD_TARGET_DISTANCE:
            target, yaw = self._follow_target_from_latest_platform()
            if self._target_distance_recovery_ok(target, yaw):
                self._hold_recovery_timer(
                    now, "目标回到 %.2f m 接入恢复范围" % FOLLOW_ACQUIRE_RECOVERY_DISTANCE_M)
            else:
                self.follow_hold_recovery_started_monotonic = None
            return
        rate_states = self.soft_rate_states()
        if rate_states:
            if self.follow_hold_recovery_started_monotonic is not None:
                rospy.logwarn("FOLLOW_HOLD 恢复稳定计时被低频打断；重新等待。")
                self.follow_hold_recovery_started_monotonic = None
            return
        rates = [health.soft_rate_state()[0] for health in (self.cf17_health, self.cf231_health)]
        if any(rate < HOLD_RECOVERY_RATE_HZ for rate in rates):
            if self.follow_hold_recovery_started_monotonic is not None:
                rospy.logwarn("FOLLOW_HOLD 恢复稳定计时被 %.2f Hz 打断；重新等待。", min(rates))
                self.follow_hold_recovery_started_monotonic = None
            return
        if self.follow_hold_recovery_started_monotonic is None:
            self.follow_hold_recovery_started_monotonic = now
            rospy.loginfo("FOLLOW_HOLD 恢复稳定计时开始：两机频率均 >= %.2f Hz。", HOLD_RECOVERY_RATE_HZ)
            return
        stable = now - self.follow_hold_recovery_started_monotonic
        if stable < HOLD_RECOVERY_STABLE_S:
            return
        rospy.loginfo("FOLLOW_HOLD[%s] 恢复稳定 %.3f s；转入 FOLLOW_ACQUIRE。", self.follow_hold_reason, stable)
        self._clear_follow_hold()
        self._begin_follow_acquire()

    def _hold_recovery_timer(self, now, detail):
        if self.follow_hold_recovery_started_monotonic is None:
            self.follow_hold_recovery_started_monotonic = now
            rospy.loginfo("FOLLOW_HOLD[%s] 恢复稳定计时开始：%s。", self.follow_hold_reason, detail)
            return
        if now - self.follow_hold_recovery_started_monotonic >= TARGET_HOLD_STABLE_S:
            rospy.loginfo("FOLLOW_HOLD[%s] 已稳定 %.3f s；转入 FOLLOW_ACQUIRE。",
                          self.follow_hold_reason, now - self.follow_hold_recovery_started_monotonic)
            self._clear_follow_hold()
            self._begin_follow_acquire()

    def _target_recovery_ok(self, target, yaw):
        if not np.all(np.isfinite(target)) or not math.isfinite(yaw):
            self.enter_fault("FOLLOW 目标含 NaN/Inf")
            return False
        cf = self.cf17_health.snapshot().pose
        horizontal = np.linalg.norm(target[:2] - np.array([cf.position.x, cf.position.y]))
        vertical = abs(target[2] - cf.position.z)
        return (TARGET_HOLD_RECOVERY_HEIGHT_MIN_M <= target[2] <= TARGET_HOLD_RECOVERY_HEIGHT_MAX_M
                and horizontal <= TARGET_HOLD_RECOVERY_HORIZONTAL_M
                and vertical <= TARGET_HOLD_RECOVERY_VERTICAL_M)

    def _target_distance_recovery_ok(self, target, yaw):
        """Distance HOLD hysteresis, with all final height/vertical checks intact."""
        reasons = self._follow_geometry_reasons(target, yaw, FOLLOW_ACQUIRE_MAX_DISTANCE_M)
        if reasons:
            return False
        cf = self.cf17_health.snapshot().pose
        horizontal = np.linalg.norm(target[:2] - np.array([cf.position.x, cf.position.y]))
        return horizontal <= FOLLOW_ACQUIRE_RECOVERY_DISTANCE_M

    def _handle_airborne_outlier(self, health, msg, detail):
        """Quarantine airborne outliers; recovery must return to old trajectory."""
        now = time.monotonic()
        reason = HOLD_OUTLIER_CF17 if health.name == "cf17" else HOLD_OUTLIER_CF231
        if self.flight_mode != self.MODE_FOLLOW_HOLD or self.outlier_context is None:
            history = list(health.accepted_history)
            if len(history) < 2:
                self.enter_fault("%s 离群但缺少两个离群前有效样本" % health.name)
                return
            old0, old1 = history[-2], history[-1]
            delta_t = max(1e-3, old1[1] - old0[1])
            p0 = np.array([old0[0].pose.position.x, old0[0].pose.position.y, old0[0].pose.position.z])
            p1 = np.array([old1[0].pose.position.x, old1[0].pose.position.y, old1[0].pose.position.z])
            velocity = (p1 - p0) / delta_t
            speed = np.linalg.norm(velocity)
            if speed > OUTLIER_PREDICTION_MAX_SPEED_MPS:
                velocity *= OUTLIER_PREDICTION_MAX_SPEED_MPS / speed
            self.outlier_context = {"name": health.name, "anchor": p1,
                                    "anchor_time": old1[1], "velocity": velocity,
                                    "candidates": deque()}
            self._enter_follow_hold(reason, "%s：%s" % (health.name, detail))
            # The triggering frame is evidence of the anomaly, never recovery
            # evidence and never an accepted reference.
            return
        context = self.outlier_context
        if context is None or context["name"] != health.name:
            return
        candidate = health.raw_snapshot()
        if candidate is None:
            self.enter_fault("%s 离群恢复缺少原始位姿" % health.name)
            return
        position = np.array([candidate.pose.position.x, candidate.pose.position.y,
                             candidate.pose.position.z], dtype=float)
        expected = context["anchor"] + context["velocity"] * (now - context["anchor_time"])
        count, minimum_duration, radius, max_deviation, _ = OUTLIER_RULES[health.name]
        if len(context["candidates"]) and np.linalg.norm(position - expected) > max_deviation:
            self.enter_fault("%s 离群恢复偏离旧轨迹预测 %.3f m > %.3f m" % (
                health.name, np.linalg.norm(position - expected), max_deviation))
            return
        context["candidates"].append((candidate, now))
        positions = np.array([[item.pose.position.x, item.pose.position.y, item.pose.position.z]
                              for item, _ in context["candidates"]])
        center = positions.mean(axis=0)
        if np.max(np.linalg.norm(positions - center, axis=1)) > radius:
            self.enter_fault("%s 离群恢复候选不聚类（半径 > %.3f m）" % (health.name, radius))
            return
        duration = now - context["candidates"][0][1]
        if (len(context["candidates"]) < count or duration < minimum_duration
                or np.linalg.norm(center - expected) > max_deviation):
            return
        health.commit_airborne_recovery(context["candidates"])
        rospy.loginfo("FOLLOW_HOLD[%s] 恢复：%d 个候选返回旧轨迹 %.3f m 内，持续 %.3f s；转 FOLLOW_ACQUIRE。",
                      self.follow_hold_reason, len(context["candidates"]), max_deviation, duration)
        self._clear_follow_hold()
        self._begin_follow_acquire()

    @staticmethod
    def _position_from_message(message):
        return np.array([message.pose.position.x, message.pose.position.y,
                         message.pose.position.z], dtype=float)

    def _observe_platform_motion(self, message):
        """Record already accepted cf231 motion; never changes its pose gate."""
        if message is None:
            return
        if not hasattr(self, "platform_motion_observations"):
            self.platform_motion_observations = deque(maxlen=PLATFORM_MOTION_CONFIRM_SAMPLES)
        self.platform_motion_observations.append(
            (self._position_from_message(message), time.monotonic())
        )

    def _start_platform_motion_confirmation(self, detail):
        """Quarantine a cf231 jump while testing for bounded real movement."""
        if self.flight_mode not in (self.MODE_FOLLOW, self.MODE_FOLLOW_ACQUIRE,
                                    self.MODE_FOLLOW_HOLD):
            self._handle_airborne_outlier(self.cf231_health, None, detail)
            return
        history = list(self.cf231_health.accepted_history)
        if len(history) < 2:
            self.enter_fault("cf231 平台运动确认缺少两个离群前有效样本")
            return
        old0, old1 = history[-2], history[-1]
        anchor = self._position_from_message(old1[0])
        velocity = (anchor - self._position_from_message(old0[0])) / max(1e-3, old1[1] - old0[1])
        speed = np.linalg.norm(velocity)
        if speed > OUTLIER_PREDICTION_MAX_SPEED_MPS:
            velocity *= OUTLIER_PREDICTION_MAX_SPEED_MPS / speed
        self.platform_motion_context = {
            "anchor": anchor,
            "anchor_time": old1[1],
            "prediction_velocity": velocity,
            "candidates": deque(),
            "last_velocity": None,
        }
        self._enter_follow_hold(HOLD_PLATFORM_MOTION, "cf231：" + detail)
        self._update_platform_motion_confirmation()

    def _update_platform_motion_confirmation(self):
        """Commit only a short, coherent cf231 motion sequence.

        The 0.50 m migration test is deliberately against the limited
        pre-jump prediction at the same time, never against cumulative travel
        from the last accepted point. A platform may therefore travel farther
        than 0.50 m over normal accepted motion without faulting.
        """
        context = self.platform_motion_context
        candidate = self.cf231_health.raw_snapshot()
        now = time.monotonic()
        if context is None or candidate is None:
            self.enter_fault("cf231 平台运动确认上下文或原始位姿缺失")
            return
        position = self._position_from_message(candidate)
        expected = (context["anchor"] + context["prediction_velocity"]
                    * (now - context["anchor_time"]))
        if np.linalg.norm(position - expected) > PLATFORM_MOTION_MAX_PREDICTION_DEVIATION_M:
            self.enter_fault("cf231 相对离群前预测轨迹迁移 %.3f m > %.2f m" % (
                np.linalg.norm(position - expected), PLATFORM_MOTION_MAX_PREDICTION_DEVIATION_M))
            return
        candidates = context["candidates"]
        if not candidates:
            anchor_dt = now - context["anchor_time"]
            if anchor_dt <= 0.0:
                return
            anchor_speed = np.linalg.norm(position - context["anchor"]) / anchor_dt
            if anchor_speed > PLATFORM_MOTION_MAX_SPEED_MPS:
                return
        if candidates:
            previous_position, previous_time, _ = candidates[-1]
            dt = now - previous_time
            if dt <= 0.0 or dt > RESYNC_MAX_SAMPLE_GAP_S:
                candidates.clear()
                context["last_velocity"] = None
                return
            velocity = (position - previous_position) / dt
            speed = np.linalg.norm(velocity)
            if speed > PLATFORM_MOTION_MAX_SPEED_MPS:
                candidates.clear()
                context["last_velocity"] = None
                return
            previous_velocity = context["last_velocity"]
            if previous_velocity is not None:
                previous_speed = np.linalg.norm(previous_velocity)
                accel = np.linalg.norm(velocity - previous_velocity) / dt
                direction = (1.0 if previous_speed < 1e-6 or speed < 1e-6 else
                             float(np.dot(velocity, previous_velocity) / (speed * previous_speed)))
                if (accel > PLATFORM_MOTION_MAX_ACCEL_MPS2
                        or direction < PLATFORM_MOTION_MIN_DIRECTION_COS):
                    candidates.clear()
                    context["last_velocity"] = None
                    return
            context["last_velocity"] = velocity
        candidates.append((position, now, copy.deepcopy(candidate)))
        if len(candidates) < PLATFORM_MOTION_CONFIRM_SAMPLES:
            return
        duration = now - candidates[0][1]
        if duration < PLATFORM_MOTION_CONFIRM_DURATION_S:
            return
        self.cf231_health.commit_airborne_recovery(
            [(item[2], item[1]) for item in candidates]
        )
        self._observe_platform_motion(self.cf231_health.snapshot())
        rospy.loginfo("FOLLOW_HOLD[%s] 确认 cf231 连续运动：%d 样本、%.3f s；转 FOLLOW_ACQUIRE。",
                      HOLD_PLATFORM_MOTION, len(candidates), duration)
        self._clear_follow_hold()
        self._begin_follow_acquire()

    def _target_gate(self, target, yaw, step_reference=None):
        health_reasons = self._command_reasons()
        if health_reasons:
            rospy.logwarn(
                "goTo 拒绝：命令门控失败；target=(%.3f, %.3f, %.3f), yaw=%.3f；%s",
                target[0], target[1], target[2], yaw, "; ".join(health_reasons),
            )
            self.enter_fault("; ".join(health_reasons))
            return False
        cf17_msg = self.cf17_health.snapshot()
        if step_reference is None:
            step_reference = self.previous_target
        reasons = target_reasons(target, yaw, cf17_msg.pose, step_reference)
        if reasons:
            rospy.logwarn(
                "goTo 拒绝：目标门控失败；target=(%.3f, %.3f, %.3f), yaw=%.3f；%s",
                target[0], target[1], target[2], yaw, "; ".join(reasons),
            )
            self.enter_fault("; ".join(reasons))
            return False
        return True

    def _follow_geometry_reasons(self, target, yaw,
                                 horizontal_limit=FOLLOW_NORMAL_DISTANCE_M):
        """Final FOLLOW feasibility, excluding a command-to-command step.

        ``horizontal_limit`` only selects the FOLLOW admission envelope.  It
        never relaxes the target-height or vertical-distance checks below.
        """
        cf17_msg = self.cf17_health.snapshot()
        return target_reasons(
            target, yaw, cf17_msg.pose, previous_target=None,
            horizontal_limit=horizontal_limit,
        )

    def _follow_target_horizontal_distance(self, target):
        cf17_msg = self.cf17_health.snapshot()
        return float(np.linalg.norm(
            np.asarray(target, dtype=float)[:2]
            - np.array([cf17_msg.pose.position.x, cf17_msg.pose.position.y])
        ))

    def _enter_follow_target_hold(self, target, reasons):
        """Classify final-goal geometry without treating it as a pose fault."""
        if np.all(np.isfinite(target)):
            horizontal = self._follow_target_horizontal_distance(target)
            if horizontal > FOLLOW_ACQUIRE_MAX_DISTANCE_M:
                self._enter_follow_hold(
                    HOLD_TARGET_DISTANCE,
                    "目标水平距离 %.3f m > FOLLOW 接入范围 %.2f m；等待回到 %.2f m 内"
                    % (horizontal, FOLLOW_ACQUIRE_MAX_DISTANCE_M,
                       FOLLOW_ACQUIRE_RECOVERY_DISTANCE_M),
                )
                return
        self._enter_follow_hold(HOLD_TARGET, "; ".join(reasons))

    def _follow_target_or_hold(self, target, yaw):
        # Test the final target before limiting it.  Thus a short intermediate
        # waypoint can never hide an invalid height or vertical separation.
        reasons = self._follow_target_admission_reasons(target, yaw)
        if not reasons:
            horizontal = self._follow_target_horizontal_distance(target)
            if horizontal > FOLLOW_NORMAL_DISTANCE_M:
                self._begin_follow_acquire(final_target=target, final_yaw=yaw)
                return False
            return True
        if any("NaN/Inf" in reason for reason in reasons):
            self.enter_fault("FOLLOW 目标编程安全违规: " + "; ".join(reasons))
        else:
            self._enter_follow_target_hold(target, reasons)
        return False

    def _send_target(self, target, yaw, duration, step_reference=None):
        if not hasattr(self, "cf17_command_lock"):
            self.cf17_command_lock = threading.RLock()
        with self.cf17_command_lock:
            if self.flight_mode not in (
                self.MODE_MANUAL, self.MODE_FOLLOW, self.MODE_FOLLOW_ACQUIRE,
            ):
                rospy.logwarn("goTo 拒绝：当前模式=%s，不允许新的飞行目标。", self.flight_mode)
                return False
            if not self._target_gate(target, yaw, step_reference=step_reference):
                return False
            try:
                self.cf17.goTo(target, yaw=yaw, duration=duration, relative=False)
            except Exception as exc:
                self.enter_fault("goTo 服务调用失败: %s" % exc)
                return False
            self.previous_target = np.array(target, dtype=float, copy=True)
            self.last_successful_target = np.array(target, dtype=float, copy=True)
            self.last_successful_target_yaw = float(yaw)
            self.last_successful_target_duration = float(duration)
            self.last_successful_target_monotonic = time.monotonic()
            return True

    def _next_limited_follow_command(self, final_target, now):
        """Return a bounded intermediate command without changing state."""
        if self.previous_target is None:
            self.enter_fault("FOLLOW 限速器缺少最后成功 goTo 目标")
            return None, None, None
        last_time = getattr(self, "follow_command_last_sent_monotonic", None)
        dt = 0.2 if last_time is None else now - last_time
        if dt <= 0.0:
            return None, None, None
        dt = min(dt, FOLLOW_COMMAND_DT_MAX_S)
        next_speed = min(
            FOLLOW_MAX_COMMAND_SPEED_MPS,
            getattr(self, "follow_command_speed_mps", 0.0) + FOLLOW_MAX_COMMAND_ACCEL_MPS2 * dt,
        )
        max_step = min(MAX_TARGET_STEP_M, next_speed * dt)
        delta = final_target - self.previous_target
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9 or max_step <= 0.0:
            return np.array(final_target, dtype=float, copy=True), next_speed, dt
        step = min(distance, max_step)
        command = self.previous_target + delta * (step / distance)
        duration = max(FOLLOW_COMMAND_MIN_DURATION_S, step / max(next_speed, 1e-6))
        return command, next_speed, duration

    def send_follow_target(self, duration=None):
        """Generate a rate/acceleration-limited target from cf231 in FOLLOW."""
        if self.flight_mode != self.MODE_FOLLOW:
            rospy.logwarn("拒绝 FOLLOW 目标：当前不在跟随模式。")
            return False
        if not hasattr(self, "follow_command_lock"):
            self.follow_command_lock = threading.Lock()
        if not self.follow_command_lock.acquire(blocking=False):
            return False
        try:
            if not self._follow_command_allowed():
                return False
            final_target, yaw = self._follow_target_from_latest_platform()
            # Validate the final, unconstrained geometry before calculating an
            # intermediate point; a limiter must never hide an infeasible goal.
            if not self._follow_target_or_hold(final_target, yaw):
                return False
            previous_target = np.array(self.previous_target, dtype=float, copy=True)
            command, next_speed, command_duration = self._next_limited_follow_command(
                final_target, time.monotonic())
            if command is None:
                return False
            sent = self._send_target(command, yaw, command_duration)
            if sent:
                self.follow_command_speed_mps = next_speed
                self.follow_command_last_sent_monotonic = time.monotonic()
                now = self.follow_command_last_sent_monotonic
                if (getattr(self, "follow_command_last_log_monotonic", None) is None
                        or now - self.follow_command_last_log_monotonic >= 1.0):
                    self.follow_command_last_log_monotonic = now
                    rospy.loginfo(
                        "FOLLOW 限速 goTo：step=%.3f m, speed<=%.3f m/s, duration=%.3f s",
                        np.linalg.norm(command - previous_target), next_speed, command_duration,
                    )
            return sent
        finally:
            self.follow_command_lock.release()

    def _follow_target_from_latest_platform(self):
        platform_msg = self.cf231_health.snapshot()
        return self.get_pad_global_target(
            platform_msg.pose,
            -0.10 + self.manual_x,
            self.manual_y,
            self.manual_z,
        )

    def _follow_target_admission_reasons(self, target, yaw):
        """Validate final geometry for both normal FOLLOW and acquisition.

        The wider 1.20 m envelope is only an admission range for conservative
        acquisition.  Height and vertical limits remain the normal hard
        limits, and command waypoints still pass the ordinary target gate.
        """
        return self._follow_geometry_reasons(
            target, yaw, horizontal_limit=FOLLOW_ACQUIRE_MAX_DISTANCE_M)

    def _capture_manual_reference(self):
        """Capture only live cf17 state; this intentionally never writes previous_target."""
        cf17_msg = self.cf17_health.snapshot()
        self.manual_reference_target = np.array(
            [
                cf17_msg.pose.position.x,
                cf17_msg.pose.position.y,
                cf17_msg.pose.position.z,
            ],
            dtype=float,
        )
        self.manual_target = np.array(self.manual_reference_target, dtype=float, copy=True)
        self.manual_yaw = self.pose_yaw(cf17_msg.pose)

    def _return_to_manual_after_acquire_rejection(self, reason):
        self._capture_manual_reference()
        self.acquire_anchor = None
        self.flight_mode = self.MODE_MANUAL
        rospy.logwarn("FOLLOW 接入拒绝：%s；保持 MANUAL，未发送 goTo。", reason)

    def send_follow_acquire_target(self):
        """Send at most one conservative, gated waypoint toward current FOLLOW."""
        if not self._follow_command_allowed():
            return False
        final_target, yaw = self._follow_target_from_latest_platform()
        admission_reasons = self._follow_target_admission_reasons(final_target, yaw)
        if admission_reasons:
            if any("NaN/Inf" in reason for reason in admission_reasons):
                self.enter_fault("FOLLOW 接入目标编程安全违规: " + "; ".join(admission_reasons))
            else:
                self._enter_follow_target_hold(final_target, admission_reasons)
            return False
        if self.acquire_anchor is None:
            self.enter_fault("FOLLOW 接入锚点缺失")
            return False
        delta = final_target - self.acquire_anchor
        distance = float(np.linalg.norm(delta))
        if distance <= FOLLOW_ACQUIRE_COMPLETE_TOLERANCE_M:
            self.acquire_anchor = None
            self.flight_mode = self.MODE_FOLLOW
            self.follow_command_speed_mps = FOLLOW_ACQUIRE_MAX_SPEED_MPS
            self.follow_command_last_sent_monotonic = time.monotonic()
            rospy.loginfo("FOLLOW 接入完成：误差 %.3f m，切换到 FOLLOW。", distance)
            return True
        step_distance = min(distance, FOLLOW_ACQUIRE_MAX_STEP_M)
        waypoint = self.acquire_anchor + delta * (step_distance / distance)
        duration = max(
            FOLLOW_ACQUIRE_MIN_DURATION_S,
            step_distance / FOLLOW_ACQUIRE_MAX_SPEED_MPS,
        )
        sent = self._send_target(
            waypoint, yaw, duration, step_reference=self.acquire_anchor
        )
        if sent:
            self.acquire_anchor = np.array(waypoint, dtype=float, copy=True)
            rospy.loginfo(
                "FOLLOW 接入 goTo：step=%.3f m, speed<=%.3f m/s, duration=%.3f s",
                step_distance, FOLLOW_ACQUIRE_MAX_SPEED_MPS, duration,
            )
        return sent

    def send_manual_target(self, duration=0.3):
        """Send the MANUAL-owned target without consulting cf231 pose."""
        if self.flight_mode != self.MODE_MANUAL:
            rospy.logwarn("拒绝 MANUAL 目标：当前不在手动模式。")
            return False
        if self.manual_target is None or self.manual_yaw is None:
            rospy.logwarn("MANUAL goTo 拒绝：手动目标基准缺失。")
            self.enter_fault("手动目标基准缺失")
            return False
        if self.soft_rate_states():
            rospy.logwarn("MANUAL goTo 拒绝：位姿频率低于 %.2f Hz，未发送 goTo。", MIN_POSE_RATE_HZ)
            return False
        sent = self._send_target(self.manual_target, self.manual_yaw, duration)
        if sent:
            rospy.loginfo(
                "MANUAL goTo 已发送：target=(%.3f, %.3f, %.3f), yaw=%.3f, "
                "duration=%.3f, relative=False",
                self.manual_target[0], self.manual_target[1], self.manual_target[2],
                self.manual_yaw, duration,
            )
        else:
            rospy.logwarn(
                "MANUAL goTo 未发送：target=(%.3f, %.3f, %.3f), yaw=%.3f",
                self.manual_target[0], self.manual_target[1], self.manual_target[2],
                self.manual_yaw,
            )
        return sent

    def track_loop(self, _event):
        if self.flight_mode == self.MODE_TAKEOFF_WAIT_BRUSHED:
            self.update_takeoff_wait()
        elif self.flight_mode == self.MODE_LANDING:
            self.update_landing()
        elif self.is_flying and self.flight_mode == self.MODE_FOLLOW_ACQUIRE:
            self.send_follow_acquire_target()
        elif self.is_flying and self.flight_mode == self.MODE_FOLLOW_HOLD:
            self.update_follow_hold()
        elif self.is_flying and self.flight_mode == self.MODE_FOLLOW:
            self.send_follow_target()

    def takeoff(self):
        if self.is_flying or self.flight_mode == self.MODE_TAKEOFF_WAIT_BRUSHED:
            rospy.logwarn("起飞已在进行或等待有刷起飞确认。")
            return
        reasons = self.localization_reasons()
        if reasons:
            rospy.logwarn("拒绝起飞：%s", "; ".join(reasons))
            return
        platform_msg = self.cf231_health.snapshot()
        cf17_msg = self.cf17_health.snapshot()
        start_height = platform_msg.pose.position.z + 0.30
        if not TARGET_HEIGHT_MIN_M <= start_height <= TARGET_HEIGHT_MAX_M:
            rospy.logwarn("拒绝起飞：目标高度 %.3f m 超出安全范围", start_height)
            return
        try:
            self.cf17.takeoff(targetHeight=start_height, duration=TAKEOFF_DURATION_S)
        except Exception as exc:
            rospy.logerr("有刷 takeoff 服务未接受请求：%s", exc)
            return
        self.takeoff_target_height = start_height
        self.takeoff_start_height = cf17_msg.pose.position.z
        self.takeoff_started_monotonic = time.monotonic()
        self.takeoff_height_stable_samples = 0
        self.takeoff_valid_samples = 0
        self.flight_mode = self.MODE_TAKEOFF_WAIT_BRUSHED
        rospy.loginfo("有刷起飞请求已接受；进入 TAKEOFF_WAIT_BRUSHED，等待连续健康高度上升确认。")

    def update_takeoff_wait(self):
        now = time.monotonic()
        if self.takeoff_started_monotonic is None or self.takeoff_target_height is None or self.takeoff_start_height is None:
            self.flight_mode = self.MODE_TAKEOFF_UNCONFIRMED
            rospy.logerr("TAKEOFF_UNCONFIRMED：起飞确认状态缺失；不发送 goTo，请按 s 请求 Land。")
            return
        elapsed = now - self.takeoff_started_monotonic
        hard = self.hard_localization_reasons()
        if hard:
            self.flight_mode = self.MODE_TAKEOFF_UNCONFIRMED
            rospy.logerr("TAKEOFF_UNCONFIRMED：起飞状态未确认，位姿硬故障：%s；不自动 stop，请按 s 请求 Land。", "; ".join(hard))
            return
        if elapsed > TAKEOFF_DURATION_S + TAKEOFF_COMPLETION_TIMEOUT_S:
            self.flight_mode = self.MODE_TAKEOFF_UNCONFIRMED
            rospy.logerr("TAKEOFF_UNCONFIRMED：服务曾成功但未在时限内确认离地；不发送 goTo，请按 s 请求 Land。")
            return
        if self.soft_rate_states() or elapsed < TAKEOFF_DURATION_S + TAKEOFF_SETTLE_MARGIN_S:
            self.takeoff_height_stable_samples = 0
            return
        cf17_msg = self.cf17_health.snapshot()
        rise = cf17_msg.pose.position.z - self.takeoff_start_height
        safe_height = cf17_msg.pose.position.z >= TAKEOFF_MIN_SAFE_HEIGHT_M
        near_target = abs(cf17_msg.pose.position.z - self.takeoff_target_height) <= TAKEOFF_HEIGHT_TOLERANCE_M
        if rise < TAKEOFF_MIN_RISE_M or not safe_height or not near_target:
            self.takeoff_height_stable_samples = 0
            return
        self.takeoff_valid_samples += 1
        self.takeoff_height_stable_samples += 1
        if self.takeoff_height_stable_samples < TAKEOFF_STABLE_SAMPLES:
            return
        self.is_flying = True
        self.flight_mode = self.MODE_IDLE
        self.start_manual_mode()
        if self.flight_mode == self.MODE_MANUAL:
            rospy.loginfo("有刷起飞确认通过：连续健康高度上升，已进入 MANUAL。")

    def start_tracking(self):
        if self.flight_mode == self.MODE_TAKEOFF_WAIT_BRUSHED:
            rospy.logwarn("TAKEOFF_WAIT 中拒绝进入 FOLLOW；等待起飞轨迹和高度稳定。")
            return
        if self.flight_mode == self.MODE_FOLLOW_HOLD:
            rospy.logwarn("FOLLOW_HOLD 中拒绝 f；仅允许恢复状态机或按 s 正常降落。")
            return
        if not self.is_flying:
            rospy.logwarn("飞机尚未起飞；拒绝进入跟随模式。")
            return
        reasons = self._command_reasons()
        if reasons:
            self.enter_fault("; ".join(reasons))
            return
        rate_states = self.soft_rate_states()
        if rate_states:
            self._enter_follow_hold(
                HOLD_RATE,
                "; ".join("%s=%.2f Hz, age=%.3f s" % state for state in rate_states),
            )
            return
        self._begin_follow_acquire()

    def _begin_follow_acquire(self, final_target=None, final_yaw=None):
        """Enter acquisition from a validated live cf17 pose; no immediate goTo."""
        if final_target is None or final_yaw is None:
            final_target, final_yaw = self._follow_target_from_latest_platform()
        admission_reasons = self._follow_target_admission_reasons(final_target, final_yaw)
        if admission_reasons:
            if any("NaN/Inf" in reason for reason in admission_reasons):
                self.enter_fault("FOLLOW 接入目标编程安全违规: " + "; ".join(admission_reasons))
            else:
                self._enter_follow_target_hold(final_target, admission_reasons)
            return
        cf17_msg = self.cf17_health.snapshot()
        self.acquire_anchor = np.array(
            [cf17_msg.pose.position.x, cf17_msg.pose.position.y, cf17_msg.pose.position.z],
            dtype=float,
        )
        self.flight_mode = self.MODE_FOLLOW_ACQUIRE
        rospy.loginfo(
            "进入 FOLLOW_ACQUIRE：以最新 cf17 位姿接入，未直接发送最终 FOLLOW 目标；"
            "正常距离<=%.2f m，接入上限<=%.2f m。",
            FOLLOW_NORMAL_DISTANCE_M, FOLLOW_ACQUIRE_MAX_DISTANCE_M,
        )

    def start_manual_mode(self):
        if self.flight_mode == self.MODE_TAKEOFF_WAIT_BRUSHED:
            rospy.logwarn("TAKEOFF_WAIT 中拒绝进入 MANUAL；等待起飞轨迹和高度稳定。")
            return
        if self.flight_mode == self.MODE_FOLLOW_HOLD:
            rospy.logwarn("FOLLOW_HOLD 中拒绝 m；不得绕过 HOLD 进入 MANUAL。")
            return
        if not self.is_flying:
            rospy.logwarn("飞机尚未起飞，无法进入手动模式。")
            return
        reasons = self._command_reasons()
        if reasons:
            self.enter_fault("; ".join(reasons))
            return
        if self.soft_rate_states():
            rospy.logwarn("MANUAL 中拒绝进入/移动：位姿频率低于 %.2f Hz，未发送 goTo。", MIN_POSE_RATE_HZ)
            return
        self._capture_manual_reference()
        self.acquire_anchor = None
        self.flight_mode = self.MODE_MANUAL
        rospy.loginfo(
            "进入手动微调模式，自动跟随已暂停；MANUAL 基准=(%.3f, %.3f, %.3f), "
            "锁定 yaw=%.3f rad。",
            self.manual_reference_target[0], self.manual_reference_target[1],
            self.manual_reference_target[2], self.manual_yaw,
        )

    def adjust_manual_target(self, dx=0.0, dy=0.0, dz=0.0):
        if not self.is_flying or self.flight_mode != self.MODE_MANUAL:
            rospy.logwarn(
                "MANUAL 按键拒绝：flying=%s, mode=%s；仅飞行中的手动模式允许微调。",
                self.is_flying, self.flight_mode,
            )
            return
        now = time.monotonic()
        if now - self.last_manual_command_time < self.manual_command_period:
            rospy.loginfo("MANUAL 按键忽略：%.3f s 节流窗口内。", self.manual_command_period)
            return
        command_reasons = self._command_reasons()
        if command_reasons:
            rospy.logwarn("MANUAL 按键拒绝：%s", "; ".join(command_reasons))
            self.enter_fault("; ".join(command_reasons))
            return
        if self.soft_rate_states():
            rospy.logwarn("MANUAL 按键拒绝：位姿频率低于 %.2f Hz，未发送 goTo。", MIN_POSE_RATE_HZ)
            return
        if self.manual_target is None or self.manual_reference_target is None:
            rospy.logwarn("MANUAL 按键拒绝：手动目标基准缺失。")
            self.enter_fault("手动目标基准缺失")
            return
        manual_before = np.array(self.manual_target, dtype=float, copy=True)
        local_x = self.manual_target[0] - self.manual_reference_target[0]
        local_y = self.manual_target[1] - self.manual_reference_target[1]
        cos_yaw = math.cos(self.manual_yaw)
        sin_yaw = math.sin(self.manual_yaw)
        body_x = cos_yaw * local_x + sin_yaw * local_y
        body_y = -sin_yaw * local_x + cos_yaw * local_y
        body_x = self.clamp(body_x + dx, self.max_xy_offset)
        body_y = self.clamp(body_y + dy, self.max_xy_offset)
        body_z = self.clamp(
            self.manual_target[2] - self.manual_reference_target[2] + dz,
            self.max_z_offset,
        )
        body_z = max(
            TARGET_HEIGHT_MIN_M - self.manual_reference_target[2],
            min(TARGET_HEIGHT_MAX_M - self.manual_reference_target[2], body_z),
        )
        candidate = self.manual_reference_target + np.array(
            [
                cos_yaw * body_x - sin_yaw * body_y,
                sin_yaw * body_x + cos_yaw * body_y,
                body_z,
            ],
            dtype=float,
        )
        world_delta = candidate - manual_before
        rospy.loginfo(
            "MANUAL 增量：local=(%.3f, %.3f, %.3f), world=(%.3f, %.3f, %.3f), "
            "target_before=(%.3f, %.3f, %.3f), target_after=(%.3f, %.3f, %.3f)",
            dx, dy, dz, world_delta[0], world_delta[1], world_delta[2],
            manual_before[0], manual_before[1], manual_before[2],
            candidate[0], candidate[1], candidate[2],
        )
        if np.allclose(candidate, manual_before, rtol=0.0, atol=1e-12):
            rospy.logwarn("MANUAL 按键拒绝：目标已在现有边界，未发送 goTo。")
            return
        self.last_manual_command_time = now
        self.manual_target = candidate
        self.send_manual_target(duration=0.3)

    def reset_manual_target(self):
        if self.flight_mode == self.MODE_TAKEOFF_WAIT_BRUSHED:
            rospy.logwarn("TAKEOFF_WAIT 中拒绝复位目标。")
            return
        if self.flight_mode == self.MODE_FOLLOW_HOLD:
            rospy.logwarn("FOLLOW_HOLD 中拒绝 r；不得产生或改变控制目标。")
            return
        if self.flight_mode == self.MODE_MANUAL:
            if self.manual_reference_target is None:
                self.enter_fault("手动目标基准缺失")
                return
            self.manual_target = np.array(
                self.manual_reference_target, dtype=float, copy=True
            )
            rospy.loginfo("手动目标已复位到进入 MANUAL 时的 cf17 基准；未发送 goTo。")
            return
        self.manual_x = self.manual_y = self.manual_z = 0.0
        rospy.loginfo("FOLLOW 手动校正已归零。")

    def land(self):
        # s is intentionally available for a possibly airborne unconfirmed takeoff.
        if not self.is_flying and self.flight_mode not in (self.MODE_TAKEOFF_WAIT_BRUSHED, self.MODE_TAKEOFF_UNCONFIRMED):
            return
        if not hasattr(self, "cf17_command_lock"):
            self.cf17_command_lock = threading.RLock()
        with self.cf17_command_lock:
            cf17_msg = self.cf17_health.snapshot()
            cf231_msg = self.cf231_health.snapshot()
            ground_reference = self.takeoff_start_height
            if ground_reference is None or not math.isfinite(ground_reference):
                # An unconfirmed takeoff may have no reliable pre-takeoff
                # sample. Preserve the legacy world-origin fallback instead
                # of inventing a height from an airborne pose.
                ground_reference = 0.0
            self.flight_mode = self.MODE_LANDING
            self.landing_started_monotonic = time.monotonic()
            self.landing_start_height = None if cf17_msg is None else cf17_msg.pose.position.z
            self.landing_target_height = float(ground_reference)
            self.landing_start_position = None if cf17_msg is None else np.array([
                cf17_msg.pose.position.x, cf17_msg.pose.position.y,
                cf17_msg.pose.position.z,
            ], dtype=float)
            self.landing_start_yaw = None if cf17_msg is None else self.pose_yaw(cf17_msg.pose)
            self.landing_stable_samples = 0
            self.landing_last_audit_log_monotonic = None
            cf17_text = "unavailable" if cf17_msg is None else (
                "xyz=(%.3f, %.3f, %.3f), yaw=%.3f" % (
                    self.landing_start_position[0], self.landing_start_position[1],
                    self.landing_start_position[2], self.landing_start_yaw))
            cf231_text = "unavailable" if cf231_msg is None else (
                "xyz=(%.3f, %.3f, %.3f), yaw=%.3f" % (
                    cf231_msg.pose.position.x, cf231_msg.pose.position.y,
                    cf231_msg.pose.position.z, self.pose_yaw(cf231_msg.pose)))
            last_target = getattr(self, "last_successful_target", None)
            last_target_text = "none" if last_target is None else (
                "xyz=(%.3f, %.3f, %.3f), yaw=%.3f, duration=%.3f" % (
                    last_target[0], last_target[1], last_target[2],
                    self.last_successful_target_yaw, self.last_successful_target_duration))
            rospy.loginfo(
                "LAND_AUDIT_START: mode->landing; cf17=%s; cf231=%s; "
                "previous_target=%s; manual_target=%s; manual_offset=(%.3f, %.3f, %.3f); "
                "last_successful_goTo=%s; ground_reference=%.3f; "
                "Land(height=%.3f, duration=%.3f, groupMask=0).",
                cf17_text, cf231_text, self.previous_target, getattr(self, "manual_target", None),
                getattr(self, "manual_x", 0.0), getattr(self, "manual_y", 0.0),
                getattr(self, "manual_z", 0.0), last_target_text,
                self.landing_target_height, self.landing_target_height, LAND_DURATION_S,
            )
            try:
                self.cf17.land(
                    targetHeight=self.landing_target_height,
                    duration=LAND_DURATION_S,
                )
            except Exception as exc:
                self.flight_mode = self.MODE_LANDING_UNCONFIRMED
                rospy.logerr("LANDING_UNCONFIRMED：land 服务失败：%s；请人工确认，不自动 stop。", exc)
                return
        rospy.loginfo("有刷 land 请求已接受；等待连续健康下降与落地高度确认。")

    def update_landing(self):
        if self.flight_mode != self.MODE_LANDING:
            return
        now = time.monotonic()
        if self.landing_started_monotonic is None or self.landing_target_height is None:
            self.flight_mode = self.MODE_LANDING_UNCONFIRMED
            return
        hard = self.cf17_health.hard_reasons(now)
        if hard:
            self.flight_mode = self.MODE_LANDING_UNCONFIRMED
            rospy.logerr("LANDING_UNCONFIRMED：降落中位姿失效；不声称已落地，请人工确认。")
            return
        if now - self.landing_started_monotonic > LAND_DURATION_S + LAND_CONFIRM_TIMEOUT_S:
            self.flight_mode = self.MODE_LANDING_UNCONFIRMED
            rospy.logerr("LANDING_UNCONFIRMED：未在时限内确认落地；请人工确认。")
            return
        msg = self.cf17_health.snapshot()
        position = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
        ], dtype=float)
        start = getattr(self, "landing_start_position", None)
        if start is not None:
            delta = position - start
            horizontal = float(np.linalg.norm(delta[:2]))
            last_log = getattr(self, "landing_last_audit_log_monotonic", None)
            if last_log is None or now - last_log >= LAND_AUDIT_LOG_PERIOD_S:
                self.landing_last_audit_log_monotonic = now
                rospy.loginfo(
                    "LAND_AUDIT_PROGRESS: elapsed=%.2f s; cf17_xyz=(%.3f, %.3f, %.3f); "
                    "delta_xyz=(%.3f, %.3f, %.3f); horizontal=%.3f m; stable=%d/%d.",
                    now - self.landing_started_monotonic, position[0], position[1], position[2],
                    delta[0], delta[1], delta[2], horizontal,
                    self.landing_stable_samples, LAND_STABLE_SAMPLES,
                )
        descended = self.landing_start_height is None or msg.pose.position.z <= self.landing_start_height - 0.05
        if (msg.pose.position.z > self.landing_target_height + LAND_GROUND_HEIGHT_TOLERANCE_M
                or not descended):
            self.landing_stable_samples = 0
            return
        self.landing_stable_samples += 1
        if self.landing_stable_samples < LAND_STABLE_SAMPLES:
            return
        self.is_flying = False
        self.previous_target = None
        self._clear_follow_hold()
        self.flight_mode = self.MODE_IDLE
        if start is not None:
            delta = position - start
            rospy.loginfo(
                "LAND_AUDIT_CONFIRM: final_xyz=(%.3f, %.3f, %.3f); "
                "delta_world=(%.3f, %.3f, %.3f); horizontal=%.3f m; start_yaw=%.3f.",
                position[0], position[1], position[2], delta[0], delta[1], delta[2],
                float(np.linalg.norm(delta[:2])), self.landing_start_yaw,
            )
        rospy.loginfo("有刷降落已由连续健康位姿确认。")


def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def print_help():
    print("状态启动为 IDLE；等待 cf17/cf231 连续健康定位后才允许起飞。")
    print("w 起飞；f 跟随；m 手动；r 校正归零；s 降落；c 人工确认恢复")
    print("手动模式：i/k 前后，j/l 左右，u/o 上下（每次 20 mm）")
    print("Ctrl+C：飞行或起飞状态未确认时请求正常 Land；不发送显式解锁控制")
    print("所有飞行命令仅发送给 cf17；cf231 仅提供 /cf231/pose")


def make_test_pose(stamp, x=0.0, frame=EXPECTED_FRAME, quaternion=(0, 0, 0, 1)):
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time.from_sec(stamp)
    msg.pose.position.x = x
    (
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    ) = quaternion
    return msg


def run_self_tests():
    """Static/offline safety checks; never initializes ROS or radio."""
    source = Path(__file__).read_text()
    forbidden = ("request" + "Arm", "supervisor" + "State", "/cf17/" + "arm")
    assert not any(item in source for item in forbidden)
    assert "TAKEOFF_WAIT_BRUSHED" in source
    assert "LANDING_UNCONFIRMED" in source
    for required in (
        "FOLLOW_NORMAL_DISTANCE_M = 0.80", "FOLLOW_ACQUIRE_MAX_DISTANCE_M = 1.20",
        "FOLLOW_MAX_COMMAND_SPEED_MPS = 0.40", "FOLLOW_MAX_COMMAND_ACCEL_MPS2 = 0.60",
        "MAX_TARGET_STEP_M = 0.20", "HOLD_PLATFORM_MOTION", "FOLLOW_HOLD",
        "FOLLOW_ACQUIRE", "previous_target", "follow_command_lock", "xy_step = 0.020",
    ):
        assert required in source, required
    assert not any("self.cf231." + command in source
                   for command in ("takeoff", "land", "goTo", "stop"))
    health = PoseHealth("test")
    assert normalize_pose_message(make_test_pose(1.0))[0] is not None
    assert normalize_pose_message(make_test_pose(1.0, frame="bad"))[0] is None
    print("SELF-TEST PASS: legacy brushed has no explicit arming/Supervisor dependency")


def run_manual_mode_state_machine_tests():
    """Offline state-machine tests with no ROS transport or hardware objects."""
    class FakeCF:
        def __init__(self, fail_takeoff=False, fail_land=False): self.calls=[]; self.fail_takeoff=fail_takeoff; self.fail_land=fail_land; self.last_land_kwargs=None
        def takeoff(self, **_):
            self.calls.append("takeoff")
            if self.fail_takeoff: raise RuntimeError("offline")
        def land(self, **_):
            self.calls.append("land")
            self.last_land_kwargs = dict(_)
            if self.fail_land: raise RuntimeError("offline")
        def goTo(self, *_a, **_k): self.calls.append("goTo")
    class Health:
        def __init__(self, z): self.z=z; self.bad=[]
        def snapshot(self): return make_test_pose(1.0, x=0.0)._replace if False else pose(self.z)
        def health_reasons(self): return list(self.bad)
        def hard_reasons(self, *_): return list(self.bad)
        def soft_rate_state(self): return None
        def summary(self): return "offline"
        def latch_fault(self, *_): pass
    def pose(z):
        p=make_test_pose(1.0); p.pose.position.z=z; return p
    t=TrackerFlight.__new__(TrackerFlight)
    t.cf17=FakeCF(); t.cf231=object(); t.cf17_health=Health(0.0); t.cf231_health=Health(0.4)
    t.is_flying=False; t.flight_mode=t.MODE_IDLE; t.takeoff_target_height=None; t.takeoff_start_height=None; t.takeoff_started_monotonic=None; t.takeoff_height_stable_samples=0; t.takeoff_valid_samples=0
    t.previous_target=None; t._clear_follow_hold=lambda: None; t.start_manual_mode=lambda: setattr(t, 'flight_mode', t.MODE_MANUAL)
    t.localization_reasons=lambda: []; t.hard_localization_reasons=lambda: t.cf17_health.hard_reasons()+t.cf231_health.hard_reasons(); t.soft_rate_states=lambda: []
    t.takeoff(); assert t.cf17.calls == ["takeoff"] and not t.is_flying
    # Success response but no rise cannot claim flight and eventually becomes unconfirmed.
    t.takeoff_started_monotonic=time.monotonic()-TAKEOFF_DURATION_S-TAKEOFF_COMPLETION_TIMEOUT_S-0.1; t.update_takeoff_wait(); assert not t.is_flying and t.flight_mode==t.MODE_TAKEOFF_UNCONFIRMED
    # A new attempt with continuous healthy height rise is the only path to MANUAL/flying.
    t.flight_mode=t.MODE_IDLE; t.cf17.calls=[]; t.takeoff(); t.cf17_health.z=t.takeoff_target_height
    t.takeoff_started_monotonic=time.monotonic()-TAKEOFF_DURATION_S-TAKEOFF_SETTLE_MARGIN_S-0.1
    for _ in range(TAKEOFF_STABLE_SAMPLES): t.update_takeoff_wait()
    assert t.is_flying and t.flight_mode==t.MODE_MANUAL and "goTo" not in t.cf17.calls
    # Landing success without descent stays unconfirmed after timeout; successful descent proves idle.
    t.land(); assert "land" in t.cf17.calls
    t.landing_started_monotonic=time.monotonic()-LAND_DURATION_S-LAND_CONFIRM_TIMEOUT_S-0.1; t.update_landing(); assert t.flight_mode==t.MODE_LANDING_UNCONFIRMED
    t.flight_mode=t.MODE_MANUAL; t.is_flying=True; t.land(); t.cf17_health.z=0.0
    for _ in range(LAND_STABLE_SAMPLES): t.update_landing()
    assert not t.is_flying and t.flight_mode==t.MODE_IDLE
    # A brushed vehicle may have a non-zero Lighthouse ground height. Land
    # must reuse its pre-takeoff cf17 reference instead of forcing world z=0.
    t.flight_mode=t.MODE_MANUAL; t.is_flying=True; t.takeoff_start_height=0.62; t.cf17_health.z=0.90
    t.land(); assert t.cf17.last_land_kwargs["targetHeight"] == 0.62
    # A Timer callback that reaches _send_target after Land won the shared
    # command lock must not issue a trailing goTo.
    before_calls = list(t.cf17.calls)
    assert not t._send_target(np.array([0.0, 0.0, 0.3]), 0.0, 0.3)
    assert t.cf17.calls == before_calls
    assert not hasattr(t.cf231, "goTo")
    print("OFFLINE STATE-MACHINE PASS: landing pose confirmation, no trailing goTo, and cf231 zero control")
def main():
    if "--self-test" in sys.argv:
        run_self_tests()
        return
    if "--offline-state-machine-test" in sys.argv:
        run_manual_mode_state_machine_tests()
        return
    import rospkg

    package_path = rospkg.RosPack().get_path("crazyflie_test_ros1_brushed")
    tracker = TrackerFlight(package_path + "/config/crazyflies_brushed.yaml")
    settings = termios.tcgetattr(sys.stdin)
    print_help()
    try:
        while not rospy.is_shutdown():
            key = get_key(settings)
            if key.lower() == "w":
                tracker.takeoff()
            elif key.lower() == "f":
                tracker.start_tracking()
            elif key.lower() == "m":
                tracker.start_manual_mode()
            elif key.lower() == "c":
                tracker.confirm_recovery()
            elif key.lower() == "i":
                rospy.loginfo("收到手动按键=i，当前模式=%s", tracker.flight_mode)
                tracker.adjust_manual_target(dx=tracker.xy_step)
            elif key.lower() == "k":
                rospy.loginfo("收到手动按键=k，当前模式=%s", tracker.flight_mode)
                tracker.adjust_manual_target(dx=-tracker.xy_step)
            elif key.lower() == "j":
                rospy.loginfo("收到手动按键=j，当前模式=%s", tracker.flight_mode)
                tracker.adjust_manual_target(dy=tracker.xy_step)
            elif key.lower() == "l":
                rospy.loginfo("收到手动按键=l，当前模式=%s", tracker.flight_mode)
                tracker.adjust_manual_target(dy=-tracker.xy_step)
            elif key.lower() == "u":
                rospy.loginfo("收到手动按键=u，当前模式=%s", tracker.flight_mode)
                tracker.adjust_manual_target(dz=tracker.z_step)
            elif key.lower() == "o":
                rospy.loginfo("收到手动按键=o，当前模式=%s", tracker.flight_mode)
                tracker.adjust_manual_target(dz=-tracker.z_step)
            elif key.lower() == "r":
                tracker.reset_manual_target()
            elif key.lower() == "s":
                tracker.land()
            elif key == "\x03":
                break
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        if tracker.is_flying or tracker.flight_mode in (tracker.MODE_TAKEOFF_WAIT_BRUSHED,
                                                        tracker.MODE_TAKEOFF_UNCONFIRMED):
            tracker.land()


if __name__ == "__main__":
    main()
