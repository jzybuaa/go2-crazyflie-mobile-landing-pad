#!/usr/bin/env python3
"""Track a Lighthouse-localized dog platform with cf17 only."""

import copy
import math
import select
import sys
import termios
import threading
import time
import tty
import types
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

# Supervisor state is published at 10 Hz. Use the same 0.55 s tolerance as
# pose health so an isolated measured 0.31 s radio gap does not trip the gate.
SUPERVISOR_TIMEOUT_S = 0.55
TAKEOFF_FLYING_CONFIRM_TIMEOUT_S = 4.0
LAND_STOPPED_CONFIRM_TIMEOUT_S = 5.0
SUPERVISOR_POLL_PERIOD_S = 0.05

# Takeoff completion is deliberately distinct from Supervisor isFlying=1: the
# latter confirms thrust, while the 2 s takeoff trajectory may still be moving.
TAKEOFF_DURATION_S = 2.0
TAKEOFF_SETTLE_MARGIN_S = 0.25
TAKEOFF_HEIGHT_TOLERANCE_M = 0.05
TAKEOFF_STABLE_SAMPLES = 5  # 5 x 5 Hz timer samples = about one stable second.
TAKEOFF_COMPLETION_TIMEOUT_S = 5.0

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
    MODE_TAKEOFF_WAIT = "takeoff_wait"
    MODE_FOLLOW = "follow"
    MODE_FOLLOW_ACQUIRE = "follow_acquire"
    MODE_FOLLOW_HOLD = "follow_hold"
    MODE_LANDING = "landing"
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
        self.is_armed = False
        self.flight_mode = self.MODE_IDLE
        # Only _send_target() writes this: it is the last successfully sent
        # absolute goTo target, never a pose, candidate, or MANUAL reference.
        self.previous_target = None
        self.takeoff_target_height = None
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
        # rospy.Timer callbacks normally serialize, but protect the target
        # generator as the goTo service may release the interpreter lock.
        self.follow_command_lock = threading.Lock()

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
        state = self.supervisor_state()
        allow_ground_resync = (
            not self.is_flying
            and not self.is_armed
            and state is not None
            and not state["isArmed"]
            and not state["isFlying"]
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

    def supervisor_state(self):
        return self.cf17.supervisorState(maxAge=SUPERVISOR_TIMEOUT_S)

    def supervisor_reasons(self, require_armed=False, require_flying=False):
        state = self.supervisor_state()
        if state is None:
            return ["cf17 supervisor.info 缺失或过期"]
        reasons = []
        if state["isLocked"]:
            reasons.append("cf17 Supervisor locked")
        if state["isTumbled"]:
            reasons.append("cf17 Supervisor tumbled")
        if require_armed and not state["isArmed"]:
            reasons.append("cf17 isArmed=0")
        if require_armed and not state["canFly"]:
            reasons.append("cf17 canFly=0")
        if require_flying and not state["isFlying"]:
            reasons.append("cf17 isFlying=0")
        return reasons

    def wait_for_supervisor(self, predicate, timeout_s):
        deadline = time.monotonic() + timeout_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            state = self.supervisor_state()
            if state is not None and predicate(state):
                return state
            rospy.sleep(SUPERVISOR_POLL_PERIOD_S)
        return None

    def safe_disarm_if_not_flying(self, context):
        state = self.supervisor_state()
        if state is None:
            rospy.logerr("%s：Supervisor 状态不可用，无法安全确认 disarm。", context)
            return False
        if state["isFlying"]:
            rospy.logerr("%s：isFlying=1，拒绝 disarm。", context)
            return False
        if not state["isArmed"]:
            self.is_armed = False
            return True
        try:
            response = self.cf17.requestArm(False)
        except Exception as exc:
            rospy.logerr("%s：disarm 服务失败: %s", context, exc)
            return False
        if not response.success or response.is_armed:
            rospy.logerr("%s：disarm 未被状态确认: %s", context, response.message)
            return False
        self.is_armed = False
        return True

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
            "状态=%s, flying=%s | %s | %s | 定位=%s",
            self.flight_mode,
            self.is_flying,
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
        health_reasons.extend(
            self.supervisor_reasons(require_armed=True, require_flying=True)
        )
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
        reasons = self.supervisor_reasons(require_armed=True, require_flying=True)
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
        if not self._target_gate(target, yaw, step_reference=step_reference):
            return False
        try:
            self.cf17.goTo(target, yaw=yaw, duration=duration, relative=False)
        except Exception as exc:
            self.enter_fault("goTo 服务调用失败: %s" % exc)
            return False
        self.previous_target = np.array(target, dtype=float, copy=True)
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
        if self.flight_mode == self.MODE_TAKEOFF_WAIT:
            self.update_takeoff_wait()
        elif self.is_flying and self.flight_mode == self.MODE_FOLLOW_ACQUIRE:
            self.send_follow_acquire_target()
        elif self.is_flying and self.flight_mode == self.MODE_FOLLOW_HOLD:
            self.update_follow_hold()
        elif self.is_flying and self.flight_mode == self.MODE_FOLLOW:
            self.send_follow_target()

    def takeoff(self):
        if self.is_flying or self.flight_mode == self.MODE_TAKEOFF_WAIT:
            rospy.logwarn("起飞已在进行或等待完成确认。")
            return
        reasons = self.localization_reasons()
        state = self.supervisor_state()
        if state is None:
            reasons.append("cf17 supervisor.info 缺失或过期")
        elif state["isLocked"] or state["isTumbled"]:
            reasons.append("cf17 Supervisor locked/tumbled")
        elif not state["isArmed"] and not state["canArm"]:
            reasons.append("cf17 canArm=0")
        if reasons:
            rospy.logwarn("拒绝起飞：%s", "; ".join(reasons))
            return
        platform_msg = self.cf231_health.snapshot()
        start_height = platform_msg.pose.position.z + 0.30
        if not TARGET_HEIGHT_MIN_M <= start_height <= TARGET_HEIGHT_MAX_M:
            rospy.logwarn(
                "拒绝起飞：目标高度 %.3f m 超出 [%.2f, %.2f] m",
                start_height,
                TARGET_HEIGHT_MIN_M,
                TARGET_HEIGHT_MAX_M,
            )
            return
        if not state["isArmed"]:
            try:
                arm_response = self.cf17.requestArm(True)
            except Exception as exc:
                rospy.logerr("拒绝起飞：arm 服务失败: %s", exc)
                return
            if (
                not arm_response.success
                or not arm_response.is_armed
                or not arm_response.can_fly
            ):
                rospy.logerr("拒绝起飞：arm 未获 Supervisor 确认: %s", arm_response.message)
                return
        self.is_armed = True
        try:
            self.cf17.takeoff(targetHeight=start_height, duration=TAKEOFF_DURATION_S)
        except Exception as exc:
            rospy.logerr("起飞服务未接受请求：%s", exc)
            self.safe_disarm_if_not_flying("Takeoff 调用失败后的回退")
            return
        self.takeoff_target_height = start_height
        self.takeoff_started_monotonic = time.monotonic()
        self.takeoff_height_stable_samples = 0
        self.takeoff_low_rate_last_log_monotonic = None
        self.flight_mode = self.MODE_TAKEOFF_WAIT
        rospy.loginfo("cf17 起飞请求已被服务接受；进入 TAKEOFF_WAIT，等待轨迹和高度稳定。")

    def update_takeoff_wait(self):
        """Non-blocking takeoff completion state machine, driven by the 5 Hz timer."""
        now = time.monotonic()
        if self.takeoff_started_monotonic is None or self.takeoff_target_height is None:
            self.enter_fault("TAKEOFF_WAIT 缺少起飞状态")
            return
        elapsed = now - self.takeoff_started_monotonic
        if elapsed > TAKEOFF_DURATION_S + TAKEOFF_COMPLETION_TIMEOUT_S:
            self.enter_fault("起飞完成确认超时；未发送新的 goTo")
            return
        # A fresh, structurally valid stream below 7 Hz is not evidence of a
        # broken pose.  It pauses takeoff-completion confirmation within the
        # existing bounded timeout; stale/invalid/jump/latch failures remain
        # immediate faults below.
        health_reasons = self.hard_localization_reasons()
        state = self.supervisor_state()
        if health_reasons:
            self.enter_fault("TAKEOFF_WAIT: " + "; ".join(health_reasons))
            return
        if state is None or state["isLocked"] or state["isTumbled"]:
            self.enter_fault("TAKEOFF_WAIT: Supervisor 状态异常或过期")
            return
        if not (state["isArmed"] and state["canFly"] and state["isFlying"]):
            self.takeoff_height_stable_samples = 0
            return
        self.is_flying = True
        rate_states = self.soft_rate_states()
        if rate_states:
            self.takeoff_height_stable_samples = 0
            last_log = getattr(self, "takeoff_low_rate_last_log_monotonic", None)
            if last_log is None or now - last_log >= 1.0:
                rospy.logwarn(
                    "TAKEOFF_WAIT 暂停完成确认：仅位姿低频（%s）；保持等待，不发送新的 goTo。",
                    "; ".join("%s=%.2f Hz, age=%.3f s" % item for item in rate_states),
                )
                self.takeoff_low_rate_last_log_monotonic = now
            return
        if elapsed < TAKEOFF_DURATION_S + TAKEOFF_SETTLE_MARGIN_S:
            self.takeoff_height_stable_samples = 0
            return
        cf17_msg = self.cf17_health.snapshot()
        height_error = abs(cf17_msg.pose.position.z - self.takeoff_target_height)
        if height_error > TAKEOFF_HEIGHT_TOLERANCE_M:
            self.takeoff_height_stable_samples = 0
            return
        self.takeoff_height_stable_samples += 1
        if self.takeoff_height_stable_samples < TAKEOFF_STABLE_SAMPLES:
            return
        # Completion has now been proved; leave the command-blocking state
        # before calling the normal MANUAL entry gate.
        self.flight_mode = self.MODE_IDLE
        self.start_manual_mode()
        if self.flight_mode == self.MODE_MANUAL:
            rospy.loginfo("起飞完成确认通过；已用最新空中 cf17 位姿建立 MANUAL 基准。")

    def start_tracking(self):
        if self.flight_mode == self.MODE_TAKEOFF_WAIT:
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
        if self.flight_mode == self.MODE_TAKEOFF_WAIT:
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
        if self.flight_mode == self.MODE_TAKEOFF_WAIT:
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
        if not self.is_flying:
            return
        self.flight_mode = self.MODE_LANDING
        platform_msg = self.cf231_health.snapshot()
        target_height = 0.0
        if platform_msg is not None:
            target_height = max(0.0, platform_msg.pose.position.z + 0.05)
        try:
            self.cf17.land(targetHeight=target_height, duration=2.0)
        except Exception as exc:
            self.enter_fault("land 服务调用失败: %s" % exc)
            return
        rospy.loginfo("cf17 降落请求已被服务接受。")
        stopped_state = self.wait_for_supervisor(
            lambda value: not value["isFlying"], LAND_STOPPED_CONFIRM_TIMEOUT_S
        )
        if stopped_state is None:
            self.enter_fault("land 后超时未确认 isFlying=0；未发送 disarm 或 cmd_stop")
            return
        self.is_flying = False
        if not self.safe_disarm_if_not_flying("降落后的 disarm"):
            self.enter_fault("降落后未确认 disarm")
            return
        self.previous_target = None
        self._clear_follow_hold()
        self.flight_mode = self.MODE_IDLE


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
    print("Ctrl+C：未起飞但已 arm 时安全 disarm；飞行时请求正常降落后退出")
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
    """Pure pose/target gate tests; creates no ROS node and calls no services."""
    now = 100.0

    def healthy_monitor(name="test", gap_at=None):
        health = PoseHealth(name)
        times = [now + i * 0.1 for i in range(31)]
        if gap_at is not None:
            times = [t if i < gap_at else t + 0.31 for i, t in enumerate(times)]
        for index, sample_time in enumerate(times):
            msg = make_test_pose(sample_time, x=index * 0.0001)
            assert health.update(msg, sample_time, sample_time)[0]
        return health, times[-1]

    health, end = healthy_monitor()
    assert not health.health_reasons(end)
    assert health.health_reasons(end + POSE_TIMEOUT_S + 0.01)

    for label, msg in (
        ("nan", make_test_pose(now, x=float("nan"))),
        ("zero quaternion", make_test_pose(now, quaternion=(0, 0, 0, 0))),
        ("wrong frame", make_test_pose(now, frame="map")),
    ):
        invalid = PoseHealth(label)
        assert not invalid.update(msg, now, now)[0]
        assert invalid.health_reasons(now)

    low_rate = PoseHealth("low rate")
    for i in range(5):
        stamp = now + i * 0.5
        assert low_rate.update(make_test_pose(stamp), stamp, stamp)[0]
    assert any("频率" in item for item in low_rate.health_reasons(now + 2.0))

    gap_health, gap_end = healthy_monitor("0.31 gap", gap_at=15)
    assert not gap_health.health_reasons(gap_end)

    jumping, jump_end = healthy_monitor("jump")
    jump = make_test_pose(jump_end + 0.1, x=1.0)
    valid, reason = jumping.update(jump, jump_end + 0.1, jump_end + 0.1)
    assert not valid and "飞行中故障锁定" in reason

    # A single ground outlier never contaminates the accepted reference. The
    # next sample near the old reference is accepted normally.
    recovered, recovered_end = healthy_monitor("single outlier")
    outlier_time = recovered_end + 0.1
    assert not recovered.update(
        make_test_pose(outlier_time, x=1.0),
        outlier_time,
        outlier_time,
        allow_ground_resync=True,
    )[0]
    recovery_time = outlier_time + 0.1
    assert recovered.update(
        make_test_pose(recovery_time, x=0.0031),
        recovery_time,
        recovery_time,
        allow_ground_resync=True,
    )[0]
    assert not recovered.resync_candidates

    # Moving a disarmed vehicle to a new stable location establishes a new
    # reference only after a spatially clustered sequence.
    moved, moved_end = healthy_monitor("ground move")
    for index in range(RESYNC_MIN_SAMPLES):
        stamp = moved_end + 0.1 * (index + 1)
        valid, _ = moved.update(
            make_test_pose(stamp, x=0.50 + 0.002 * (index % 2)),
            stamp,
            stamp,
            allow_ground_resync=True,
        )
        assert valid == (index == RESYNC_MIN_SAMPLES - 1)
    assert "地面重同步成功" in moved.consume_event()
    assert abs(moved.snapshot().pose.position.x - 0.502) < 0.003

    # During flight, a coordinate discontinuity remains rejected and cannot
    # automatically resynchronize even if the new stream is internally stable.
    flight_jump, flight_end = healthy_monitor("flight jump")
    for index in range(RESYNC_MIN_SAMPLES + 2):
        stamp = flight_end + 0.1 * (index + 1)
        valid, reason = flight_jump.update(
            make_test_pose(stamp, x=0.60),
            stamp,
            stamp,
            allow_ground_resync=False,
        )
        assert not valid and "飞行中故障锁定" in reason
    assert abs(flight_jump.snapshot().pose.position.x - 0.003) < 0.001

    # The dt-aware bound accepts normal 0.5 m/s motion at both 10 Hz and 3 Hz.
    for sample_period in (0.1, 1.0 / 3.0):
        moving = PoseHealth("motion %.1f Hz" % (1.0 / sample_period))
        for index in range(20):
            stamp = now + index * sample_period
            assert moving.update(
                make_test_pose(stamp, x=0.5 * index * sample_period),
                stamp,
                stamp,
            )[0]

    # Quaternion changes never participate in the position-step comparison.
    rotating = PoseHealth("rotation only")
    for index in range(31):
        stamp = now + 0.1 * index
        yaw = 0.1 * index
        quaternion = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        assert rotating.update(
            make_test_pose(stamp, x=0.0, quaternion=quaternion),
            stamp,
            stamp,
        )[0]

    cf_pose = make_test_pose(now).pose
    assert not target_reasons(np.array([0.1, 0.0, 0.3]), 0.0, cf_pose, None)
    assert target_reasons(np.array([0.1, 0.0, 2.0]), 0.0, cf_pose, None)
    assert target_reasons(np.array([float("nan"), 0.0, 0.3]), 0.0, cf_pose, None)
    assert target_reasons(
        np.array([0.4, 0.0, 0.3]), 0.0, cf_pose, np.array([0.0, 0.0, 0.3])
    )

    # Exercise the real controller gate methods with a fake command sink.
    class FakeCf:
        def __init__(self):
            self.calls = []
            self.goto_calls = []
            self.state = None

        def supervisorState(self, **_kwargs):
            return self.state

        def requestArm(self, arm):
            self.calls.append("arm" if arm else "disarm")
            return types.SimpleNamespace(
                success=False,
                is_armed=False,
                can_fly=False,
                message="fake rejection",
            )

        def takeoff(self, **_kwargs):
            self.calls.append("takeoff")

        def goTo(self, *_args, **_kwargs):
            self.calls.append("goTo")
            self.goto_calls.append((_args, _kwargs))

        def land(self, **_kwargs):
            self.calls.append("land")

        def stop(self):
            self.calls.append("stop")

    class StaticHealth:
        def __init__(self, name, message):
            self.name = name
            self.message = message
            self.fault_latched = False
            self.fault_reason = None

        def health_reasons(self, *_args, **_kwargs):
            return []

        def hard_reasons(self, *_args, **_kwargs):
            return []

        def soft_rate_state(self, *_args, **_kwargs):
            return (10.0, 0.01)

        def snapshot(self):
            return copy.deepcopy(self.message)

        def latch_fault(self, reason):
            self.fault_latched = True
            self.fault_reason = reason

    tracker = TrackerFlight.__new__(TrackerFlight)
    tracker.cf17 = FakeCf()
    tracker.cf17_health = PoseHealth("cf17")
    tracker.cf231_health = PoseHealth("cf231")
    tracker.is_flying = False
    tracker.flight_mode = TrackerFlight.MODE_IDLE
    tracker.previous_target = None
    tracker.takeoff_target_height = None
    tracker.takeoff_started_monotonic = None
    tracker.takeoff_height_stable_samples = 0
    tracker.acquire_anchor = None
    tracker.manual_x = tracker.manual_y = tracker.manual_z = 0.0
    tracker.is_armed = False
    tracker.takeoff()
    tracker.start_tracking()
    tracker.land()
    assert tracker.cf17.calls == []

    # The real callbacks update only their matching health object, and a
    # rotating cf231 cannot invalidate a stationary cf17.
    callback_tracker = TrackerFlight.__new__(TrackerFlight)
    callback_tracker.cf17_health = PoseHealth("cf17")
    callback_tracker.cf231_health = PoseHealth("cf231")
    callback_tracker.is_flying = False
    callback_tracker.is_armed = False
    callback_tracker.flight_mode = TrackerFlight.MODE_IDLE
    callback_tracker.supervisor_state = lambda: {
        "isArmed": False,
        "isFlying": False,
    }
    callback_start = time.time()
    callback_now = [callback_start]
    original_time_now = rospy.Time.now
    rospy.Time.now = staticmethod(
        lambda: rospy.Time.from_sec(callback_now[0])
    )
    try:
        for index in range(31):
            stamp = callback_start + 0.001 * index
            callback_now[0] = stamp
            yaw = 0.1 * index
            callback_tracker.cf17_pose_callback(make_test_pose(stamp, x=0.0))
            callback_tracker.cf231_pose_callback(
                make_test_pose(
                    stamp,
                    x=0.2,
                    quaternion=(
                        0.0,
                        0.0,
                        math.sin(yaw / 2.0),
                        math.cos(yaw / 2.0),
                    ),
                )
            )
    finally:
        rospy.Time.now = original_time_now
    assert callback_tracker.cf17_health.consecutive_valid == 31
    assert callback_tracker.cf231_health.consecutive_valid == 31
    assert callback_tracker.cf17_health.name == "cf17"
    assert callback_tracker.cf231_health.name == "cf231"

    # A flight-time jump enters bounded quarantine; a stable displaced stream
    # cannot become an airborne resynchronization and faults on the next frame.
    callback_tracker.cf17 = FakeCf()
    callback_tracker.is_flying = True
    callback_tracker.is_armed = True
    callback_tracker.previous_target = None
    callback_tracker.follow_hold_started_monotonic = None
    callback_tracker.follow_hold_deadline_monotonic = None
    callback_tracker.follow_hold_recovery_started_monotonic = None
    callback_tracker.follow_hold_reason = None
    callback_tracker.follow_hold_detail = None
    callback_tracker.outlier_context = None
    jump_stamp = callback_start + 0.031
    callback_monotonic = [time.monotonic()]
    original_monotonic = time.monotonic
    rospy.Time.now = staticmethod(
        lambda: rospy.Time.from_sec(callback_now[0])
    )
    time.monotonic = lambda: callback_monotonic[0]
    try:
        callback_now[0] = jump_stamp
        callback_monotonic[0] += 0.001
        callback_tracker.cf17_pose_callback(make_test_pose(jump_stamp, x=0.8))
        assert callback_tracker.flight_mode == TrackerFlight.MODE_FOLLOW_HOLD
        assert not callback_tracker.cf17_health.fault_latched
        for index in range(1, RESYNC_MIN_SAMPLES + 2):
            stamp = jump_stamp + 0.001 * index
            callback_now[0] = stamp
            callback_monotonic[0] += 0.001
            callback_tracker.cf17_pose_callback(
                make_test_pose(stamp, x=0.8)
            )
    finally:
        rospy.Time.now = original_time_now
        time.monotonic = original_monotonic
    assert callback_tracker.cf17.calls == []
    assert callback_tracker.flight_mode == TrackerFlight.MODE_FAULT

    # Once actual Supervisor state is disarmed/on-ground, clustered samples may
    # rebuild the reference, but the flight-time fault latch and FAULT mode
    # remain until explicit operator confirmation.
    callback_tracker.is_flying = False
    callback_tracker.is_armed = False
    ground_start = jump_stamp + 0.02
    rospy.Time.now = staticmethod(
        lambda: rospy.Time.from_sec(callback_now[0])
    )
    time.monotonic = lambda: callback_monotonic[0]
    try:
        for index in range(RESYNC_MIN_SAMPLES):
            stamp = ground_start + 0.1 * index
            callback_now[0] = stamp
            callback_monotonic[0] += 0.1
            callback_tracker.cf17_pose_callback(
                make_test_pose(stamp, x=0.8 + 0.001 * (index % 2))
            )
    finally:
        rospy.Time.now = original_time_now
        time.monotonic = original_monotonic
    resynced_x = callback_tracker.cf17_health.snapshot().pose.position.x
    assert abs(resynced_x - 0.801) < 0.002, (
        resynced_x,
        len(callback_tracker.cf17_health.resync_candidates),
        callback_tracker.cf17_health.last_invalid_reason,
    )
    assert callback_tracker.cf17_health.fault_latched
    assert callback_tracker.flight_mode == TrackerFlight.MODE_FAULT
    assert callback_tracker.cf17.calls == []

    # An arm rejection must prevent Takeoff entirely.
    tracker.cf17_health = StaticHealth("cf17", make_test_pose(now))
    tracker.cf231_health = StaticHealth("cf231", make_test_pose(now))
    tracker.cf17.state = {
        "canArm": True,
        "isArmed": False,
        "canFly": False,
        "isFlying": False,
        "isTumbled": False,
        "isLocked": False,
    }
    tracker.takeoff()
    assert tracker.cf17.calls == ["arm"]
    assert not tracker.is_flying

    # Accepted Takeoff enters the non-blocking completion state; it must not
    # synchronously claim MANUAL/flying completion from the service response.
    class ArmedFakeCf(FakeCf):
        def requestArm(self, arm):
            self.calls.append("arm" if arm else "disarm")
            if arm:
                self.state = dict(self.state, isArmed=True, canFly=True)
                return types.SimpleNamespace(
                    success=True, is_armed=True, can_fly=True, message="confirmed"
                )
            self.state = dict(self.state, isArmed=False, canFly=False)
            return types.SimpleNamespace(
                success=True, is_armed=False, can_fly=False, message="confirmed"
            )

    tracker.cf17 = ArmedFakeCf()
    tracker.cf17.state = {
        "canArm": True,
        "isArmed": False,
        "canFly": False,
        "isFlying": False,
        "isTumbled": False,
        "isLocked": False,
    }
    tracker.is_armed = False
    tracker.takeoff()
    assert tracker.cf17.calls == ["arm", "takeoff"]
    assert not tracker.is_flying
    assert tracker.is_armed
    assert tracker.flight_mode == TrackerFlight.MODE_TAKEOFF_WAIT

    # A FOLLOW target-bound violation must fault before the real goTo call.
    tracker.cf17 = FakeCf()
    tracker.cf17_health = StaticHealth("cf17", make_test_pose(now))
    tracker.cf231_health = StaticHealth("cf231", make_test_pose(now))
    tracker.is_flying = True
    tracker.flight_mode = TrackerFlight.MODE_FOLLOW
    tracker.previous_target = np.array([0.0, 0.0, 0.3])
    tracker.manual_x = 10.0
    assert not tracker.send_follow_target()
    assert tracker.cf17.calls == []

    run_manual_mode_state_machine_tests()
    print("SELF-TEST PASS: pose, frequency, 0.31 s gap, jump and target gates")
    print("SELF-TEST PASS: all abnormal cases produced zero control calls")


def run_manual_mode_state_machine_tests():
    """Offline fake-interface tests for strict FOLLOW/MANUAL separation."""
    class FakeCf:
        def __init__(self):
            self.calls = []
            self.goto_calls = []
            self.state = {
                "canArm": True,
                "isArmed": True,
                "canFly": True,
                "isFlying": True,
                "isTumbled": False,
                "isLocked": False,
            }

        def supervisorState(self, **_kwargs):
            return self.state

        def goTo(self, *args, **kwargs):
            self.calls.append("goTo")
            self.goto_calls.append((args, kwargs))

        def land(self, **_kwargs):
            self.calls.append("land")

        def requestArm(self, arm):
            self.calls.append("arm" if arm else "disarm")
            self.state["isArmed"] = arm
            self.state["canFly"] = arm
            return types.SimpleNamespace(
                success=True, is_armed=arm, can_fly=arm, message="offline"
            )

    class Health:
        def __init__(self, name, message):
            self.name = name
            self.message = message
            self.reasons = []
            self.fault_latched = False
            self.fault_reason = None
            self.rate = 10.0
            self.age = 0.01

        def health_reasons(self, *_args, **_kwargs):
            reasons = self.hard_reasons()
            if not reasons and self.rate < MIN_POSE_RATE_HZ:
                reasons.append("位姿频率 %.2f Hz < %.2f Hz" % (self.rate, MIN_POSE_RATE_HZ))
            return reasons

        def hard_reasons(self, *_args, **_kwargs):
            return list(self.reasons)

        def soft_rate_state(self, *_args, **_kwargs):
            if self.reasons:
                return None
            return (self.rate, self.age)

        def raw_age(self, *_args, **_kwargs):
            return self.age

        def snapshot(self):
            return copy.deepcopy(self.message)

        def latch_fault(self, reason):
            self.fault_latched = True
            self.fault_reason = reason

    def pose_with_yaw(x, y, z, yaw):
        result = make_test_pose(
            100.0,
            x=x,
            quaternion=(
                0.0,
                0.0,
                math.sin(yaw / 2.0),
                math.cos(yaw / 2.0),
            ),
        )
        result.pose.position.y = y
        result.pose.position.z = z
        return result

    def new_manual_tracker(yaw=math.pi / 2, height=0.30):
        tracker = TrackerFlight.__new__(TrackerFlight)
        tracker.cf17 = FakeCf()
        # A sentinel object proves that no path issues a control call to cf231.
        tracker.cf231 = FakeCf()
        tracker.cf17_health = Health("cf17", pose_with_yaw(0.0, 0.0, height, yaw))
        tracker.cf231_health = Health("cf231", pose_with_yaw(0.10, 0.0, 0.0, 0.0))
        tracker.is_flying = True
        tracker.is_armed = True
        tracker.flight_mode = TrackerFlight.MODE_IDLE
        # Model the already successful takeoff target in the offline fake.
        tracker.previous_target = np.array([0.0, 0.0, height], dtype=float)
        tracker.takeoff_target_height = None
        tracker.takeoff_started_monotonic = None
        tracker.takeoff_height_stable_samples = 0
        tracker.acquire_anchor = None
        tracker.follow_hold_started_monotonic = None
        tracker.follow_hold_deadline_monotonic = None
        tracker.follow_hold_recovery_started_monotonic = None
        tracker.follow_hold_reason = None
        tracker.follow_hold_detail = None
        tracker.outlier_context = None
        tracker.platform_motion_context = None
        tracker.platform_motion_observations = deque(maxlen=PLATFORM_MOTION_CONFIRM_SAMPLES)
        tracker.follow_command_speed_mps = FOLLOW_ACQUIRE_MAX_SPEED_MPS
        tracker.follow_command_last_sent_monotonic = time.monotonic()
        tracker.follow_command_lock = threading.Lock()
        tracker.manual_x = tracker.manual_y = tracker.manual_z = 0.0
        tracker.manual_reference_target = None
        tracker.manual_target = None
        tracker.manual_yaw = None
        tracker.xy_step = 0.020
        tracker.z_step = 0.020
        tracker.max_xy_offset = 0.40
        tracker.max_z_offset = 0.30
        tracker.manual_command_period = 0.10
        tracker.last_manual_command_time = 0.0
        return tracker

    tracker = new_manual_tracker()

    # (a) FOLLOW uses cf231 only after a completed bounded acquisition.
    tracker.start_tracking()
    assert tracker.flight_mode == TrackerFlight.MODE_FOLLOW_ACQUIRE
    tracker.track_loop(None)
    assert tracker.flight_mode == TrackerFlight.MODE_FOLLOW
    assert tracker.send_follow_target()
    follow_first = tracker.cf17.goto_calls[-1][0][0].copy()
    tracker.cf231_health.message.pose.position.x += 0.01
    assert tracker.send_follow_target()
    follow_second = tracker.cf17.goto_calls[-1][0][0].copy()
    assert not np.allclose(follow_first, follow_second)

    # (b) m captures cf17 only; cf231 motion cannot alter MANUAL or cause a
    # follow command through the timer.
    tracker.start_manual_mode()
    assert tracker.flight_mode == TrackerFlight.MODE_MANUAL
    manual_reference = tracker.manual_target.copy()
    calls_before = len(tracker.cf17.goto_calls)
    tracker.cf231_health.message.pose.position.x += 0.20
    tracker.track_loop(None)
    assert len(tracker.cf17.goto_calls) == calls_before
    assert np.allclose(tracker.manual_target, manual_reference)

    # (c, d) i is +x in the locked entry-yaw frame: at +90 degrees it advances
    # world +y by exactly 20 mm each time, and never changes cf231's target role.
    tracker.adjust_manual_target(dx=tracker.xy_step)
    first_manual = tracker.cf17.goto_calls[-1][0][0].copy()
    assert np.allclose(first_manual, manual_reference + np.array([0.0, 0.020, 0.0]))
    tracker.last_manual_command_time = 0.0
    tracker.adjust_manual_target(dx=tracker.xy_step)
    second_manual = tracker.cf17.goto_calls[-1][0][0].copy()
    assert np.allclose(second_manual, manual_reference + np.array([0.0, 0.040, 0.0]))

    # (e) The other keys retain their +/- x, +/- y and +/- z symmetry in the
    # same locked frame.
    tracker.last_manual_command_time = 0.0
    tracker.adjust_manual_target(dx=-tracker.xy_step)
    assert np.allclose(tracker.manual_target, manual_reference + np.array([0.0, 0.020, 0.0]))
    tracker.last_manual_command_time = 0.0
    tracker.adjust_manual_target(dy=tracker.xy_step)
    assert np.allclose(tracker.manual_target, manual_reference + np.array([-0.020, 0.020, 0.0]))
    tracker.last_manual_command_time = 0.0
    tracker.adjust_manual_target(dy=-tracker.xy_step)
    tracker.last_manual_command_time = 0.0
    tracker.adjust_manual_target(dz=tracker.z_step)
    assert np.allclose(tracker.manual_target, manual_reference + np.array([0.0, 0.020, 0.020]))
    tracker.last_manual_command_time = 0.0
    tracker.adjust_manual_target(dz=-tracker.z_step)
    assert np.allclose(tracker.manual_target, manual_reference + np.array([0.0, 0.020, 0.0]))

    # yaw=0 and yaw=-90 degrees preserve the expected world-frame rotation.
    for yaw, expected in (
        (0.0, np.array([0.020, 0.0, 0.0])),
        (-math.pi / 2, np.array([0.0, -0.020, 0.0])),
    ):
        yaw_tracker = new_manual_tracker(yaw=yaw)
        yaw_tracker.start_manual_mode()
        yaw_tracker.adjust_manual_target(dx=yaw_tracker.xy_step)
        assert np.allclose(
            yaw_tracker.manual_target,
            yaw_tracker.manual_reference_target + expected,
        )

    # r only changes local state in MANUAL; it deliberately emits no goTo.
    calls_before_reset = len(tracker.cf17.goto_calls)
    tracker.reset_manual_target()
    assert np.allclose(tracker.manual_target, manual_reference)
    assert len(tracker.cf17.goto_calls) == calls_before_reset

    # (f) FOLLOW, IDLE, FAULT, every existing gate and not-flying state block
    # MANUAL goTo.
    for mode in (TrackerFlight.MODE_FOLLOW, TrackerFlight.MODE_IDLE, TrackerFlight.MODE_FAULT):
        calls_before_mode = len(tracker.cf17.goto_calls)
        tracker.flight_mode = mode
        tracker.last_manual_command_time = 0.0
        tracker.adjust_manual_target(dx=tracker.xy_step)
        assert len(tracker.cf17.goto_calls) == calls_before_mode
    for reason, flying in (("fault", True), ("localization", True), ("not flying", False)):
        calls_before_gate = len(tracker.cf17.goto_calls)
        tracker.flight_mode = TrackerFlight.MODE_MANUAL
        tracker.is_flying = flying
        tracker.cf17_health.reasons = [reason] if reason != "not flying" else []
        tracker.last_manual_command_time = 0.0
        tracker.adjust_manual_target(dx=tracker.xy_step)
        assert len(tracker.cf17.goto_calls) == calls_before_gate
        tracker.cf17_health.reasons = []
        tracker.is_flying = True

    # Vertical moves use the same 20 mm step. They are clamped to the existing
    # height bounds, and an outward key at the boundary sends no invalid goTo.
    upper_tracker = new_manual_tracker(height=0.98)
    upper_tracker.start_manual_mode()
    upper_tracker.adjust_manual_target(dz=upper_tracker.z_step)
    assert np.allclose(upper_tracker.manual_target[2], TARGET_HEIGHT_MAX_M)
    upper_calls = len(upper_tracker.cf17.goto_calls)
    upper_tracker.last_manual_command_time = 0.0
    upper_tracker.adjust_manual_target(dz=upper_tracker.z_step)
    assert len(upper_tracker.cf17.goto_calls) == upper_calls
    upper_tracker.last_manual_command_time = 0.0
    upper_tracker.adjust_manual_target(dz=-upper_tracker.z_step)
    assert np.allclose(upper_tracker.manual_target[2], 0.98)
    lower_tracker = new_manual_tracker(height=0.16)
    lower_tracker.start_manual_mode()
    lower_tracker.adjust_manual_target(dz=-lower_tracker.z_step)
    assert np.allclose(lower_tracker.manual_target[2], TARGET_HEIGHT_MIN_M)
    lower_calls = len(lower_tracker.cf17.goto_calls)
    lower_tracker.last_manual_command_time = 0.0
    lower_tracker.adjust_manual_target(dz=-lower_tracker.z_step)
    assert len(lower_tracker.cf17.goto_calls) == lower_calls

    # (g) f alone restores FOLLOW through acquisition; it never installs an
    # unsent MANUAL target as previous_target.
    tracker.cf231_health.message.pose.position.x = 0.12
    tracker.start_tracking()
    assert tracker.flight_mode == TrackerFlight.MODE_FOLLOW_ACQUIRE
    previous_before_acquire = tracker.previous_target.copy()
    calls_before_follow_restore = len(tracker.cf17.goto_calls)
    tracker.track_loop(None)
    assert tracker.flight_mode == TrackerFlight.MODE_FOLLOW
    assert np.allclose(tracker.previous_target, previous_before_acquire)
    assert tracker.send_follow_target()
    assert len(tracker.cf17.goto_calls) == calls_before_follow_restore + 1

    # TAKEOFF_WAIT rejects all mode/movement keys even if Supervisor has
    # already reported thrust/isFlying=1 but the 2 s trajectory is unfinished.
    wait_tracker = new_manual_tracker(height=0.30)
    wait_tracker.is_flying = False
    wait_tracker.flight_mode = TrackerFlight.MODE_TAKEOFF_WAIT
    wait_tracker.takeoff_target_height = 0.70
    wait_tracker.takeoff_started_monotonic = time.monotonic() - 0.10
    wait_tracker.takeoff_height_stable_samples = 0
    wait_tracker.start_tracking()
    wait_tracker.start_manual_mode()
    wait_tracker.adjust_manual_target(dx=wait_tracker.xy_step)
    wait_tracker.reset_manual_target()
    wait_tracker.update_takeoff_wait()
    assert wait_tracker.flight_mode == TrackerFlight.MODE_TAKEOFF_WAIT
    assert wait_tracker.cf17.goto_calls == []

    # Expired duration is insufficient until height has converged for five
    # valid timer samples; only then is the live airborne pose captured.
    wait_tracker.takeoff_started_monotonic = (
        time.monotonic() - TAKEOFF_DURATION_S - TAKEOFF_SETTLE_MARGIN_S - 0.01
    )
    # A fresh cf231 stream at 6.85 Hz pauses completion confirmation instead
    # of latching cf17; recovery to >=7 Hz resumes the unchanged height gate.
    wait_tracker.cf231_health.rate = 6.85
    wait_tracker.update_takeoff_wait()
    assert wait_tracker.flight_mode == TrackerFlight.MODE_TAKEOFF_WAIT
    assert wait_tracker.takeoff_height_stable_samples == 0
    assert wait_tracker.cf17.goto_calls == []
    wait_tracker.cf231_health.rate = 10.0
    wait_tracker.update_takeoff_wait()
    assert wait_tracker.flight_mode == TrackerFlight.MODE_TAKEOFF_WAIT
    wait_tracker.cf17_health.message.pose.position.z = 0.70
    for _ in range(TAKEOFF_STABLE_SAMPLES):
        wait_tracker.update_takeoff_wait()
    assert wait_tracker.flight_mode == TrackerFlight.MODE_MANUAL
    assert np.allclose(wait_tracker.manual_reference_target, [0.0, 0.0, 0.70])
    assert wait_tracker.cf17.goto_calls == []

    timeout_tracker = new_manual_tracker(height=0.30)
    timeout_tracker.is_flying = False
    timeout_tracker.flight_mode = TrackerFlight.MODE_TAKEOFF_WAIT
    timeout_tracker.takeoff_target_height = 0.70
    timeout_tracker.takeoff_started_monotonic = (
        time.monotonic() - TAKEOFF_DURATION_S - TAKEOFF_COMPLETION_TIMEOUT_S - 0.01
    )
    timeout_tracker.takeoff_height_stable_samples = 0
    timeout_tracker.update_takeoff_wait()
    assert timeout_tracker.flight_mode == TrackerFlight.MODE_FAULT
    assert timeout_tracker.manual_target is None
    assert timeout_tracker.cf17.goto_calls == []

    # Reproduce the former 0.765 m MANUAL->FOLLOW mismatch.  Acquisition uses
    # live airborne cf17 as its anchor, ignores a stale prior value only for
    # this gated segment, and sends at most 0.05 m at <= 0.10 m/s.
    acquire_tracker = new_manual_tracker(yaw=0.0, height=0.693)
    acquire_tracker.cf17_health.message.pose.position.x = 0.121
    acquire_tracker.cf17_health.message.pose.position.y = 0.432
    acquire_tracker.cf231_health.message.pose.position.x = 0.347
    acquire_tracker.cf231_health.message.pose.position.y = 0.162
    acquire_tracker.cf231_health.message.pose.position.z = 0.393
    acquire_tracker.start_manual_mode()
    acquire_tracker.previous_target = np.array([0.121, 0.432, -0.012])
    acquire_tracker.start_tracking()
    assert acquire_tracker.flight_mode == TrackerFlight.MODE_FOLLOW_ACQUIRE
    assert acquire_tracker.send_follow_acquire_target()
    acquire_args, acquire_kwargs = acquire_tracker.cf17.goto_calls[-1]
    waypoint = acquire_args[0]
    duration = acquire_kwargs["duration"]
    relative = acquire_kwargs["relative"]
    assert np.linalg.norm(waypoint - acquire_tracker.manual_reference_target) <= FOLLOW_ACQUIRE_MAX_STEP_M + 1e-12
    assert duration >= FOLLOW_ACQUIRE_MAX_STEP_M / FOLLOW_ACQUIRE_MAX_SPEED_MPS
    assert relative is False
    assert np.allclose(acquire_tracker.previous_target, waypoint)
    assert acquire_tracker.flight_mode != TrackerFlight.MODE_FAULT

    # Service failure never advances previous_target.
    failed_previous = acquire_tracker.previous_target.copy()
    acquire_tracker.acquire_anchor = failed_previous.copy()
    def failing_goto(*_args, **_kwargs):
        raise RuntimeError("offline goTo failure")
    acquire_tracker.cf17.goTo = failing_goto
    assert not acquire_tracker.send_follow_acquire_target()
    assert np.allclose(acquire_tracker.previous_target, failed_previous)

    # m interrupts FOLLOW_ACQUIRE without sending a command and re-bases from
    # the latest healthy cf17 pose rather than any acquisition waypoint.
    switch_tracker = new_manual_tracker(yaw=0.0, height=0.50)
    switch_tracker.start_manual_mode()
    switch_tracker.cf231_health.message.pose.position.x = 0.60
    switch_tracker.start_tracking()
    assert switch_tracker.flight_mode == TrackerFlight.MODE_FOLLOW_ACQUIRE
    calls_before_manual_switch = len(switch_tracker.cf17.goto_calls)
    switch_tracker.cf17_health.message.pose.position.x = 0.11
    switch_tracker.start_manual_mode()
    assert switch_tracker.flight_mode == TrackerFlight.MODE_MANUAL
    assert np.allclose(switch_tracker.manual_reference_target, [0.11, 0.0, 0.50])
    assert len(switch_tracker.cf17.goto_calls) == calls_before_manual_switch
    switch_tracker.track_loop(None)
    assert len(switch_tracker.cf17.goto_calls) == calls_before_manual_switch

    # A final target beyond the dedicated acquisition range is a bounded
    # distance hold, not a pose fault, and emits zero commands.
    infeasible_tracker = new_manual_tracker(yaw=0.0, height=0.30)
    infeasible_tracker.start_manual_mode()
    infeasible_tracker.cf231_health.message.pose.position.x = 2.0
    infeasible_tracker.start_tracking()
    assert infeasible_tracker.flight_mode == TrackerFlight.MODE_FOLLOW_HOLD
    assert infeasible_tracker.follow_hold_reason == HOLD_TARGET_DISTANCE
    assert infeasible_tracker.cf17.goto_calls == []
    assert not infeasible_tracker.cf17_health.fault_latched

    # z=1.045 is a temporary final-target geometry issue, not a cf17 pose
    # fault; it holds without clamping or a control call.
    height_hold = new_manual_tracker(yaw=0.0, height=0.50)
    height_hold.cf231_health.message.pose.position.z = 0.745
    height_hold.start_tracking()
    assert height_hold.flight_mode == TrackerFlight.MODE_FOLLOW_HOLD
    assert height_hold.follow_hold_reason == HOLD_TARGET
    assert not height_hold.cf17_health.fault_latched
    assert height_hold.cf17.goto_calls == []

    # True pose/Supervisor failures still enter the existing fault lock.
    fault_tracker = new_manual_tracker(yaw=0.0, height=0.30)
    fault_tracker.start_manual_mode()
    fault_tracker.cf17_health.reasons = ["offline localization fault"]
    fault_tracker.start_tracking()
    assert fault_tracker.flight_mode == TrackerFlight.MODE_FAULT
    assert fault_tracker.cf17.goto_calls == []

    # A fresh/valid 6.56 Hz stream is soft only in FOLLOW: the generic target
    # gate stays non-faulting, while the FOLLOW state machine enters HOLD.
    hold_tracker = new_manual_tracker(yaw=0.0, height=0.50)
    hold_tracker.previous_target = None
    hold_tracker.cf231_health.message.pose.position.x = 0.50
    hold_tracker.flight_mode = TrackerFlight.MODE_FOLLOW
    hold_tracker.cf231_health.rate = 6.56
    clock = [100.0]
    original_monotonic = time.monotonic
    time.monotonic = lambda: clock[0]
    try:
        target, yaw = hold_tracker._follow_target_from_latest_platform()
        assert hold_tracker._target_gate(target, yaw)
        calls_before_hold = len(hold_tracker.cf17.goto_calls)
        assert not hold_tracker.send_follow_target()
        assert hold_tracker.flight_mode == TrackerFlight.MODE_FOLLOW_HOLD
        assert len(hold_tracker.cf17.goto_calls) == calls_before_hold
        assert not hold_tracker.cf17_health.fault_latched

        # HOLD blocks f/m/r and every manual vector without producing a target.
        hold_tracker.start_tracking()
        hold_tracker.start_manual_mode()
        hold_tracker.reset_manual_target()
        for delta in ((hold_tracker.xy_step, 0.0, 0.0), (-hold_tracker.xy_step, 0.0, 0.0),
                      (0.0, hold_tracker.xy_step, 0.0), (0.0, -hold_tracker.xy_step, 0.0),
                      (0.0, 0.0, hold_tracker.z_step), (0.0, 0.0, -hold_tracker.z_step)):
            hold_tracker.adjust_manual_target(*delta)
        assert len(hold_tracker.cf17.goto_calls) == calls_before_hold

        # 7.4 Hz is no longer soft-low, but is below the 7.5 Hz recovery
        # hysteresis; 7.6 then starts recovery, and a 7.4 interruption resets it.
        hold_tracker.cf231_health.rate = 7.4
        clock[0] = 100.2
        hold_tracker.update_follow_hold()
        assert hold_tracker.follow_hold_recovery_started_monotonic is None
        hold_tracker.cf231_health.rate = 7.6
        clock[0] = 100.4
        hold_tracker.update_follow_hold()
        assert hold_tracker.follow_hold_recovery_started_monotonic == 100.4
        hold_tracker.cf231_health.rate = 7.4
        clock[0] = 100.7
        hold_tracker.update_follow_hold()
        assert hold_tracker.follow_hold_recovery_started_monotonic is None
        hold_tracker.cf231_health.rate = 7.6
        clock[0] = 101.0
        hold_tracker.update_follow_hold()
        clock[0] = 101.9
        hold_tracker.update_follow_hold()
        assert hold_tracker.flight_mode == TrackerFlight.MODE_FOLLOW_HOLD
        clock[0] = 102.01
        hold_tracker.update_follow_hold()
        assert hold_tracker.flight_mode == TrackerFlight.MODE_FOLLOW_ACQUIRE
        assert len(hold_tracker.cf17.goto_calls) == calls_before_hold
        acquire_anchor_before_send = hold_tracker.acquire_anchor.copy()
        assert hold_tracker.send_follow_acquire_target()
        first_acquire = hold_tracker.cf17.goto_calls[-1][0][0]
        assert np.linalg.norm(first_acquire - acquire_anchor_before_send) <= FOLLOW_ACQUIRE_MAX_STEP_M + 1e-12

        # A hard pose failure is never reclassified as HOLD.
        hard_tracker = new_manual_tracker(yaw=0.0, height=0.50)
        hard_tracker.flight_mode = TrackerFlight.MODE_FOLLOW
        hard_tracker.cf231_health.reasons = ["offline pose stale"]
        assert not hard_tracker.send_follow_target()
        assert hard_tracker.flight_mode == TrackerFlight.MODE_FAULT

        # HOLD expiry faults with zero further goTo.
        timeout_hold = new_manual_tracker(yaw=0.0, height=0.50)
        timeout_hold.flight_mode = TrackerFlight.MODE_FOLLOW_HOLD
        timeout_hold.follow_hold_started_monotonic = 200.0
        timeout_hold.follow_hold_deadline_monotonic = 200.0 + HOLD_MAX_DURATION_S
        timeout_hold.follow_hold_reason = HOLD_RATE
        timeout_hold.follow_hold_recovery_started_monotonic = None
        clock[0] = 200.0 + HOLD_MAX_DURATION_S + 0.01
        timeout_hold.update_follow_hold()
        assert timeout_hold.flight_mode == TrackerFlight.MODE_FAULT
        assert timeout_hold.cf17.goto_calls == []

        # Repeated/less severe reasons never reset a bounded transaction; an
        # escalation can only retain or shorten the existing deadline.
        deadline_tracker = new_manual_tracker(yaw=0.0, height=0.50)
        clock[0] = 300.0
        deadline_tracker._enter_follow_hold(HOLD_RATE, "offline rate")
        first_deadline = deadline_tracker.follow_hold_deadline_monotonic
        clock[0] = 301.0
        deadline_tracker._enter_follow_hold(HOLD_RATE, "repeat")
        assert deadline_tracker.follow_hold_deadline_monotonic == first_deadline
        deadline_tracker._enter_follow_hold(HOLD_TARGET, "temporary geometry")
        assert deadline_tracker.follow_hold_deadline_monotonic <= first_deadline
        deadline_tracker._enter_follow_hold(HOLD_OUTLIER_CF17, "stricter")
        assert deadline_tracker.follow_hold_deadline_monotonic <= first_deadline
    finally:
        time.monotonic = original_monotonic

    # MANUAL treats a pure low rate as a command rejection (not a new fault).
    low_manual = new_manual_tracker(yaw=0.0, height=0.50)
    low_manual.start_manual_mode()
    low_manual.cf231_health.rate = 6.56
    low_manual.adjust_manual_target(dx=low_manual.xy_step)
    assert not low_manual.send_manual_target()
    assert low_manual.flight_mode == TrackerFlight.MODE_MANUAL
    assert low_manual.cf17.goto_calls == []

    # s remains available in HOLD and invokes the normal land service path.
    land_hold = new_manual_tracker(yaw=0.0, height=0.50)
    land_hold.flight_mode = TrackerFlight.MODE_FOLLOW_HOLD
    land_hold.cf17.state["isFlying"] = False
    land_hold.land()
    assert "land" in land_hold.cf17.calls

    # (h) cf231 is never a command target.
    assert tracker.cf231.calls == []

    # FOLLOW limiter: a distant but geometrically valid final target is split
    # from the last successful target, with speed/acceleration bounded by dt.
    limiter = new_manual_tracker(yaw=0.0, height=0.50)
    limiter.flight_mode = TrackerFlight.MODE_FOLLOW
    limiter.previous_target = np.array([0.0, 0.0, 0.50])
    limiter.follow_command_speed_mps = 0.10
    limiter.follow_command_last_sent_monotonic = 10.0
    command, speed, command_duration = limiter._next_limited_follow_command(
        np.array([0.70, 0.0, 0.30]), 10.2)
    assert np.linalg.norm(command - limiter.previous_target) <= min(
        MAX_TARGET_STEP_M, speed * 0.2) + 1e-12
    assert speed <= FOLLOW_MAX_COMMAND_SPEED_MPS
    assert command_duration >= FOLLOW_COMMAND_MIN_DURATION_S

    # A normal 0.80 m FOLLOW envelope remains, but a valid target within the
    # separate 1.20 m acquisition envelope must be approached conservatively
    # rather than being parked in TARGET_HOLD.  The platform x includes the
    # existing -0.10 m pad offset.
    acquire_0877 = new_manual_tracker(yaw=0.0, height=0.50)
    acquire_0877.cf231_health.message.pose.position.x = 0.977
    acquire_0877.start_tracking()
    assert acquire_0877.flight_mode == TrackerFlight.MODE_FOLLOW_ACQUIRE
    assert acquire_0877.cf17.goto_calls == []
    acquire_anchor = acquire_0877.acquire_anchor.copy()
    assert acquire_0877.send_follow_acquire_target()
    acquire_args, acquire_kwargs = acquire_0877.cf17.goto_calls[-1]
    waypoint = acquire_args[0]
    waypoint_yaw = acquire_kwargs["yaw"]
    waypoint_duration = acquire_kwargs["duration"]
    relative = acquire_kwargs["relative"]
    step = float(np.linalg.norm(waypoint - acquire_anchor))
    assert step <= MAX_TARGET_STEP_M + 1e-12
    assert step / waypoint_duration <= FOLLOW_MAX_COMMAND_SPEED_MPS + 1e-12
    assert waypoint_yaw == 0.0 and relative is False

    acquire_119 = new_manual_tracker(yaw=0.0, height=0.50)
    acquire_119.cf231_health.message.pose.position.x = 1.29
    acquire_119.start_tracking()
    assert acquire_119.flight_mode == TrackerFlight.MODE_FOLLOW_ACQUIRE
    assert acquire_119.cf17.goto_calls == []

    hold_121 = new_manual_tracker(yaw=0.0, height=0.50)
    hold_121.cf231_health.message.pose.position.x = 1.31
    hold_121.start_tracking()
    assert hold_121.flight_mode == TrackerFlight.MODE_FOLLOW_HOLD
    assert hold_121.follow_hold_reason == HOLD_TARGET_DISTANCE
    assert hold_121.cf17.goto_calls == []

    # Distance HOLD recovers only after crossing the 1.10 m hysteresis, and
    # always returns through acquisition rather than directly to FOLLOW.
    hold_121.cf231_health.message.pose.position.x = 1.20
    original_monotonic = time.monotonic
    acquire_clock = [500.0]
    time.monotonic = lambda: acquire_clock[0]
    try:
        hold_121.follow_hold_started_monotonic = 499.0
        hold_121.follow_hold_deadline_monotonic = 510.0
        hold_121.update_follow_hold()
        assert hold_121.follow_hold_recovery_started_monotonic == 500.0
        acquire_clock[0] = 501.01
        hold_121.update_follow_hold()
        assert hold_121.flight_mode == TrackerFlight.MODE_FOLLOW_ACQUIRE
        assert hold_121.cf17.goto_calls == []
    finally:
        time.monotonic = original_monotonic

    # If the platform expands past the acquisition range while approaching,
    # no next waypoint is sent; the only legal outcome is distance HOLD.
    expand_acquire = new_manual_tracker(yaw=0.0, height=0.50)
    expand_acquire.cf231_health.message.pose.position.x = 1.20
    expand_acquire.start_tracking()
    assert expand_acquire.flight_mode == TrackerFlight.MODE_FOLLOW_ACQUIRE
    expand_acquire.cf231_health.message.pose.position.x = 1.35
    assert not expand_acquire.send_follow_acquire_target()
    assert expand_acquire.flight_mode == TrackerFlight.MODE_FOLLOW_HOLD
    assert expand_acquire.follow_hold_reason == HOLD_TARGET_DISTANCE
    assert expand_acquire.cf17.goto_calls == []

    # Intermediate points cannot bypass final height or vertical safety gates.
    height_reject = new_manual_tracker(yaw=0.0, height=0.50)
    height_reject.cf231_health.message.pose.position.z = 0.745  # final z=1.045
    height_reject.start_tracking()
    assert height_reject.flight_mode == TrackerFlight.MODE_FOLLOW_HOLD
    assert height_reject.follow_hold_reason == HOLD_TARGET
    assert height_reject.cf17.goto_calls == []
    vertical_reject = new_manual_tracker(yaw=0.0, height=0.30)
    vertical_reject.cf231_health.message.pose.position.z = 0.70  # final z=1.00, dz=0.70
    vertical_reject.start_tracking()
    assert vertical_reject.flight_mode == TrackerFlight.MODE_FOLLOW_HOLD
    assert vertical_reject.follow_hold_reason == HOLD_TARGET
    assert vertical_reject.cf17.goto_calls == []

    # A failed FOLLOW goTo does not advance either transaction state.
    failed_follow = new_manual_tracker(yaw=0.0, height=0.50)
    failed_follow.flight_mode = TrackerFlight.MODE_FOLLOW
    failed_follow.previous_target = np.array([0.0, 0.0, 0.50])
    failed_follow.follow_command_speed_mps = 0.10
    failed_follow.follow_command_last_sent_monotonic = time.monotonic() - 0.2
    previous_before = failed_follow.previous_target.copy()
    speed_before = failed_follow.follow_command_speed_mps
    failed_follow.cf231_health.message.pose.position.x = 0.50
    failed_follow.cf17.goTo = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline follow failure"))
    assert not failed_follow.send_follow_target()
    assert np.allclose(failed_follow.previous_target, previous_before)
    assert failed_follow.follow_command_speed_mps == speed_before

    # The non-blocking lock prevents a re-entrant Timer callback from issuing
    # a duplicate goTo.
    reentrant = new_manual_tracker(yaw=0.0, height=0.50)
    reentrant.flight_mode = TrackerFlight.MODE_FOLLOW
    reentrant.follow_command_lock.acquire()
    try:
        assert not reentrant.send_follow_target()
        assert reentrant.cf17.goto_calls == []
    finally:
        reentrant.follow_command_lock.release()

    # cf231 0.7 m/s can be confirmed as continuous motion while cf17's command
    # limiter remains independent. A 1.7 m/s first frame only enters HOLD.
    original_monotonic = time.monotonic
    platform_clock = [100.0]
    time.monotonic = lambda: platform_clock[0]
    try:
        def platform_health():
            health = PoseHealth("cf231")
            for index in range(22):
                stamp = 98.0 + 0.1 * index
                assert health.update(make_test_pose(stamp, x=0.0), stamp, stamp)[0]
            return health

        # Normal accepted cf231 movement may accumulate far beyond 0.50 m;
        # that distance is intentionally not a coordinate-migration test.
        long_motion = PoseHealth("cf231")
        for index in range(31):
            stamp = 90.0 + 0.1 * index
            assert long_motion.update(make_test_pose(stamp, x=0.07 * index), stamp, stamp)[0]
        assert long_motion.snapshot().pose.position.x > 0.50
        assert not long_motion.fault_latched

        continuous = new_manual_tracker(yaw=0.0, height=0.50)
        continuous.cf231_health = platform_health()
        continuous.flight_mode = TrackerFlight.MODE_FOLLOW_HOLD
        continuous.follow_hold_reason = HOLD_PLATFORM_MOTION
        continuous.platform_motion_context = {
            "anchor": np.array([0.0, 0.0, 0.0]), "anchor_time": 100.1,
            "prediction_velocity": np.zeros(3), "candidates": deque(), "last_velocity": None,
        }
        for stamp, x in ((100.2, 0.07), (100.3, 0.14), (100.4, 0.21), (100.5, 0.28),
                         (100.6, 0.35)):
            platform_clock[0] = stamp
            result, detail = continuous.cf231_health.update(
                make_test_pose(stamp, x=x), stamp, stamp,
                quarantine_airborne=True,
            )
            assert result is None
            continuous._update_platform_motion_confirmation()
        assert continuous.flight_mode == TrackerFlight.MODE_FOLLOW_ACQUIRE
        assert continuous.cf17.goto_calls == []
        assert continuous.follow_command_speed_mps <= FOLLOW_MAX_COMMAND_SPEED_MPS

        fast_first = new_manual_tracker(yaw=0.0, height=0.50)
        fast_first.flight_mode = TrackerFlight.MODE_FOLLOW
        fast_first.cf231_health = platform_health()
        platform_clock[0] = 100.2
        result, detail = fast_first.cf231_health.update(make_test_pose(100.2, x=0.17), 100.2, 100.2)
        assert result is False
        fast_first._start_platform_motion_confirmation(detail)
        assert fast_first.flight_mode == TrackerFlight.MODE_FOLLOW_HOLD
        assert fast_first.follow_hold_reason == HOLD_PLATFORM_MOTION
        assert fast_first.platform_motion_context["candidates"] == deque()
        assert fast_first.cf17.goto_calls == []
        platform_clock[0] = 101.0
        fast_first.update_follow_hold()
        assert fast_first.flight_mode == TrackerFlight.MODE_FAULT

        migration = new_manual_tracker(yaw=0.0, height=0.50)
        migration.flight_mode = TrackerFlight.MODE_FOLLOW
        migration.cf231_health = platform_health()
        platform_clock[0] = 100.2
        result, detail = migration.cf231_health.update(make_test_pose(100.2, x=0.60), 100.2, 100.2)
        assert result is False
        migration._start_platform_motion_confirmation(detail)
        assert migration.flight_mode == TrackerFlight.MODE_FAULT
        assert migration.cf17.goto_calls == []
    finally:
        time.monotonic = original_monotonic
    assert tracker.cf231.goto_calls == []
    print("OFFLINE STATE-MACHINE PASS: FOLLOW/MANUAL isolation and manual keys")


def main():
    if "--self-test" in sys.argv:
        run_self_tests()
        return
    if "--offline-state-machine-test" in sys.argv:
        run_manual_mode_state_machine_tests()
        return
    import rospkg

    package_path = rospkg.RosPack().get_path("crazyflie_test_ros1")
    tracker = TrackerFlight(package_path + "/config/crazyflies.yaml")
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
        if tracker.is_flying:
            tracker.land()
        elif tracker.is_armed:
            tracker.safe_disarm_if_not_flying("退出清理")


if __name__ == "__main__":
    main()
