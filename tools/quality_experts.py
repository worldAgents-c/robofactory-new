# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import numpy as np


class QualityFailure(RuntimeError):
    pass


def require_bounded_command(command, limits, tolerance=1e-6):
    """Validate arm joint targets and the gripper command against actuator limits."""
    command, limits = np.asarray(command), np.asarray(limits)
    if command.shape != (8,) or limits.shape != (7, 2) or not np.isfinite(command).all():
        raise QualityFailure('invalid actuator command shape or nonfinite values')
    if np.any(command[:7] < limits[:, 0] - tolerance) or np.any(command[:7] > limits[:, 1] + tolerance):
        raise QualityFailure('expert arm target exceeds physical joint limits')
    if not -1 - tolerance <= command[-1] <= 1 + tolerance:
        raise QualityFailure('expert gripper drive exceeds [-1, 1]')


def raw_pose(actor):
    return actor.pose.raw_pose.detach().cpu().numpy().reshape(7).copy()


def pose_change(initial, final):
    translation = np.linalg.norm(final[:3] - initial[:3])
    q0, q1 = initial[3:] / np.linalg.norm(initial[3:]), final[3:] / np.linalg.norm(final[3:])
    angle = 2 * np.arccos(np.clip(abs(q0 @ q1), 0, 1))
    return float(translation), float(np.degrees(angle))


def audit_carry_trace(task, states):
    """Reject sustained loss of a required grasp before intentional placement.

    Contact flags can flicker for one simulation tick. Three consecutive missing
    contacts (150 ms at 20 Hz) fail this teacher-only gate; all gaps are reported.
    """
    required = {
        'PlaceFood': [('meat', 0, {'lift_clear', 'orient_at_clearance', 'carry_over_pot'})],
        'TwoRobotsStackCube': [
            ('cubeA', 0, {'lift_cubes', 'carry_bottom'}),
            ('cubeB', 1, {'lift_cubes', 'carry_bottom', 'place_bottom', 'retreat_bottom', 'carry_top'}),
        ],
    }[task]
    report = []
    for name, agent, phases in required:
        gaps, run, longest, checked = [], 0, 0, 0
        for state in states:
            if state['phase'] not in phases:
                run = 0
                continue
            checked += 1
            if not state['grasps'][name][agent]:
                gaps.append(state['step'])
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        report.append({'object': name, 'agent': agent, 'checked_steps': checked,
                       'ungrasped_steps': gaps, 'longest_ungrasped_run': longest})
        if not checked or longest >= 3:
            raise QualityFailure(f'carry grasp quality failed for {name}, Agent {agent+1}: {longest} steps')
    return report


def audit_meat_carry(states):
    initial_z=states[0]['objects']['meat'][2]
    lifted=[i for i,s in enumerate(states) if s['objects']['meat'][2]>initial_z+.10]
    if not lifted:
        raise QualityFailure('meat was never lifted by 10 cm')
    run=longest=0
    for s in states[lifted[0]:]:
        run=0 if s['grasps']['meat'][2] else run+1
        longest=max(longest,run)
    if longest>=3 or not states[-1]['grasps']['meat'][2]:
        raise QualityFailure(f'meat grip lost after lift: longest missing run {longest}')
    return {'first_lift_step':lifted[0],'longest_ungrasped_run':longest,'final_grasp':True}


class Expert:
    def __init__(self, env, trace, food_clearance=0.08):
        from robofactory.planner.motionplanner import PandaArmMotionPlanningSolver
        self.env, self.core, self.trace = env, env.unwrapped, trace
        self.food_clearance = food_clearance
        self.planner = PandaArmMotionPlanningSolver(
            env, debug=False, vis=False,
            base_pose=[a.robot.pose for a in self.core.agent.agents],
            visualize_target_grasp_pose=False, print_env_info=False, is_multi_agent=True,
        )

    def move(self, phase, poses, ids, refine=4):
        self.trace.set_phase(phase)
        result = self.planner.move_to_pose_with_screw(poses, move_id=ids, refine_steps=refine)
        if isinstance(result, (int, np.integer)) and result == -1:
            raise QualityFailure(f"planning failed in {phase}")
        return result

    def hold(self, phase, steps=4):
        self.trace.set_phase(phase)
        for _ in range(steps):
            action = {f"panda-{i}": np.r_[a.robot.get_qpos()[0, :7].cpu().numpy(), self.planner.gripper_state[i]]
                      for i, a in enumerate(self.core.agent.agents)}
            self.env.step(action)

    def close(self, ids):
        self.trace.set_phase("close_gripper")
        self.planner.close_gripper(ids)
        self.hold("settle_grasp")

    def open(self, ids):
        self.trace.set_phase("release")
        self.planner.open_gripper(ids)

    def require_grasp(self, actor, agent):
        if not bool(self.core.agent.agents[agent].is_grasping(actor).item()):
            raise QualityFailure(f"Agent {agent+1} did not hold {actor.name} in {self.trace.phase}")

    def require_stationary(self, actor, initial):
        shift, rotation = pose_change(initial, raw_pose(actor))
        self.trace.checks.append({"object": actor.name, "phase": self.trace.phase,
                                  "pregrasp_shift_m": shift, "pregrasp_rotation_deg": rotation})
        if shift > 0.005 or rotation > 5:
            raise QualityFailure(f"{actor.name} moved before closure: {shift:.4f} m, {rotation:.2f} deg")


def place_food(expert: Expert):
    e, p = expert.core, expert.planner
    lid = p.get_grasp_pose_w_labeled_direction(e.pot, e.annotation_data['pot'], pre_dis=0, id=3)
    food = p.get_grasp_pose_w_labeled_direction(e.meat, e.annotation_data['meat'], pre_dis=0, id=0)
    lid_above, food_above = lid.copy(), food.copy()
    lid_above[2] += .2
    food_above[2] += .15
    expert.move("pregrasp", [food_above, lid_above], [0, 1])
    lid[2] -= .02  # Lower the grasp target to engage the lid handle.
    expert.move("descend", [food, lid], [0, 1])
    expert.close([0, 1])
    expert.require_grasp(e.meat, 0)
    lid[2] += .3
    food[2] += .22
    expert.move("lift_clear", [food, lid], [0, 1])
    expert.require_grasp(e.meat, 0)

    drop = p.get_target_pose_w_labeled_direction(e.pot, e.annotation_data['pot'], pre_dis=0)
    # Maintain the grasp orientation while placing the food.
    drop[3:] = food[3:]
    over = drop.copy()
    over[2] = max(food[2], drop[2] + expert.food_clearance)
    # Keep the grasp orientation at clearance, then translate over the rim.
    rotate = over.copy()
    rotate[:2] = food[:2]
    expert.move("orient_at_clearance", rotate, [0])
    expert.require_grasp(e.meat, 0)
    expert.move("carry_over_pot", over, [0])
    expert.require_grasp(e.meat, 0)
    expert.move("lower_inside_pot", drop, [0])
    expert.open([0])
    expert.hold("settle_release", 12)


def stack_cubes(expert: Expert):
    e, p = expert.core, expert.planner
    actors = [e.cubeA, e.cubeB]
    initial = [raw_pose(a) for a in actors]
    grasp = [p.get_grasp_pose_from_obb(a, i) for i, a in enumerate(actors)]
    above = [v.copy() for v in grasp]
    for pose in above:
        pose[2] += .12
    expert.move("align_above_cubes", above, [0, 1])
    for a, before in zip(actors, initial):
        expert.require_stationary(a, before)
    # Approach the grasp plane directly.
    expert.move("descend_aligned", grasp, [0, 1], refine=6)
    for a, before in zip(actors, initial):
        expert.require_stationary(a, before)
    expert.close([0, 1])
    for i, a in enumerate(actors):
        expert.require_grasp(a, i)
    lifted = [v.copy() for v in grasp]
    for pose in lifted:
        pose[2] += .2
    expert.move("lift_cubes", lifted, [0, 1])
    for i, a in enumerate(actors):
        expert.require_grasp(a, i)
    bottom = p.get_grasp_pose_for_stack(grasp[0], e.goal_region)
    bottom_above = bottom.copy()
    bottom_above[2] = lifted[0][2]
    expert.move("carry_bottom", bottom_above, [0])
    expert.move("place_bottom", bottom, [0])
    expert.open([0])
    expert.move("retreat_bottom", lifted[0], [0])
    top = p.get_grasp_pose_for_stack(grasp[1], e.cubeA)
    top_above = top.copy()
    top_above[2] = lifted[1][2]
    expert.move("carry_top", top_above, [1])
    expert.require_grasp(e.cubeB, 1)
    expert.move("place_top", top, [1])
    expert.open([1])
    expert.hold("settle_stack", 8)


EXPERTS = {"PlaceFood": place_food}
