# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from recording import (
    DepthUint16Wrapper, isolated_legacy_numpy_rng, as_numpy, success_value,
    annotate_depth_datasets, sensor_metrics,
)
from robofactory.utils.wrappers.record import RecordEpisodeMA
from quality_experts import EXPERTS, Expert, QualityFailure, raw_pose, audit_carry_trace, audit_meat_carry, require_bounded_command
from action_contract import source_legacy_numpy_seed, file_sha256

TASKS = {"PlaceFood": ("PlaceFood-rf", "place_food.yaml"),
         "TakePhoto": ("TakePhoto-rf", "take_photo.yaml"),
         "CameraAlignment": ("CameraAlignment-rf", "camera_alignment.yaml")}


class QualityTrace(gym.Wrapper):
    def __init__(self, env, writer, max_steps):
        super().__init__(env)
        self.writer, self.max_steps = writer, max_steps
        self.phase, self.steps = 'reset', 0
        self.states, self.commands, self.events, self.checks = [], [], [], []
        self.ever_success = False

    def set_phase(self, name):
        self.phase = name
        self.events.append({'phase': name, 'start_step': self.steps})

    def observe(self, obs, info):
        agents = self.unwrapped.agent.agents
        # Robot proprioception only. Object truth stays out of training observations.
        obs['extra']['agent_tcp_pose'] = torch.stack([a.tcp.pose.raw_pose for a in agents], dim=1)
        objects = {k: getattr(self.unwrapped, k) for k in ['meat','pot','cubeA','cubeB','goal_region','camera'] if hasattr(self.unwrapped,k)}
        state = {'step': self.steps, 'phase': self.phase,
                 'tcp': as_numpy(obs['extra']['agent_tcp_pose']).tolist(),
                 'qpos': [as_numpy(a.robot.get_qpos()).tolist() for a in agents],
                 'objects': {k:raw_pose(a).tolist() for k,a in objects.items()},
                 'grasps': {k:[bool(agent.is_grasping(a).item()) for agent in agents]
                            for k,a in objects.items() if k!='goal_region' and hasattr(a,'_bodies')},
                 'success': success_value(info)}
        self.states.append(state)
        self.ever_success |= state['success']
        if self.writer:
            keys=['head_camera_global']+[f'wrist_camera_agent{i}' for i in range(len(agents))]
            images=[]
            for i,k in enumerate(keys):
                frame=Image.fromarray(as_numpy(obs['sensor_data'][k]['rgb']))
                label='Global' if i==0 else f'Agent {i} Wrist'
                ImageDraw.Draw(frame).text((3,3),f'{label} | {self.steps} {self.phase}',fill='white',stroke_width=1,stroke_fill='black')
                images.append(np.asarray(frame))
            self.writer.append_data(np.concatenate(images,axis=1))
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observe(obs, info), info

    def step(self, action):
        if self.steps >= self.max_steps:
            raise QualityFailure(f"official step budget {self.max_steps} exhausted")
        for i, agent in enumerate(self.unwrapped.agent.agents):
            require_bounded_command(as_numpy(action[f'panda-{i}']), as_numpy(agent.robot.get_qlimits())[:7])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        self.commands.append({k:as_numpy(v).tolist() for k,v in action.items()})
        return self.observe(obs,info), reward, terminated, truncated, info


def validate_calibration(h5_path):
    from scipy.spatial.transform import Rotation
    result={}
    with h5py.File(h5_path,'r') as f:
        g=f['traj_0']
        tcp=g['obs/extra/agent_tcp_pose'][:]
        for i in range(tcp.shape[1]):
            poses=tcp[:,i]
            world=np.tile(np.eye(4),(len(poses),1,1))
            world[:,:3,3]=poses[:,:3]
            world[:,:3,:3]=Rotation.from_quat(poses[:,[4,5,6,3]]).as_matrix()
            cams=g[f'obs/sensor_param/wrist_camera_agent{i}/cam2world_gl'][:]
            transforms=np.linalg.inv(world)@cams
            median=transforms[0]
            error=np.max(np.abs(transforms-median))
            if error>1e-4:
                raise QualityFailure(f"nonrigid TCP-camera calibration for agent {i}: {error}")
            result[f'wrist_camera_agent{i}']={'tcp_to_camera_gl':median.tolist(),'max_matrix_residual':float(error)}
    return result


def replay_camera_task(env, trace, source):
    """Preserve the official source commands, except explicit physical limits."""
    bounded=0
    limits=[as_numpy(a.robot.get_qlimits())[:7] for a in env.unwrapped.agent.agents]
    trace.set_phase('source_replay')
    with h5py.File(source,'r') as handle:
        group=handle['traj_0/actions']
        for step in range(len(group['panda-0'])):
            action={}
            for i,limit in enumerate(limits):
                name=f'panda-{i}'
                command=group[name][step].astype(np.float32)
                target=np.clip(command[:7],limit[:,0],limit[:,1])
                bounded += int(np.any(command[:7]!=target))
                command[:7]=target
                action[name]=command
            env.step(action)
    return bounded


def write_review_sheet(h5_path, output, states, events, task):
    """Representative temporal overview; full all-view MP4 is retained."""
    last=len(states)-1
    selected=set(np.linspace(0,last,9,dtype=int).tolist())
    selected.update(min(last,e['start_step']+1) for e in events if e['phase'] in
        {'descend','descend_aligned','settle_grasp','lift_clear','carry_over_pot','lower_inside_pot','place_top'})
    # Contact transitions are especially important for source-replayed tasks.
    for i in range(1,len(states)):
        if states[i]['grasps']!=states[i-1]['grasps']:
            selected.update([max(0,i-1),i,min(last,i+3)])
    selected=sorted(selected)
    if len(selected)>24:
        selected=[selected[i] for i in np.linspace(0,len(selected)-1,24,dtype=int)]
    with h5py.File(h5_path,'r') as handle:
        keys=['head_camera_global']+[f'wrist_camera_agent{i}' for i in range(len(states[0]['tcp']))]
        for page,start in enumerate(range(0,len(selected),8)):
            indices=selected[start:start+8]
            sheet=Image.new('RGB',(320*len(keys),260*len(indices)),color='white')
            draw=ImageDraw.Draw(sheet)
            for row,index in enumerate(indices):
                draw.text((4,row*260),f'{task} | frame {index} | {states[index]["phase"]} | success={states[index]["success"]}',fill='black')
                for col,key in enumerate(keys):
                    sheet.paste(Image.fromarray(handle[f'traj_0/obs/sensor_data/{key}/rgb'][index]),(col*320,row*260+20))
            sheet.save(output/f'review_{page+1:02d}.jpg',quality=92)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',choices=TASKS,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--episode',type=int)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--food-clearance',type=float,default=.08)
    p.add_argument('--no-video',action='store_true')
    p.add_argument('--min-free-gb',type=float,default=180)
    p.add_argument('--source-h5',type=Path)
    p.add_argument('--legacy-numpy-seed',type=int)
    p.add_argument('--config',type=Path)
    p.add_argument('--takephoto-camera',choices=['original','side_x'],default='side_x')
    args=p.parse_args()
    if not .02<=args.food_clearance<=.15:
        p.error('food-clearance must be in [0.02,0.15] m')
    args.output_root.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.output_root).free/2**30 < args.min_free_gb:
        raise RuntimeError('disk guard triggered before recording')
    ep=args.seed if args.episode is None else args.episode
    name=f'episode_{ep:06d}_seed_{args.seed:06d}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
    output=args.output_root/args.task/name
    output.mkdir(parents=True,exist_ok=False)
    import robofactory
    config=args.config or Path(robofactory.__file__).resolve().parent/'configs/worlddreamer'/TASKS[args.task][1]
    payload=yaml.safe_load(config.read_text())
    if args.task=='TakePhoto' and args.takephoto_camera=='side_x':
        sensor=next(s for s in payload['cameras']['sensor'] if s['uid']=='wrist_camera_agent2')
        sensor['pose']={'type':'look_at','params':[[.09,0.,-.045],[0.,0.,.02],[1.,0.,0.]]}
    snapshot=output/'env_config.yaml'
    snapshot.write_text(yaml.safe_dump(payload,sort_keys=False))
    config=snapshot
    legacy_seed=args.seed
    if args.task in ('TakePhoto','CameraAlignment'):
        if args.source_h5 is None:
            p.error('camera tasks require a recorded source H5')
        legacy_seed=(args.legacy_numpy_seed if args.legacy_numpy_seed is not None
                     else source_legacy_numpy_seed(args.source_h5,args.seed))
    writer=None if args.no_video else imageio.get_writer(output/'global_wrist_hstack.mp4',fps=20,codec='libx264',quality=8,macro_block_size=1)
    env=trace=None
    report={'schema_version':2,'task':args.task,'episode':ep,'seed':args.seed,
            'expert_version':'staged_geometry_v3_keep_grasp_orientation' if args.task in EXPERTS else 'bounded_source_replay_v3',
            'status':'running','accepted':False,'visual_review':'pending',
            'config':str(config),'config_sha256':file_sha256(config),'food_clearance_m':args.food_clearance,'legacy_numpy_seed':legacy_seed,
            'source_h5':str(args.source_h5) if args.source_h5 else None,
            'generator_code_sha256':{name:file_sha256(Path(__file__).with_name(name))
                                     for name in ['quality_rollout.py','quality_experts.py']},
            'camera_variant':args.takephoto_camera if args.task=='TakePhoto' else 'unchanged',
            'rng_policy':'environment construction/reset NumPy scope restored; no global Torch/Python seed changes'}
    (output/'validation.json').write_text(json.dumps(report,indent=2))
    try:
        with isolated_legacy_numpy_rng(legacy_seed):
            core=gym.make(TASKS[args.task][0],config=str(config),obs_mode='rgb+depth',control_mode='pd_joint_pos',
                          render_mode='sensors',sim_backend='cpu',sensor_configs={'shader_pack':'default'})
            trace=QualityTrace(DepthUint16Wrapper(core),writer,max_steps=500)
            env=RecordEpisodeMA(trace,output_dir=str(output),trajectory_name=f'{args.task}_wrist_rgbd',
                                save_trajectory=True,save_video=False,record_observation=True,
                                record_env_state=False,clean_on_close=False)
            env.reset(seed=args.seed)
        if args.task in EXPERTS:
            expert=Expert(env,trace,args.food_clearance)
            EXPERTS[args.task](expert)
        else:
            report['bounded_agent_steps']=replay_camera_task(env,trace,args.source_h5)
        report.update(status='complete',success=trace.states[-1]['success'],ever_success=trace.ever_success)
        if not report['success']:
            raise QualityFailure('expert did not finish with official success')
        report['carry_quality']=(audit_carry_trace(args.task,trace.states) if args.task in EXPERTS
                                 else audit_meat_carry(trace.states))
        report['accepted']=True
    except Exception as exc:
        report.update(status='rejected' if isinstance(exc,QualityFailure) else 'error',
                      error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    finally:
        if trace:
            report.update(raw_steps=trace.steps,phases=trace.events,quality_checks=trace.checks,
                          success=trace.states[-1]['success'] if trace.states else False,ever_success=trace.ever_success,
                          agent_count=len(trace.unwrapped.agent.agents))
            (output/'telemetry.json').write_text(json.dumps({'states':trace.states,'commands':trace.commands},separators=(',',':')))
        if env:
            env.flush_trajectory(verbose=False)
            env.close()
        if writer:
            writer.close()
    files=list(output.glob('*.h5'))
    if report['accepted']:
        try:
            assert len(files)==1
            cameras=tuple(['head_camera_global']+[f'wrist_camera_agent{i}' for i in range(report['agent_count'])])
            annotate_depth_datasets(files[0],cameras)
            report['calibration']=validate_calibration(files[0])
            # Validate array lengths and actual RGB-D/parameter presence without duplicating RGB video files.
            with h5py.File(files[0],'r') as f:
                g=f['traj_0']
                for c in cameras:
                    for field in [f'obs/sensor_data/{c}/rgb',f'obs/sensor_data/{c}/depth',
                                  f'obs/sensor_param/{c}/intrinsic_cv',f'obs/sensor_param/{c}/extrinsic_cv',f'obs/sensor_param/{c}/cam2world_gl']:
                        assert len(g[field])==trace.steps+1,field
                assert len(g['obs/extra/agent_tcp_pose'])==trace.steps+1
            report['h5']=str(files[0])
            motion,depth,parameters=sensor_metrics(files[0],cameras)
            if any(d['valid_pixel_ratio']<=0 for d in depth.values()):
                raise QualityFailure('empty depth observation')
            report['sensors']={'motion':motion,'depth':depth,'parameters':parameters}
            write_review_sheet(files[0],output,trace.states,trace.events,args.task)
        except Exception as exc:
            report.update(status='error',accepted=False,error=f'postflight: {exc}',traceback=traceback.format_exc())
    report['automated_pass']=report['accepted']
    report['accepted']=False  # Publication requires the separate visual-review decision.
    (output/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'output':str(output),'validation':report}),flush=True)
    return 0 if report['automated_pass'] else 2


if __name__=='__main__':
    sys.exit(main())
