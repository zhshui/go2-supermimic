from pathlib import Path

root = Path('/root/go2-supermimic/go2_supermimic')
p = root/'supermimic/state.py'
s = p.read_text().replace('CONTEXT_DIM = 4  # foot friction (1), base COM offset (3)',
    'CONTEXT_DIM = 41  # friction + COM (4), mass ratios (13), actual kp/kd (24)',1)
marker = 'def robot_consts(env) -> RobotConsts:\n'
helper = '''def actual_pd_gains(robot):
  kp = torch.zeros_like(robot.data.joint_pos)
  kd = torch.zeros_like(kp)
  for actuator in robot.actuators:
    ids = actuator.target_ids
    k = getattr(actuator,"stiffness",None)
    d = getattr(actuator,"damping",None)
    kp[:,ids] = k if k is not None else float(actuator.cfg.stiffness)
    kd[:,ids] = d if d is not None else float(actuator.cfg.damping)
  return kp, kd


'''
assert marker in s
s = s.replace(marker,helper+marker,1)
s = s.replace('    kp=torch.tensor(kp, device=dev),\n    kd=torch.tensor(kd, device=dev),',
    '    kp=actual_pd_gains(robot)[0],\n    kd=actual_pd_gains(robot)[1],',1)
s = s.replace('    self.base_ipos0 = self._ipos().clone()', '''    self.base_ipos0 = env.sim.get_default_field("body_ipos")[self.base_body].expand(env.num_envs,3).clone()
    self.robot = robot
    self.body_ids = robot.indexing.body_ids
    assert len(self.body_ids) == 13, "update context mass layout for a different robot"
    self.mass0 = env.sim.get_default_field("body_mass")[self.body_ids].clone()''',1)
s = s.replace('    return torch.cat([mu.unsqueeze(-1), self._ipos() - self.base_ipos0], dim=-1)', '''    mass = self.env.sim.model.body_mass
    masses = mass[:, self.body_ids] if mass.ndim == 2 else mass[self.body_ids].expand(self.env.num_envs,-1)
    kp, kd = actual_pd_gains(self.robot)
    return torch.cat([mu.unsqueeze(-1), self._ipos()-self.base_ipos0, masses/self.mass0.clamp_min(1e-6), kp/100.0, kd/10.0], dim=-1)''',1)
p.write_text(s)

p = root/'supermimic/imagine.py'
s = p.read_text().replace('from go2_supermimic.supermimic.locomotion import stair_tracking',
    'from go2_supermimic.supermimic.locomotion import stair_tracking, locomotion_command, delayed_action',1)
s = s.replace('CRITIC_DIM = st.STATE_DIM + 2 * 7 + 2 + 1 + st.CONTEXT_DIM',
              'CRITIC_DIM = st.STATE_DIM + 2 * 7 + 2 + 1 + st.CONTEXT_DIM + 28',1)
s = s.replace('tau = (c.kp * (q_target - q) - c.kd * s1[:, st.QD])',
              'tau = (k.get("kp",c.kp) * (q_target - q) - k.get("kd",c.kd) * s1[:, st.QD])',1)
s = s.replace('"action_rate_l2": (a - a_prev).square().sum(-1),',
              '"action_rate_l2": (k.get("raw_action",a) - k.get("raw_action_prev",a_prev)).square().sum(-1),',1)
s = s.replace('terms.update(stair_tracking(s1[:, st.V_YAW], s1[:, st.W_B]))',
              'terms.update(stair_tracking(s1[:, st.V_YAW], s1[:, st.W_B], locomotion_command(gst,s1[:, st.V_YAW])))',1)
s = s.replace('return torch.cat(parts + [xi], dim=-1)',
    'return torch.cat(parts + [xi, locomotion_command(gst,s[:,st.V_YAW]), gst["action_queue"], gst["action_delay"].unsqueeze(-1)], dim=-1)',1)
s = s.replace('    a = actor(hist)\n    relief =', '''    raw_action = actor(hist)
    raw_previous = gst["action_queue"][:, :12]
    a, next_queue = delayed_action(raw_action,gst["action_queue"],gst["action_delay"])
    gst = {**gst,"action_queue":next_queue}
    relief =''',1)
s = s.replace('    r, _ = self._rewards(s, s_next, pose_next, gst, a, a_prev, k)',
    '    step_k = {**k,"raw_action":raw_action,"raw_action_prev":raw_previous}\n    r, _ = self._rewards(s, s_next, pose_next, gst, a, a_prev, step_k)',1)
s = s.replace('frame = st.obs_frame(s_next, a, tgt,','frame = st.obs_frame(s_next, raw_action, tgt,',1)
s = s.replace('      "default_q": self.c.default_q[env_ids],',
    '      "kp": self.c.kp[env_ids] if self.c.kp.ndim == 2 else self.c.kp,\n      "kd": self.c.kd[env_ids] if self.c.kd.ndim == 2 else self.c.kd,\n      "default_q": self.c.default_q[env_ids],',1)
p.write_text(s)

p = root/'supermimic/train.py'
s = p.read_text().replace('      s = st.robot_state(env, field=self.field)', '''      s = st.robot_state(env, field=self.field)
      if self.term.cfg.velocity_commands:
        from go2_supermimic.env.locomotion_dynamics import random_push_before_state
        from go2_supermimic.target.interface import yaw_rotate_inverse
        delta, pushed = random_push_before_state(env)
        s[:, st.V_YAW] += yaw_rotate_inverse(delta, env.scene["robot"].data.heading_w)
        if self.buf.size and pushed.any():
          self.buf.done[pushed, (self.buf.ptr-1)%self.buf.L] = 1.0
        self.ep_stats["Metrics/locomotion/push_events"].append(float(pushed.sum()))''',1)
s = s.replace('      self.buf.add(s, pose, a, self.a_prev, self.obs, gen, done, relief)',
    '      executed = env.action_manager.get_term("joint_pos").last_executed_action.clone() if self.term.cfg.velocity_commands else a\n      self.buf.add(s, pose, executed, self.a_prev, self.obs, gen, done, relief)',1)
s = s.replace('s[keep], a[keep], self.a_prev[keep],','s[keep], executed[keep], self.a_prev[keep],',1)
s = s.replace('self.a_prev = torch.where(done.unsqueeze(-1), torch.zeros_like(a), a)',
              'self.a_prev = torch.where(done.unsqueeze(-1), torch.zeros_like(executed), executed)',1)
s = s.replace('      self.env_steps += env.num_envs', '''      if self.term.cfg.velocity_commands:
        d = env.scene["robot"].data
        from go2_supermimic.target.interface import yaw_rotate_inverse
        v = yaw_rotate_inverse(d.root_link_lin_vel_w,d.heading_w)
        cmd = self.term.gen.velocity_command
        self.ep_stats["Metrics/locomotion/yaw_error"].append(float((d.root_link_ang_vel_b[:,2]-cmd[:,2]).abs().mean()))
        self.ep_stats["Metrics/locomotion/velocity_error"].append(float((v[:,:2]-cmd[:,:2]).norm(dim=-1).mean()))
      self.env_steps += env.num_envs''',1)
s = s.replace('    else:\n      self.policy_start = max(self.policy_start, self.cfg.wm_rewarmup_iterations)',
    '    elif expand_world_model_context(ckpt["world_model"], self.wm):\n      print("world model: physical context expanded with zero-weight new inputs",flush=True)\n    else:\n      self.policy_start = max(self.policy_start, self.cfg.wm_rewarmup_iterations)',1)
marker = '    critic_sd = self.critic.state_dict()\n'
addition = '''    if self.term.cfg.velocity_commands and not ckpt.get("velocity_commands",False):
      offset = self.history * sum(dim for name,dim in self.terms[:5])
      indices = torch.tensor([offset+t*28+j for t in range(self.history) for j in range(3)],device=self.dev)
      with torch.no_grad():
        self.actor.net[0].weight[:,indices] = 0
        self.actor.norm.mean[indices] = 0
        self.actor.norm.var[indices] = 1
      print("actor: velocity-command columns enabled with zero initial weights",flush=True)
'''
assert marker in s
s = s.replace(marker,addition+marker,1)
s = s.replace('        "supermimic": True,',
              '        "supermimic": True,\n        "velocity_commands": bool(self.term.cfg.velocity_commands),',1)
marker = 'def main() -> None:\n'
helper = '''def expand_world_model_context(old, model):
  new = model.state_dict()
  key = "net.0.weight"
  if key not in old or old[key].shape[0] != new[key].shape[0]:
    return False
  prefix = st.STATE_DIM + 2*st.NUM_JOINTS + 4
  new_prefix = st.STATE_DIM + 2*st.NUM_JOINTS + st.CONTEXT_DIM
  if old[key].shape[1] != prefix + tr.SCAN_POINTS or new[key].shape[1] != new_prefix + tr.SCAN_POINTS:
    return False
  for name, value in old.items():
    if name == key:
      new[name].zero_(); new[name][:,:prefix] = value[:,:prefix]; new[name][:,new_prefix:] = value[:,prefix:]
    elif name in ("in_norm.mean","in_norm.var"):
      new[name][:prefix] = value[:prefix]; new[name][new_prefix:] = value[prefix:]
    elif name not in new or new[name].shape != value.shape:
      return False
    else:
      new[name] = value.clone()
  model.load_state_dict(new)
  return True


'''
assert marker in s
s = s.replace(marker,helper+marker,1)
p.write_text(s)
