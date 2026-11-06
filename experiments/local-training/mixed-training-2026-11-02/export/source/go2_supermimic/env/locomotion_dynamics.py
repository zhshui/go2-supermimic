"""Command delay and push handling shared by real sampling and imagination."""
import torch
from dataclasses import dataclass
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


from go2_supermimic.supermimic.locomotion import delayed_action


class DelayedJointPositionAction(JointPositionAction):
    def process_actions(self, actions):
        generator = self._env.command_manager.get_term('target').gen
        executed, queue = delayed_action(actions, generator.action_queue, generator.action_delay)
        generator.action_queue = queue
        self.last_executed_action = executed.clone()
        super().process_actions(executed)


@dataclass(kw_only=True)
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env):
        return DelayedJointPositionAction(self, env)


def random_push_before_state(env):
    """Velocity impulses; caller masks the previous replay transition at each impulse."""
    n, dev = env.num_envs, env.device
    cfg = env.command_manager.get_term("target").cfg
    if not cfg.pushes_enabled:
        return torch.zeros(n,3,device=dev), torch.zeros(n,dtype=torch.bool,device=dev)
    lower, upper = [max(1,round(x/env.step_dt)) for x in cfg.push_interval_range]
    now = env.common_step_counter
    if not hasattr(env, '_next_training_push'):
        env._next_training_push = now + torch.randint(lower, upper+1, (n,), device=dev)
        env._training_push_count = 0
    active = env._next_training_push <= now
    delta = torch.zeros(n, 3, device=dev)
    if active.any():
        ids = active.nonzero().flatten()
        delta[ids, :2] = torch.empty(len(ids), 2, device=dev).uniform_(*cfg.push_velocity_range)
        robot = env.scene['robot']
        velocity = torch.cat([robot.data.root_link_lin_vel_w[ids].clone(),
                              robot.data.root_link_ang_vel_w[ids].clone()], -1)
        velocity[:, :3] += delta[ids]
        robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
        env._next_training_push[ids] = now + torch.randint(lower, upper+1, (len(ids),), device=dev)
        env._training_push_count += len(ids)
    return delta, active
