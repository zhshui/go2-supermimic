"""Command delay and push handling shared by real sampling and imagination."""
import torch
from dataclasses import dataclass
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


def delayed_action(action, queue, lag):
    history = queue.reshape(-1, 2, 12)
    choices = torch.cat([action[:, None], history], dim=1)
    executed = choices[torch.arange(len(action), device=action.device), lag.long().clamp(0, 2)]
    return executed, choices[:, :2].reshape(-1, 24)


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
    now = env.common_step_counter
    if not hasattr(env, '_next_training_push'):
        env._next_training_push = now + torch.randint(150, 301, (n,), device=dev)
        env._training_push_count = 0
    active = env._next_training_push <= now
    delta = torch.zeros(n, 3, device=dev)
    if active.any():
        ids = active.nonzero().flatten()
        delta[ids, :2] = torch.empty(len(ids), 2, device=dev).uniform_(-0.4, 0.4)
        robot = env.scene['robot']
        velocity = torch.cat([robot.data.root_link_lin_vel_w[ids].clone(),
                              robot.data.root_link_ang_vel_w[ids].clone()], -1)
        velocity[:, :3] += delta[ids]
        robot.write_root_link_velocity_to_sim(velocity, env_ids=ids)
        env._next_training_push[ids] = now + torch.randint(150, 301, (len(ids),), device=dev)
        env._training_push_count += len(ids)
    return delta, active
