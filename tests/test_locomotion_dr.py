import torch
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.locomotion import stair_tracking, delayed_action
from go2_supermimic.supermimic.buffer import pack_generator, unpack_generator
from go2_supermimic.supermimic.train import expand_world_model_context
from go2_supermimic.supermimic.world_model import WorldModel, mlp
from go2_supermimic.target.generator import BallTarget, BallTargetCfg
from go2_supermimic.target.interface import oracle_observation
from go2_supermimic.target.terrain import SCAN_POINTS


def test_nonzero_yaw_command_has_correct_optimum_and_gradient():
    v = torch.zeros(3, 3)
    w = torch.zeros(3, 3, requires_grad=True)
    cmd = torch.tensor([[0.,0.,-.5],[0.,0.,0.],[0.,0.,.5]])
    r = stair_tracking(v,w,cmd)['track_ang_vel']
    r.sum().backward()
    assert w.grad[0,2] < 0 and w.grad[2,2] > 0
    assert r[1] > r[0]
    assert torch.allclose(stair_tracking(v,cmd,cmd)['track_ang_vel'],torch.ones(3))


def test_delayed_actions_match_queue_and_preserve_gradients():
    a = torch.ones(3,12,requires_grad=True)
    q = torch.cat([torch.full((3,12),2.),torch.full((3,12),3.)],-1)
    result,next_q = delayed_action(a,q,torch.tensor([0.,1.,2.]))
    assert torch.equal(result[:,0],torch.tensor([1.,2.,3.]))
    assert torch.equal(next_q[:,12:],q[:,:12])
    result.sum().backward()
    assert a.grad[0].sum()==12 and a.grad[1:].sum()==0
    later,_=delayed_action(a,next_q,torch.ones(3))
    assert torch.equal(later,a)


def test_command_observation_and_replay_round_trip():
    gen=BallTarget(BallTargetCfg(),2,'cpu')
    gen.velocity_mode[:]=1
    gen.velocity_command[:]=torch.tensor([[.3,.1,.4],[.6,0.,-.5]])
    gen.action_delay[:]=torch.tensor([1.,2.])
    gen.action_queue[:]=torch.arange(24.)
    obs=oracle_observation(gen.state(),torch.zeros(2,3),torch.zeros(2))
    assert torch.equal(obs[:,:3],gen.velocity_command)
    assert (obs[:,3:]==0).all()
    other=BallTarget(BallTargetCfg(),2,'cpu'); unpack_generator(other,pack_generator(gen))
    assert torch.equal(other.action_queue,gen.action_queue)
    assert torch.equal(other.velocity_command,gen.velocity_command)


def test_new_physical_context_inputs_start_at_zero_weight():
    torch.manual_seed(5)
    hidden=(16,16)
    model=WorldModel(hidden)
    original={k:v.clone() for k,v in model.state_dict().items()}
    prefix=st.STATE_DIM+24
    new_scan=prefix+st.CONTEXT_DIM
    columns=torch.cat([torch.arange(prefix+4),torch.arange(new_scan,new_scan+SCAN_POINTS)])
    old={k:v.clone() for k,v in original.items()}
    old['net.0.weight']=old['net.0.weight'][:,columns]
    for key in ['in_norm.mean','in_norm.var']: old[key]=old[key][columns]
    assert expand_world_model_context(old,model)
    reference=mlp(len(columns),hidden,st.STATE_DIM+3)
    reference.load_state_dict({k[4:]:v for k,v in old.items() if k.startswith('net.')})
    s,a,ap,xi,relief=[torch.randn(4,d) for d in [st.STATE_DIM,12,12,st.CONTEXT_DIM,SCAN_POINTS]]
    old_input=torch.cat([s,a,ap,xi[:,:4],s[:,st.Z]+relief],-1)
    prediction=reference(((old_input-old['in_norm.mean'])/(old['in_norm.var'].sqrt()+1e-6)).clamp(-10,10))
    with torch.no_grad(): actual,pose=model(s,a,ap,xi,relief)
    assert torch.allclose(actual,s+prediction[:,:st.STATE_DIM],atol=1e-5)
    assert torch.allclose(pose,prediction[:,st.STATE_DIM:],atol=1e-5)


def test_new_phase_learning_rate_does_not_inherit_finished_decay():
    from go2_supermimic.supermimic.train import TrainCfg, lr_at
    cfg=TrainCfg(lr=2e-4,lr_schedule_start=29000,lr_decay_iterations=12000)
    assert abs(lr_at(cfg,29000)-2e-4)<1e-12
    assert lr_at(cfg,31000)>1e-4
    assert abs(lr_at(cfg,41000)-cfg.lr*cfg.lr_final_frac)<1e-12
