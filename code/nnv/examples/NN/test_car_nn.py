import matlab.engine
import matlab
import numpy as np
from typing import List
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath("D:/1_study/5_CS521/nnv/code/nnv"))

from verse.agents.base_agent import BaseAgent
from verse.parser.parser import ControllerIR
from scipy.integrate import ode

def vehicle_dynamics(t, vars, args):
    curr_x = vars[3]
    curr_y = vars[4]
    curr_theta = vars[5] % (np.pi * 2)
    vr = args[0]
    delta = args[1]
    bx = args[2]
    by = args[3]

    if vr > 10:
        vr_sat = 10
    elif vr < -0:
        vr_sat = -0
    else:
        vr_sat = vr

    if delta > np.pi / 8:
        delta_sat = np.pi / 8
    elif delta < -np.pi / 8:
        delta_sat = -np.pi / 8
    else:
        delta_sat = delta

    # beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
    # dx = vr*np.cos(curr_theta+beta)
    # dy = vr*np.sin(curr_theta+beta)
    # dtheta = vr/Lr * np.sin(beta)
    dx = vr_sat * np.cos(curr_theta + delta_sat)
    dy = vr_sat * np.sin(curr_theta + delta_sat)
    dtheta = delta_sat
    dref_x = bx
    dref_y = by
    dref_theta = 0
    return [dref_x, dref_y, dref_theta, dx, dy, dtheta]

class CarAgent(BaseAgent):
    def __init__(self, id):
        self.id = id 
        self.decision_logic = ControllerIR.empty()

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, track_map):
        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step, 10) for i in range(0, number_points)]
        vr = initialCondition[6]
        delta = initialCondition[7]
        bx = float(mode[0])
        by = float(mode[1])
        init = initialCondition[:6] 
        trace = [[0]+init+[vr,delta]]
        for i in range(len(t)):
            r = ode(vehicle_dynamics)
            r.set_initial_value(init)
            r.set_f_params([vr, delta, bx, by])
            res: np.ndarray = r.integrate(r.t+time_step)
            init = res.flatten().tolist()
            trace.append([t[i]+time_step]+init+[vr,delta])
        return np.array(trace)

from verse.analysis.verifier import Verifier
from verse.analysis.analysis_tree import AnalysisTreeNode

def bloatCos(li, ui):
    lqout, lrem = np.divmod(li, np.pi)
    uqout, urem = np.divmod(ui, np.pi)
    if lqout == uqout:
        lo = min(np.cos(li), np.cos(ui))
        uo = max(np.cos(li), np.cos(ui))
    elif lqout == uqout-1:
        if lqout%2==0:
            lo = -1
            uo = max(np.cos(li), np.cos(ui))
        else:
            lo = min(np.cos(li), np.cos(ui))
            uo = 1
    else:
        lo = -1
        uo = 1
    return lo, uo 

def bloatSin(li, ui):
    lqout, lrem = np.divmod(li+np.pi/2, np.pi)
    uqout, urem = np.divmod(ui+np.pi/2, np.pi)
    if lqout == uqout:
        lo = min(np.sin(li), np.sin(ui))
        uo = max(np.sin(li), np.sin(ui))
    elif lqout == uqout-1:
        if lqout%2==0:
            lo = min(np.sin(li), np.sin(ui))
            uo = 1
        else:
            lo = -1
            uo = max(np.sin(li), np.sin(ui))
    else:
        lo = -1
        uo = 1
    return lo, uo 

def verifyOneStep(init, v_ref, verify_step, simulate_step):
    '''
        init: [lb, ub] where lb = ub = [x_ref, y_ref, theta_ref, x, y, theta] initial state of the vehicle
        v_ref: [vx_ref, vy_ref]
    '''
    state_lb = init[0]
    state_ub = init[1]
    error_x_l = state_lb[0] - state_ub[3]
    error_x_u = state_ub[0] - state_lb[3]
    error_y_l = state_lb[1] - state_ub[4]
    error_y_u = state_ub[1] - state_lb[4]
    error_theta_l = state_lb[2] - state_ub[5]
    error_theta_u = state_ub[2] - state_lb[5]
    error_theta_cos_l, error_theta_cos_u = bloatCos(error_theta_l, error_theta_u)
    error_theta_sin_l, error_theta_sin_u = bloatSin(error_theta_l, error_theta_u)


    lb = matlab.double([[error_x_l], [error_y_l], [error_theta_cos_l], [error_theta_sin_l]])
    ub = matlab.double([[error_x_u], [error_y_u], [error_theta_cos_u], [error_theta_sin_u]])
    res = eng.verifyCarNN(lb, ub)

    node = AnalysisTreeNode(
        init = {'test':[[state_lb+[res[0][0],res[1][0]],state_ub+[res[0][1],res[1][1]]]]},
        mode = {'test':[str(v_ref[0]),str(v_ref[1])]},
        agent = {'test':CarAgent('test')},
        start_time = 0,
        type = 'reachtube',
        assert_hits = None,
        trace = {}
    )

    from verse.scenario import ScenarioConfig
    tmp_verifier = Verifier(ScenarioConfig())
    res = tmp_verifier.postCont(node, verify_step, simulate_step, None, 1)
    return res

import torch
from utils import W1, b1, W2, b2

class FFNNC(torch.nn.Module):
    def __init__(self, D_in=4, H1=100):
        super(FFNNC, self).__init__()
        self.control1 = torch.nn.Linear(D_in, H1)
        self.control2 = torch.nn.Linear(H1, 2)

    def forward(self, x):
        h2 = torch.relu(self.control1(x))
        u = self.control2(h2)
        return u

model = FFNNC()
model.control1.weight = torch.nn.Parameter(torch.FloatTensor(W1))
model.control1.bias = torch.nn.Parameter(torch.FloatTensor(b1))
model.control2.weight = torch.nn.Parameter(torch.FloatTensor(W2))
model.control2.bias = torch.nn.Parameter(torch.FloatTensor(b2))

def simulateOneStep(init, v_ref, verify_step, simulate_step):
    error_x_l = init[0] - init[3]
    error_y_l = init[1] - init[4]
    error_theta_l = init[2] - init[5]
    
    l = torch.FloatTensor([error_x_l, error_y_l, np.cos(error_theta_l), np.sin(error_theta_l)])
    u = model(l)
    vr = u[0].item() 
    delta = u[1].item() 

    node = AnalysisTreeNode(
        init = {'test':init+[vr, delta]},
        mode = {'test':[str(v_ref[0]),str(v_ref[1])]},
        agent = {'test':CarAgent('test')},
        start_time = 0,
        type = 'simulation',
        assert_hits = None,
        trace = {}
    )

    from verse.scenario import ScenarioConfig
    # tmp_verifier = Verifier(ScenarioConfig())
    # res = tmp_verifier.postCont(node, verify_step, simulate_step, None, 1)
    trace = node.agent['test'].TC_simulate(node.mode['test'], node.init['test'], verify_step, simulate_step, None)
    node.trace['test'] = trace
    return node


if __name__ == "__main__":
    start_point = [0,0]
    end_point = [0,10]
    start_orientation = np.arctan2(end_point[1]-start_point[1], end_point[0]-start_point[0])
    total_time_span = 10
    time_span = 0.5
    assert time_span <= total_time_span
    vx = (end_point[0] - start_point[0])/total_time_span
    vy = (end_point[1] - start_point[1])/total_time_span

    large_time_step = 0.05
    small_time_step = 0.001

    num_post = round(time_span/large_time_step)
    time_point = [round(i*large_time_step,10) for i in range(num_post)]

    ref_x_init = start_point[0]
    ref_y_init = start_point[1]
    ref_theta_init = start_orientation
    xl = 1.4
    yl = -1.0
    thetal = np.pi/2
    xu = 1.5
    yu = -0.9
    thetau = np.pi/2
    res_list = []
    for t in time_point:
        # if t == 0.45:
        #     print('stop here')
        res:AnalysisTreeNode = verifyOneStep(
            [[ref_x_init,ref_y_init,ref_theta_init,xl,yl,thetal],[ref_x_init,ref_y_init,ref_theta_init,xu,yu,thetau]],
            [vx,vy],
            large_time_step,
            small_time_step
        )
        for i in range(len(res.trace['test'])):
            res.trace['test'][i][0] += t
        res_list.append(res)
        ref_x_init = res.trace['test'][-1][1]
        ref_y_init = res.trace['test'][-1][2]
        ref_theta_init = res.trace['test'][-1][3]
        xl = res.trace['test'][-2][4]
        yl = res.trace['test'][-2][5]
        thetal = res.trace['test'][-2][6]
        xu = res.trace['test'][-1][4]
        yu = res.trace['test'][-1][5]
        thetau = res.trace['test'][-1][6]

    from verse.plotter.plotter2D_old import *
    import matplotlib.pyplot as plt
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    for node in res_list:
        fig1 = plot_reachtube_tree(node,'test',0,[4],'r',fig1)
        fig2 = plot_reachtube_tree(node,'test',0,[5],'r',fig2)

    for j in range(10):
        ref_x_init = start_point[0]
        ref_y_init = start_point[1]
        ref_theta_init = start_orientation
        xl = 1.4
        yl = -1.0
        thetal = np.pi/2
        xu = 1.5
        yu = -0.9
        thetau = np.pi/2
        x = np.random.uniform(xl,xu)
        y = np.random.uniform(yl,yu)
        theta = np.random.uniform(thetal,thetau)
        res_list = []
        for t in time_point:
            # if t == 0.45:
            #     print('stop here')
            res:AnalysisTreeNode = simulateOneStep(
                [ref_x_init,ref_y_init,ref_theta_init,x,y,theta],
                [vx,vy],
                large_time_step,
                small_time_step
            )
            for i in range(len(res.trace['test'])):
                res.trace['test'][i][0] += t
            res_list.append(res)
            ref_x_init = res.trace['test'][-1][1]
            ref_y_init = res.trace['test'][-1][2]
            ref_theta_init = res.trace['test'][-1][3]
            x = res.trace['test'][-1][4]
            y = res.trace['test'][-1][5]
            theta = res.trace['test'][-1][6]

        # from verse.plotter.plotter2D_old import *
        # import plotly.graph_objects as go
        for node in res_list:
            fig1 = plot_simulation_tree(node,'test',0,[4],'b',fig1,[-0.1,0.6],[-10,10])
            fig2 = plot_simulation_tree(node,'test',0,[5],'b',fig2,[-0.1,0.6],[-4,4])
            fig3 = plot_simulation_tree(node,'test',4,[5],'b',fig3,[-3,3],[-1,5])
    ax1 = fig1.gca()
    ax1.set_xlim([-0.1,0.6])
    ax1.set_ylim([-10,10])
    ax2 = fig2.gca()
    ax2.set_xlim([-0.1,0.6])
    ax2.set_ylim([-4,10])
    ax3 = fig3.gca()
    ax3.plot([0,0],[10,0],'g')
    ax3.set_xlim([-3,3])
    ax3.set_ylim([-2,12])
    plt.show()