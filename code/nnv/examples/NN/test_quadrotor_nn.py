from turtle import pos
import matlab.engine
import matlab
import numpy as np
from typing import List
import itertools
import copy
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath("D:/1_study/5_CS521/nnv/code/nnv"))

from verse.agents.base_agent import BaseAgent
from verse.parser.parser import ControllerIR
from scipy.integrate import ode
from utils import control_input_list

def func1(t, vars, u):
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    bx = u[3]
    by = u[4]
    bz = u[5]
    # sc = u[6]

    # sc = -1 * sc

    vx = vars[6]
    vy = vars[7]
    vz = vars[8]

    dvx = 9.81 * np.sin(u1) / np.cos(u1)
    dvy = -9.81 * np.sin(u2) / np.cos(u2)


    # tmp1 = dvx * np.cos(sc) - dvy * np.sin(sc)
    # tmp2 = dvx * np.sin(sc) + dvy * np.cos(sc)
    # dvx = tmp1
    # dvy = tmp2


    dvz = u3 - 9.81
    dx = vx
    dy = vy
    dz = vz
    dref_x = bx
    dref_y = by
    dref_z = bz
    return [dref_x, dref_y, dref_z, dx, dy, dz, dvx, dvy, dvz]

class QuadrotorAgent(BaseAgent):
    def __init__(self, id):
        self.id = id 
        self.decision_logic = ControllerIR.empty()

    def TC_simulate(self, mode: List[str], initialCondition, time_bound, time_step, track_map):
        time_bound = float(time_bound)
        number_points = int(np.ceil(time_bound/time_step))
        t = [round(i*time_step, 10) for i in range(0, number_points)]
        u1 = initialCondition[9]
        u2 = initialCondition[10]
        u3 = initialCondition[11]
        bx = float(mode[0])
        by = float(mode[1])
        bz = float(mode[2])
        init = initialCondition[:9] 
        trace = [[0]+init+[u1,u2, u3]]
        for i in range(len(t)):
            r = ode(func1)
            r.set_initial_value(init)
            r.set_f_params([u1, u2, u3, bx, by, bz])
            res: np.ndarray = r.integrate(r.t+time_step)
            init = res.flatten().tolist()
            trace.append([t[i]+time_step]+init+[u1,u2,u3])
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

def verifyOneStep(init, v_ref, verify_step, simulate_step) -> List[AnalysisTreeNode]:
    '''
        init: [lb, ub] where lb = ub = [x_ref, y_ref, theta_ref, x, y, theta] initial state of the vehicle
        v_ref: [vx_ref, vy_ref]
    '''
    state_lb = init[0]
    state_ub = init[1]

    error_x_l = state_lb[3] - state_ub[0]
    error_x_u = state_ub[3] - state_lb[0]
    error_y_l = state_lb[4] - state_ub[1]
    error_y_u = state_ub[4] - state_lb[1]
    error_z_l = state_lb[5] - state_ub[2]
    error_z_u = state_ub[5] - state_lb[2]
    error_vx_l = state_lb[6] - v_ref[0] 
    error_vx_u = state_ub[6] - v_ref[0] 
    error_vy_l = state_lb[7] - v_ref[1] 
    error_vy_u = state_ub[7] - v_ref[1] 
    error_vz_l = state_lb[8] - v_ref[2] 
    error_vz_u = state_ub[8] - v_ref[2] 


    lb = matlab.double([[0.2*error_x_l], [0.2*error_y_l], [0.2*error_z_l], [0.1*error_vx_l], [0.1*error_vy_l], [0.1*error_vz_l]])
    ub = matlab.double([[0.2*error_x_u], [0.2*error_y_u], [0.2*error_z_u], [0.1*error_vx_u], [0.1*error_vy_u], [0.1*error_vz_u]])
    # print(lb)
    # print(ub)
    res = eng.verifyQuadrotorNN(lb, ub)
    
    # Determine the pseudo largest
    pseudo_largest = 0
    for i in range(8):    
        if res[i][1] > res[pseudo_largest][1]:
            pseudo_largest = i

    # Find the set intersecting the pseudo largest
    possible_actions = []
    pseudo_smallest = pseudo_largest
    for i in range(8):
        if res[i][1] > res[pseudo_largest][0]:
            possible_actions.append(control_input_list[i])
            if res[i][0] < res[pseudo_smallest][0]:
                pseudo_smallest = i
    
    res_list = []
    # possible_actions = [control_input_list[pseudo_smallest]]
    for action in possible_actions:
        node = AnalysisTreeNode(
            init = {'test':[[state_lb+action,state_ub+action]]},
            mode = {'test':[str(v_ref[0]),str(v_ref[1]),str(v_ref[2])]},
            agent = {'test':QuadrotorAgent('test')},
            start_time = 0,
            type = 'reachtube',
            assert_hits = None,
            trace = {}
        )

        from verse.scenario import ScenarioConfig
        tmp_verifier = Verifier(ScenarioConfig())
        res = tmp_verifier.postCont(node, verify_step, simulate_step, None, 1, reachability_method='DRYVR-DISC', params={'simtracenum':50})

        res_list.append(res)
    return res_list

import torch
from utils import quad_W1, quad_b1, quad_W2, quad_b2, quad_W3, quad_b3

class FFNNC(torch.nn.Module):
    def __init__(self, D_in=6, D_out=8):
        super(FFNNC, self).__init__()
        self.layer1 = torch.nn.Linear(D_in, 20)
        self.layer2 = torch.nn.Linear(20, 20)
        self.layer3 = torch.nn.Linear(20, D_out)

    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

model = FFNNC()
model.layer1.weight = torch.nn.Parameter(torch.FloatTensor(quad_W1))
model.layer1.bias = torch.nn.Parameter(torch.FloatTensor(quad_b1))
model.layer2.weight = torch.nn.Parameter(torch.FloatTensor(quad_W2))
model.layer2.bias = torch.nn.Parameter(torch.FloatTensor(quad_b2))
model.layer3.weight = torch.nn.Parameter(torch.FloatTensor(quad_W3))
model.layer3.bias = torch.nn.Parameter(torch.FloatTensor(quad_b3))

def simulateOneStep(init, v_ref, verify_step, simulate_step):
    error_x = init[3] - init[0]
    error_y = init[4] - init[1]
    error_z = init[5] - init[2]
    error_vx = init[6] - v_ref[0]
    error_vy = init[7] - v_ref[1]
    error_vz = init[8] - v_ref[2]
    
    l = torch.FloatTensor([0.2*error_x, 0.2*error_y, 0.2*error_z, 0.1*error_vx, 0.1*error_vy, 0.1*error_vz])
    res = model(l)
    res = res.detach().numpy()
    idx = np.argmax(res)
    u1, u2, u3 = control_input_list[idx]

    node = AnalysisTreeNode(
        init = {'test':init+[u1, u2, u3]},
        mode = {'test':[str(v_ref[0]),str(v_ref[1]),str(v_ref[2])]},
        agent = {'test':QuadrotorAgent('test')},
        start_time = 0,
        type = 'simulation',
        assert_hits = None,
        trace = {}
    )

    # tmp_verifier = Verifier(ScenarioConfig())
    # res = tmp_verifier.postCont(node, verify_step, simulate_step, None, 1)
    trace = node.agent['test'].TC_simulate(node.mode['test'], node.init['test'], verify_step, simulate_step, None)
    node.trace['test'] = trace
    return node

def perform_partition(lb: List[float], ub: List[float], interval: List[float]=[0.5,0.5,0.5,0.5,0.5,0.5]) -> List[List[List[float]]]: 
    assert len(lb) == len(ub) == len(interval), f"{len(lb)}, {len(ub)}, {len(interval)}"

    pi_list = []
    for i in range(len(interval)):
        lpi = int(np.floor(round(lb[i]/interval[i],10)))
        upi = int(np.ceil(round(ub[i]/interval[i],10)))
        if lpi==upi:
            pi_list.append([lpi])
        else:
            pi_list.append(list(range(lpi, upi)))
    partition_index_list = list(itertools.product(*pi_list))
    partition_list = []
    for partition_index in partition_index_list:
        partition_lb = []
        partition_ub = []
        for i in range(len(partition_index)):
            if lb[i]<partition_index[i]*interval[i]:
                partition_lb.append(partition_index[i]*interval[i])
            else:
                partition_lb.append(lb[i])
            
            if ub[i]>(partition_index[i]+1)*interval[i]:
                partition_ub.append((partition_index[i]+1)*interval[i])
            else:
                partition_ub.append(ub[i])
        
        partition_list.append([partition_lb, partition_ub])
    return partition_list

def not_in(partition_list, new_partition):
    if not partition_list:
        return True
    for partition in partition_list:
        for i in range(len(partition[0])):
            if partition[0][i] > new_partition[0][i] or partition[1][i] < new_partition[1][i]:
                return True 
    return False

def add_partition(partition_list: List[List[List[float]]], new_partitions: List[List[float]]):
    for new_partition in new_partitions:
        if not_in(partition_list, new_partition):
            partition_list.append(new_partition)

def perform_partition_multi(final_rects:List[List[List[float]]], interval: List[float]=[0.1,0.1,0.1]) -> List[List[List[float]]]:
    partition_list = []
    for rect in final_rects:
        lb = rect[0] 
        ub = rect[1]
        partitions = perform_partition(lb, ub, interval)
        # if len(partitions)>1:
        #     print("stop")
        add_partition(partition_list, partitions)
    return partition_list

def combine_partition2(partition_res_list: List[List[List[float]]]) -> List[List[List[float]]]:
    combined_idx_list = []
    combined_partitions = []
    for i in range(len(partition_res_list)):
        if i in combined_idx_list:
            continue
        base_partition = copy.deepcopy(partition_res_list[i])
        for j in range(i+1, len(partition_res_list)):
            test_partition = partition_res_list[j]
            if round(base_partition[0][3],10)==round(test_partition[0][3],10) and \
                round(base_partition[0][4],10)==round(test_partition[0][4],10) and \
                round(base_partition[0][5],10)==round(test_partition[0][5],10) and \
                round(base_partition[1][3],10)==round(test_partition[1][3],10) and \
                round(base_partition[1][4],10)==round(test_partition[1][4],10) and \
                round(base_partition[1][5],10)==round(test_partition[1][5],10):
                base_partition[0][0] = min(base_partition[0][0], test_partition[0][0])
                base_partition[1][0] = max(base_partition[1][0], test_partition[1][0])
                base_partition[0][1] = min(base_partition[0][1], test_partition[0][1])
                base_partition[1][1] = max(base_partition[1][1], test_partition[1][1])
                base_partition[0][2] = min(base_partition[0][2], test_partition[0][2])
                base_partition[1][2] = max(base_partition[1][2], test_partition[1][2])
                combined_idx_list.append(j)
        combined_partitions.append(base_partition)
    return combined_partitions

def combine_partition(partition_res_list: List[AnalysisTreeNode]) -> AnalysisTreeNode:
    '''
        Take a list of analysistreenode and input and combine these nodes into a single analysistreenode
    '''
    base_trace = copy.deepcopy(np.array(partition_res_list[0].trace['test']))
    for node in partition_res_list[1:]:
        base_trace[::2,1:] = np.minimum(
            base_trace[::2,1:],
            np.array(node.trace['test'])[::2,1:]
        )
        base_trace[1::2,1:] = np.maximum(
            base_trace[1::2,1:],
            np.array(node.trace['test'])[1::2,1:]
        )
    base_node = AnalysisTreeNode()
    base_node.trace['test'] = base_trace.tolist()
    return base_node

from multiprocessing import Pool
import time 

if __name__ == "__main__":
    start_time = time.time()
    start_point = [48.742640687118424, 79.98528137424009, 0]
    end_point = [53.74264068711783, 79.98528137424009, 3]
    total_time_span = 10
    time_span = 4.0

    xl_init = 48.49264
    yl_init = 79.79663
    zl_init = -0.34802
    vxl_init = 0
    vyl_init = 0
    vzl_init = 0
    xu_init = 48.49264+0.1
    yu_init = 79.79663+0.1
    zu_init = -0.34802+0.1
    vxu_init = 0
    vyu_init = 0
    vzu_init = 0
    
    assert time_span <= total_time_span
    vx_ref = (end_point[0] - start_point[0])/total_time_span
    vy_ref = (end_point[1] - start_point[1])/total_time_span
    vz_ref = (end_point[2] - start_point[2])/total_time_span

    large_time_step = 0.5
    small_time_step = 0.01

    num_post = round(time_span/large_time_step)
    time_point = [round(i*large_time_step,10) for i in range(num_post)]

    ref_x_init = start_point[0]
    ref_y_init = start_point[1]
    ref_z_init = start_point[2]
    xl = xl_init
    yl = yl_init
    zl = zl_init
    vxl = vxl_init
    vyl = vyl_init
    vzl = vzl_init
    xu = xu_init
    yu = yu_init
    zu = zu_init
    vxu = vxu_init
    vyu = vyu_init
    vzu = vzu_init
    res_list = []
    # partition_list = perform_partition([xl,yl,thetal], [xu,yu,thetau])
    # pool = Pool(processes = 8)
    init_rect_list = [[
        [xl, yl, zl, vxl, vyl, vzl],
        [xu, yu, zu, vxu, vyu, vzu]
    ]]
    combined_partition_list = []
    for t in time_point:
        print(t, len(init_rect_list))
        tmp_next_init_rect_list = []
        # Call verifyOneStep to verify one step
        tmp_res_list = []
        for init_rect in init_rect_list:
            # rect_list = perform_partition(init_rect[0], init_rect[1])
            # for rect in rect_list:
            for rect in [init_rect]:
                xl,yl,zl,vxl,vyl,vzl = rect[0]
                xu,yu,zu,vxu,vyu,vzu = rect[1]
                nodes = verifyOneStep(
                    [[ref_x_init,ref_y_init,ref_z_init,xl,yl,zl,vxl,vyl,vzl],[ref_x_init,ref_y_init,ref_z_init,xu,yu,zu,vxu,vyu,vzu]],
                    # init_rect,
                    [vx_ref,vy_ref,vz_ref],
                    large_time_step,
                    small_time_step
                )
                tmp_res_list += nodes

                # Get end rectangle from nodes and fill init_rect_list
                for node in nodes:
                    lb = node.trace['test'][-2]
                    ub = node.trace['test'][-1]
                    init_rect = [lb[4:10],ub[4:10]]
                    tmp_next_init_rect_list.append(init_rect)

                for node in nodes:
                    for i in range(len(node.trace['test'])):
                        node.trace['test'][i][0] += t 
                    res_list.append(node)

        init_rect_list = tmp_next_init_rect_list
        
        combined_partitions = combine_partition(tmp_res_list)
        # init_rect_list = combine_partition2(tmp_next_init_rect_list)
        combined_partition_list.append(combined_partitions)
        # for i in range(len(combined_partitions.trace['test'])):
        #     combined_partitions.trace['test'][i][0] += t
        # res_list.append(combined_partitions)
        ref_x_init = tmp_res_list[0].trace['test'][-1][1]
        ref_y_init = tmp_res_list[0].trace['test'][-1][2]
        ref_z_init = tmp_res_list[0].trace['test'][-1][3]
        # partition_list = [[combined_partitions.trace['test'][-2][4:10],combined_partitions.trace['test'][-1][4:10]]]
        # init_rect_list = perform_partition(combined_partitions.trace['test'][-2][4:10], combined_partitions.trace['test'][-1][4:10])

    from verse.plotter.plotter2D_old import *
    import matplotlib.pyplot as plt
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()
    fig5 = plt.figure()
    fig6= plt.figure()
    fig7 = plt.figure()
    for node in res_list:
        fig1 = plot_reachtube_tree(node,'test',0,[4],'r',fig1)
        fig2 = plot_reachtube_tree(node,'test',0,[5],'r',fig2)
        fig4 = plot_reachtube_tree(node,'test',0,[6],'r',fig4)
        fig5 = plot_reachtube_tree(node,'test',0,[7],'r',fig5)
        fig6 = plot_reachtube_tree(node,'test',0,[8],'r',fig6)
        fig7 = plot_reachtube_tree(node,'test',0,[9],'r',fig7)


    for j in range(100):
        ref_x_init = start_point[0]
        ref_y_init = start_point[1]
        ref_z_init = start_point[2]
        xl = xl_init
        yl = yl_init
        zl = zl_init
        vxl = vxl_init
        vyl = vyl_init
        vzl = vzl_init
        xu = xu_init
        yu = yu_init
        zu = zu_init
        vxu = vxu_init
        vyu = vyu_init
        vzu = vzu_init
        x = np.random.uniform(xl,xu)
        y = np.random.uniform(yl,yu)
        z = np.random.uniform(zl,zu)
        vx = np.random.uniform(vxl,vxu)
        vy = np.random.uniform(vyl,vyu)
        vz = np.random.uniform(vzl,vzu)
        res_list = []
        for t in time_point:
            # if t == 0.45:
            #     print('stop here')
            res:AnalysisTreeNode = simulateOneStep(
                [ref_x_init,ref_y_init,ref_z_init,x,y,z,vx,vy,vz],
                [vx_ref,vy_ref,vz_ref],
                large_time_step,
                small_time_step
            )
            for i in range(len(res.trace['test'])):
                res.trace['test'][i][0] += t
            res_list.append(res)
            ref_x_init = res.trace['test'][-1][1]
            ref_y_init = res.trace['test'][-1][2]
            ref_z_init = res.trace['test'][-1][3]
            x = res.trace['test'][-1][4]
            y = res.trace['test'][-1][5]
            z = res.trace['test'][-1][6]
            vx = res.trace['test'][-1][7]
            vy = res.trace['test'][-1][8]
            vz = res.trace['test'][-1][9]

        # from verse.plotter.plotter2D_old import *
        # import plotly.graph_objects as go
        xb1 = [np.inf, -np.inf]
        yb1 = [np.inf, -np.inf]
        xb2 = [np.inf, -np.inf]
        yb2 = [np.inf, -np.inf]
        xb3 = [np.inf, -np.inf]
        yb3 = [np.inf, -np.inf]
        for node in res_list:
            fig1 = plot_simulation_tree(node,'test',0,[4],'b',fig1, xb1, yb1)
            ax = fig1.gca()
            xb1 = ax.get_xlim()
            yb1 = ax.get_xlim()
            fig2 = plot_simulation_tree(node,'test',0,[5],'b',fig2, xb2, yb2)
            ax = fig2.gca()
            xb2 = ax.get_xlim()
            yb2 = ax.get_xlim()
            fig4 = plot_simulation_tree(node,'test',0,[6],'b',fig4, xb3, yb3)
            ax = fig3.gca()
            xb3 = ax.get_xlim()
            yb3 = ax.get_xlim()
            fig3 = plot_simulation_tree(node,'test',4,[5],'b',fig3,[-3,3],[-1,5])
            fig5 = plot_simulation_tree(node,'test',0,[7],'b',fig5)
            fig6 = plot_simulation_tree(node,'test',0,[8],'b',fig6)
            fig7 = plot_simulation_tree(node,'test',0,[9],'b',fig7)
            
    # ax1 = fig1.gca()
    # ax1.set_xlim([-0.1,0.6])
    # ax1.set_ylim([-10,10])
    # ax2 = fig2.gca()
    # ax2.set_xlim([-0.1,0.6])
    # ax2.set_ylim([-4,10])
    # ax3 = fig3.gca()
    # ax3.plot([0,0],[10,0],'g')
    # ax3.set_xlim([-3,3])
    # ax3.set_ylim([-2,12])
    plt.show()