"""
    Quadrotor simulation step and controller functions to be CPU vectorized 

    Copyright (C) 2024 Till Blaha -- TU Delft

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from gym_sim import kerneller, GRAVITY
from libs.jitMath import motorDot, forcesAndMoments, quatRotate, quatDot, sgemv
import numba as nb
from numba import cuda
from math import sqrt, acos, hypot
import numpy as np
# import numba.np as np
        
# sim reward function
@kerneller(["void(f4[::1],f4[::1],f4[::1],i8,f4[:])"], "(states),(three),(four),()->()")
def reward_function(x, pset, motor_commands, global_step_counter,r): # now computes for a single state thing
    '''NOTE: paper uses orientation error, but is unclear as they use a scalar'''
    # reward scheduling
    # intial parameters
    Cp = 1 # position weight
    Cv = .0 # velocity weight
    Cq = 0 # orientation weight
    Ca = .0 # action weight og .334, but just learns to fly out of frame
    Cw = .0 # angular velocity weight 
    Crs = .2 # reward for survival
    Cab = 0.0 # action baseline

    # curriculum parameters
    Nc = 1e5 # interval of application of curriculum

    CpC = 1.2 # position factor
    Cplim = 20 # position limit

    CvC = 1.4 # velocity factor
    Cvlim = 1.5 # velocity limit

    CaC = 1.4 # orientation factor
    Calim = .5 # orientation limit

    pos   = x[0:3]
    vel   = x[3:6]
    q     = x[6:10]
    qd    = x[10:13]
    omega = x[13:17]

    # curriculum
    if global_step_counter % Nc == 0:
        Cp = min(Cp*CpC, Cplim)
        Cv = min(Cv*CvC, Cvlim)
        Ca = min(Ca*CaC, Calim)

    # r[0] = max(-1e3,-Cp*np.sum((pos-pset)**2) \
    #         - Cv*np.sum((vel)**2) \
    #             - Ca*np.sum((motor_commands-Cab)**2) \
    #                 - Cw*np.sum((qd)**2) \
    #                     + Crs)
    # print("pos penalty: ", -Cp*np.sum((pos)**2))
    # print("vel penalty: ", - Cv*np.sum((vel)**2))
    # print("action penalty: ", - Ca*np.sum((motor_commands-Cab)**2))
    # print("orientation penalty: ", -Cq*(1-q[0]**2)) aims to keep upright!
    # print("angular velocity penalty: ", - Cw*np.sum((qd)**2))
    # print("reward for survival: ", Crs)

    # in theory pos error max sqrt( .6)*2.5 = 1.94
    # vel error max sqrt(1000)*.005 = 0.158
    # qd error max sqrt(1000)*.00 = 0.
    # should roughly be between -2 and 2
    r[0] = - Cv*np.sum((vel)**2) \
            - Ca*np.sum((motor_commands-Cab)**2) \
                -Cq*(1-q[0]**2)\
                - Cw*np.sum((qd)**2) \
                    + Crs \
                        -Cp*np.sum((pos)**2) \

# sim done function
@kerneller(["void(f4[::1],b1[::1], f4[:])"], "(states)->(),()")
def check_done(xs, done, t):
        '''Check if the episode is done
        Will reset time to zero if done

        From Learning to fly in seconds repo:
        
        namespace rl_tools::rl::environments::multirotor::parameters::termination{
            template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
            constexpr typename rl_tools::rl::environments::multirotor::ParametersBase<T, TI, 4, REWARD_FUNCTION>::MDP::Termination classic = {
                    true,           // enable
                    0.6,            // position
                    10,         // linear velocity
                    10 // angular velocity
            };
            template<typename T, typename TI, TI ACTION_DIM, typename REWARD_FUNCTION>
            constexpr typename rl_tools::rl::environments::multirotor::ParametersBase<T, TI, 4, REWARD_FUNCTION>::MDP::Termination fast_learning = {
                true,           // enable
                0.6,            // position
                1000,         // linear velocity
                1000 // angular velocity
            };
        }'''
        # if any velocity in the abs(self.xs) array is greater than 10 m/s, then the episode is done
        # if any rotational velocity in the abs(self.xs) array is greater than 10 rad/s, then the episode is done
        done[0] = False
        t[0] += 1
        
        if np.sum(np.isnan(xs))!=0:
            print("something is nan...?!")
            done[0] = True
            t[0] = 0
        
        pos_threshold = np.sum((np.abs(xs[0:3])>0.6))
        velocity_threshold = np.sum((np.abs(xs[3:6]) > 1000))
        angular_threshold  = np.sum((np.abs(xs[10:13]) > 1000))
        time_threshold = t[0]>500
        if (pos_threshold +  velocity_threshold + angular_threshold + time_threshold)!= 0: # if not zero at least one would be true
            # print(pos_threshold,velocity_threshold,angular_threshold)
            done[0] = True
            t[0] = 0

# sim stepper
@kerneller(["void(f4[::1], f4[::1], f4[::1], f4[::1], f4[:, ::1], f4[:, ::1], f4, i4, f4[:, ::1])"], "(states),(n),(n),(n),(four,n),(one,n),(),(),(iters,states)")
def step(x, d, itau, wmax, G1, G2, dt, log_to_idx, x_log):
    #pos   = x[0:3]
    #vel   = x[3:6]
    q     = x[6:10]
    Omega = x[10:13]
    omega = x[13:17]

    xdot_local = np.empty(17, dtype=nb.float32)
    #posDot = xdot_local[0:3]
    velDot = xdot_local[3:6]
    qDot = xdot_local[6:10]
    OmegaDot = xdot_local[10:13]
    omegaDot = xdot_local[13:17]

    # workspace needed for a few jit functions
    work = np.empty(4, dtype=nb.float32)
    
    #%% motor forces
    # motor model
    motorDot(omega, d, itau, wmax, omegaDot)

    # forces and moments
    fm = np.empty(6, dtype=nb.float32)
    # 
    # print('motor commands: ',d)
    forcesAndMoments( omega, omegaDot, G1, G2, fm, work[:4])
    for j in range(3):
        velDot[j] = fm[j] # still needs to be rotated, see next section
        OmegaDot[j] = fm[j+3]

    #%% kinematics
    quatRotate(q, velDot, work[:3])
    velDot[2] += GRAVITY

    quatDot(q, Omega, qDot)

    #%% step forward
    for j in range(0,3): 
        x[j] += dt * x[j+3] # position
        x[j+3] += dt * xdot_local[j+3] # velocty

    # quaternion needs to be normalzied after stepping. do that efficiently
    for j in range(4):
        q[j] += dt * qDot[j]

    iqnorm = 1. / sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    for j in range(4):
        x[j+6] = q[j] * iqnorm

    for j in range(10,17): # Omega and omega
        x[j] += dt * xdot_local[j]
    #%% save state
    if log_to_idx >= 0:
        for j in range(17):
            x_log[log_to_idx, j] = x[j]

# sim stepper with disturbance injected
@kerneller(["void(f4[::1], f4[::1], f4[::1], f4[::1], f4[:, ::1], f4[:, ::1], f4, f4[::1], i4, f4[:, ::1])"], 
           "(states),(n),(n),(n),(four,n),(one,n),(),(six),(),(iters,states)")
def step_disturbance(x, d, itau, wmax, G1, G2, dt, disturbance, log_to_idx, x_log):
    q     = x[6:10]
    Omega = x[10:13]
    omega = x[13:17]

    xdot_local = np.empty(17, dtype=nb.float32)
    #posDot = xdot_local[0:3]
    velDot = xdot_local[3:6]
    qDot = xdot_local[6:10]
    OmegaDot = xdot_local[10:13]
    omegaDot = xdot_local[13:17]

    # workspace needed for a few jit functions
    work = np.empty(4, dtype=nb.float32)

    #%% motor forces
    # motor model
    motorDot(omega, d, itau, wmax, omegaDot)

    # forces and moments
    fm = np.empty(6, dtype=nb.float32)
    forcesAndMoments( omega, omegaDot, G1, G2, fm, work[:4])
    
    for j in range(3):
        velDot[j] = fm[j] + disturbance[j] # still needs to be rotated, see next section
        OmegaDot[j] = fm[j+3]  + disturbance[j+3]

    #%% kinematics
    quatRotate(q, velDot, work[:3])
    velDot[2] += GRAVITY

    quatDot(q, Omega, qDot)

    #%% step forward
    for j in range(0,3): 
        x[j] += dt * x[j+3] # position
        x[j+3] += dt * xdot_local[j+3] # velocty

    # quaternion needs to be normalzied after stepping. do that efficiently
    for j in range(4):
        q[j] += dt * qDot[j]

    iqnorm = 1. / sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    for j in range(4):
        x[j+6] = q[j] * iqnorm

    for j in range(10,17): # Omega and omega
        x[j] += dt * xdot_local[j]

    #%% save state
    if log_to_idx >= 0:
        for j in range(17):
            x_log[log_to_idx, j] = x[j]


# example position controller
@kerneller(["void(f4[::1], f4[::1], f4[::1], f4[::1], f4[::1], f4[:, ::1])"],  "(states),(n),(three),(three),(three),(n,four)")
def controller(x, d, posPs, velPs, pSets, G1pinv):
    '''
    x: state vector
    d: output vector which returns the motor commands
    posPs: position controller gains
    velPs: velocity controller gains
    pSets: position setpoints
    G1pinv: pseudoinverse of G1 matrix'''
    pos   = x[0:3] # position x,y,z
    vel   = x[3:6] # velocity x,y,z
    qi    = x[6:10].copy(); qi[0] = -qi[0] # quaternion w,x,y,z
    Omega = x[10:13] # body rates p,q,r

    #%% position control
    # more better would be having an integrator, but I don't feel like making
    # another state for that
    forceSp = np.empty(3, dtype=nb.float32)
    for j in range(3):
        # xy are controlled separately, which isn't nice, but easy to code
        #                       |---  velocity set point ---|
        forceSp[j] = velPs[j] * ( ( posPs[j] * (pSets[j] - pos[j]) ) - vel[j] )

    forceSp[2] -= GRAVITY

    workspace = np.empty(3, dtype=nb.float32)
    quatRotate(qi, forceSp, workspace)

    #%% attitude control
    # Calculate rate derivative setpoint to steer towards commanded tilt.
    # Conveniently forget about yaw
    # tilt error is tilt_error_angle * cross(actual_tilt, commanded_tilt)
    # but the singularities make computations a bit lengthy
    tiltErr = np.empty(2, dtype=nb.float32) # roll pitch only
    tiltErr[0] = 0.
    tiltErr[1] = 0.

    cosTilt = 1.

    # only control attitude, if there is measurable fzsp
    fzsp = sqrt(forceSp[0]**2 + forceSp[1]**2 + forceSp[2]**2)
    if (fzsp > 1e-5):
        ifzsp = 1. / fzsp
        for j in range(3):
            forceSp[j] *= ifzsp

        sinTilt = hypot(forceSp[0], forceSp[1]) # cross product magnitude
        cosTilt = -forceSp[2] # dot product 
        if (sinTilt < 1e-5):
            # either we are aligned with attitude set or 180deg away
            if (cosTilt < 0):
                # 180 deg, lets use roll (index 0), otherwise just keep 0
                tiltErr[0] = -np.pi
        else:
            tiltAngleOverSinTilt = acos(cosTilt) / sinTilt
            tiltErr[0] = +forceSp[1] * tiltAngleOverSinTilt
            tiltErr[1] = -forceSp[0] * tiltAngleOverSinTilt

    # control yaw to point foward at all times
    yawRateSp = 0.
    if (cosTilt > 0.):
        # more of less upright, lets control yaw
        # error angle is 90deg - angle(body_x, (0 1 0))
        acosAngle = 2*qi[1]*qi[2] - 2*qi[3]*qi[0]
        acosAngle = -1. if acosAngle < -1. else +1. if acosAngle > +1. else acosAngle
        yawE = 0.5*np.pi - acos(acosAngle)

        if (1 - 2*qi[2]*qi[2] - 2* qi[3]*qi[3]) < 0.:
            # dot product of global x and body x is positive! We are looking
            # backwards and the angle is relative to pi
            if yawE >= 0.:
                yawE = np.pi - yawE
            else:
                yawE = -np.pi - yawE

        yawRateSp = -1 * cosTilt * yawE

    #%% NDI allocation
    # pseudocontrols:
    v = np.empty(4, dtype=nb.float32)

    v[0] = -cosTilt*fzsp if cosTilt > 0.  else 0. # z-force
    # rate derivative setpoints FIXME: hardcoded gains
    v[1] = 20. * (10. * tiltErr[0] - Omega[0])
    v[2] = 20. * (10. * tiltErr[1] - Omega[1])
    v[3] = 20. * (yawRateSp - Omega[2])

    #d[:] = G1pinv[i1] @ v
    sgemv(G1pinv, v, d)
    for j in range(4):
        d[j] = 0. if d[j] < 0. else 1. if d[j] > 1. else d[j]

