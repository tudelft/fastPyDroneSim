"""
    Quadrotor simulation step and controller functions to be GPU vectorized 

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

from __main__ import kerneller, GRAVITY
from libs.jitMath import motorDot, forcesAndMoments, quatRotate, quatDot, sgemv
import numba as nb
from numba import cuda
from math import sqrt, acos, hypot
import numpy as np


#%% sim stepper
@kerneller("void(f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, :, ::1], f4[:, :, ::1], f4, i4, f4[:, :, ::1])")
def step(x, d, itau, wmax, G1, G2, dt, log_to_idx, x_log):
    i1 = cuda.grid(1)

    x_local = cuda.local.array(17, dtype=nb.float32)
    for j in range(6,17):
        x_local[j] = x[i1, j]

    #pos   = x_local[0:3]
    #vel   = x_local[3:6]
    q     = x_local[6:10]
    Omega = x_local[10:13]
    omega = x_local[13:17]

    xdot_local = cuda.local.array(17, dtype=nb.float32)
    #posDot = xdot_local[0:3]
    velDot = xdot_local[3:6]
    qDot = xdot_local[6:10]
    OmegaDot = xdot_local[10:13]
    omegaDot = xdot_local[13:17]

    # workspace needed for a few jit functions
    work = cuda.local.array(4, dtype=nb.float32)

    #%% motor forces
    # motor model
    motorDot(omega, d[i1], itau[i1], wmax[i1], omegaDot)

    # forces and moments
    fm = cuda.local.array(6, dtype=nb.float32)
    forcesAndMoments( omega, omegaDot, G1[i1], G2[i1], fm, work[:4])

    for j in range(3):
        velDot[j] = fm[j] # still needs to be rotated, see next section
        OmegaDot[j] = fm[j+3]

    #%% kinematics
    quatRotate(q, velDot, work[:3])
    velDot[2] += GRAVITY

    quatDot(q, Omega, qDot)

    #%% step forward
    for j in range(0,3): 
        x[i1, j] += dt * x[i1, j+3] # position
        x[i1, j+3] += dt * xdot_local[j+3] # velocty

    # quaternion needs to be normalzied after stepping. do that efficiently
    for j in range(4):
        q[j] += dt * qDot[j]

    iqnorm = 1. / sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
    for j in range(4):
        x[i1, j+6] = q[j] * iqnorm

    for j in range(10,17): # Omega and omega
        x[i1, j] += dt * xdot_local[j]

    #%% save state
    if log_to_idx >= 0:
        for j in range(17):
            x_log[i1, log_to_idx, j] = x[i1, j]


#%% position controller
@kerneller("void(f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, :, ::1])")
def controller(x, d, posPs, velPs, pSets, G1pinv):
    i1 = cuda.grid(1)

    x_local = cuda.local.array(17, dtype=nb.float32)
    for j in range(17):
        x_local[j] = x[i1, j]

    pos   = x_local[0:3]
    vel   = x_local[3:6]
    qi    = x_local[6:10]; qi[0] = -qi[0]
    Omega = x_local[10:13]

    #%% position control
    # more better would be having an integrator, but I don't feel like making
    # another state for that
    forceSp = cuda.local.array(3, dtype=nb.float32)
    for j in range(3):
        # xy are controlled separately, which isn't nice, but easy to code
        #                       |---  velocity set point ---|
        forceSp[j] = velPs[i1, j] * ( ( posPs[i1, j] * (pSets[i1, j] - pos[j]) ) - vel[j] )

    forceSp[2] -= GRAVITY

    workspace = cuda.local.array(3, dtype=nb.float32)
    quatRotate(qi, forceSp, workspace)

    #%% attitude control
    # Calculate rate derivative setpoint to steer towards commanded tilt.
    # Conveniently forget about yaw
    # tilt error is tilt_error_angle * cross(actual_tilt, commanded_tilt)
    # but the singularities make computations a bit lengthy
    tiltErr = cuda.local.array(2, dtype=nb.float32) # roll pitch only
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
    v = cuda.local.array(4, dtype=nb.float32)

    v[0] = -cosTilt*fzsp if cosTilt > 0.  else 0. # z-force
    # rate derivative setpoints FIXME: hardcoded gains
    v[1] = 20. * (10. * tiltErr[0] - Omega[0])
    v[2] = 20. * (10. * tiltErr[1] - Omega[1])
    v[3] = 20. * (yawRateSp - Omega[2])

    #d[:] = G1pinv[i1] @ v
    sgemv(G1pinv[i1], v, d[i1])
    for j in range(4):
        d[i1, j] = 0. if d[i1, j] < 0. else 1. if d[i1, j] > 1. else d[i1, j]
