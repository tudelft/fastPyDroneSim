#!/usr/bin/env python3

"""
    Vectorized quadrotor simulation with websocket pose output

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

import numpy as np
import numba as nb
from math import sqrt, acos, hypot
GRAVITY = 9.80665

### data

# sim data
N = 2048 # run N simulations in parallel
dt = 0.002 # step time is dt seconds (forward Euler)
T = 10  # run for T seconds

# drone data
omegaMaxs = np.tile(np.array(
    [4000., 4000., 4000., 4000.], dtype=np.float32), (N, 1)) \
    + np.random.random((N, 4)).astype(np.float32)
taus = np.tile(np.array(
    [0.03, 0.03, 0.03, 0.03], dtype=np.float32), (N, 1)) \
    + np.random.random((N, 4)).astype(np.float32)*1e-3
G1        = 1. / (4000.**2) * np.tile( np.array(
    [
        [0.,0.,0.,0.],
        [0.,0.,0.,0.],
        [-8.,-8.,-8.,-8.],
        [-300.,-300.,300.,300.],
        [-200.,+200.,-200.,+200.],
        [-50.,+50.,+50.,-50.],
    ], dtype=np.float32), (N,1,1)) + (np.random.random((N, 6, 4)).astype(np.float32) * 1e-8)
G2        = 1e-5 * np.tile(np.array(
    [
        [0.,0.,0.,0.],
        [0.,0.,0.,0.],
        [0.,0.,0.,0.],
        [0.,0.,0.,0.],
        [0.,0.,0.,0.],
        [-50.,+50.,+50.,-50.],
    ], dtype=np.float32), (N,1,1)) + (np.random.random((N, 6, 4)).astype(np.float32) * 1e-5)

# position setpoints
grid_size = int(np.ceil(np.sqrt(N)))
x_vals = np.linspace(-3, 3, grid_size)
y_vals = np.linspace(-3, 3, grid_size)

X, Y = np.meshgrid(x_vals, y_vals)

vectors = np.column_stack((X.ravel(), Y.ravel(), -1.5*np.ones_like(X.ravel())))

pSets = vectors[:N].astype(np.float32)

# position controller gains (attitude/rate hardcoded for now, sorry)
posP = 4*np.ones((N, 3), dtype=np.float32)
velP = 2*np.ones((N, 3), dtype=np.float32)
#velI = 0.1*np.ones((N, 3), dtype=np.float32) # not implemented yet
#velIlimit = np.ones((N, 3), dtype=np.float32)
#velEI = np.zeros((N, 3), dtype=np.float32)


### Numba stuff

def dummy_decorator(f=None, *args, **kwargs):
    def decorator(func):
        return func

    if callable(f):
        return f
    else:
        return decorator

vectorize = lambda signature, map: nb.guvectorize(signature, map, target='parallel', nopython=True)
jit = lambda signature: nb.jit(signature, nopython=True)
nb.set_num_threads(max(nb.config.NUMBA_DEFAULT_NUM_THREADS-4, 1))
#jit = lambda signature: nb.cuda.jit(signature)

#vectorize = dummy_decorator
#jit = dummy_decorator


### precompute

G1pinv = np.linalg.pinv(G1)


### states
x = np.empty((N, 17), dtype=np.float32)
xDot = np.zeros_like(x, dtype=np.float32)


### functions

@vectorize([(nb.float32[:], nb.float32, nb.float32[:])],
           "(states),(),(states)")
def step(xdot, dt, x):
    #x = x + dt * xdot
    for i in range(x.size):
        x[i] = x[i] + dt * xdot[i]

    n = sqrt(x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2)
    for i in range(6,10):
        x[i] /= n

@jit("void(i4,f4[:],f4[:],f4[:])")
def motorDot(i, w, d, wDot):
    wmax = omegaMaxs[i]
    tau = taus[i]
    #wDot[:] = (d*wmax - w) / tau
    for j in range(len(w)):
        d[j] = 0. if d[j] < 0. else 1. if d[j] > 1. else d[j] 
        wDot[j] = (sqrt(d[j])*wmax[j] - w[j]) / tau[j]

@jit("void(i4,f4[:],f4[:],f4[:])")
def forcesAndMoments(i, omega, omegaDot, fm):
    o2 = omega.copy()
    for j in range(omega.size):
        o2[j] *= omega[j]

    fm[:] = G1[i] @ o2   +   G2[i] @ omegaDot

@jit("void(f4[:],f4[:])")
def quatRotate(q, v):
    w, x, y, z = q

    # Construct quaternion matrix --> unchecked chatGPT code now
    rotM = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - w*z),         2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x**2 + z**2),   2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),         1 - 2*(x**2 + y**2)]
    ], dtype=np.float32)

    v[:] = rotM @ v

@jit("void(f4[:],f4[:],f4[:])")
def quatDot(q, O, qDot):
    Ox, Oy, Oz = O
    qw, qx, qy, qz = q
    qDot[0] = .5 * (-Ox*qx - Oy*qy - Oz*qz)
    qDot[1] = .5 * ( Ox*qw + Oz*qy - Oy*qz)
    qDot[2] = .5 * ( Oy*qw - Oz*qx + Ox*qz)
    qDot[3] = .5 * ( Oz*qw + Oy*qx - Ox*qy)


@vectorize([(nb.int32, nb.float32[:], nb.float32[:], nb.float32[:])],
                   '(),(states),(n),(states)')
def Dot(i, x, d, xDot):
    pos   = x[0:3].copy()
    vel   = x[3:6].copy()
    q     = x[6:10].copy()
    Omega = x[10:13].copy()
    omega = x[13:17].copy()

    # forces
    omegaDot = np.zeros_like(omega, dtype=np.float32)
    motorDot(i, omega, d, omegaDot)

    fm = np.zeros(6, dtype=np.float32)
    forcesAndMoments(i, omega, omegaDot, fm)

    ## kinematics
    qDot = np.zeros_like(q, dtype=np.float32)
    quatDot(q, Omega, qDot)

    acc = fm[:3].copy()
    quatRotate(q, acc)

    ## assign
    xDot[13:17] = omegaDot
    xDot[10:13] = fm[3:].copy()
    xDot[6:10] = qDot
    xDot[3:6] = acc
    xDot[5] += GRAVITY
    xDot[0:3] = vel

@vectorize([(nb.int32, nb.float32[:], nb.float32[:])],
                   '(),(states),(n)')
def Controller(i, x, d):
    pos = x[:3].copy()
    vel = x[3:6].copy()

    velSp = posP[i]/velP[i] * (pSets[i] - pos)
    velE = velSp - vel
    #velEI[i][:] = np.clip(velEI[i] + velE, -velIlimit[i], velIlimit[i])

    accSp = velP[i]*velE# + velI[i]*velEI[i]

    qi = x[6:10].copy()
    qi[0] = -qi[0]

    fsp = accSp - np.array([0, 0, GRAVITY], dtype=np.float32)
    quatRotate(qi, fsp)

    fzsp = np.linalg.norm(fsp)
    if (abs(fzsp) < 1e-5):
        # we are supposed to be falling, don't control attitude
        tiltE = np.array([0., 0., 0.], dtype=np.float32)
        cosTilt = 1.
    else:
        fsp /= fzsp

        tilt = np.array([-fsp[1], fsp[0], 0.], dtype=np.float32)

        sinTilt = hypot(fsp[0], fsp[1])
        cosTilt = -fsp[2] # dot prodict 
        if (sinTilt < 1e-5):
            # either we are aligned with attitude set or 180deg away
            if (cosTilt > 0):
                # aligned
                tiltE = np.array([0., 0., 0.], dtype=np.float32)
            else:
                # 180 deg
                tiltE = np.array([np.pi, 0., 0.], dtype=np.float32)
        else:
            tiltE = tilt / sinTilt * acos(cosTilt)

    OmegaSp = -200./20. * tiltE
    OmegaE = OmegaSp - x[10:13].copy()
    OmegaDotSp = 20. * OmegaE

    fspBody = cosTilt * np.array([0., 0., -fzsp], dtype=np.float32)
    fspBody[2] = 0. if fspBody[2] > 0. else fspBody[2]
    v = np.array([fspBody[0], fspBody[1], fspBody[2],
                  OmegaDotSp[0], OmegaDotSp[1], OmegaDotSp[2]], dtype=np.float32)
    #G1u = G1u * omegaMaxs[i]**2
    #d[:] = np.linalg.pinv(G1u) @ v
    G1pinvu = G1pinv[i] / (omegaMaxs[i]*omegaMaxs[i])[:, np.newaxis]
    d[:] = G1pinvu @ v
    for j in range(len(d)):
        d[j] = 0. if d[j] < 0. else 1. if d[j] > 1. else d[j]


### websocket stuff
ws = None

import asyncio
import websockets
import json
from time import time
async def handle_client(websocket, path):
    # Send data to the client
    global ws
    ws = websocket
    while True:
        await asyncio.Future() # run forever without blokcing?

async def start_server():
    # Start WebSocket server
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()  # run forever

async def mainDebug():
    # if the dummy decorators are used, use this main.
    # Nice for debugging, as it only runs a single sim without numba jit compiling

    # havent run this function in a while, may be buggy now

    x0 = np.zeros(17).astype(np.float32);
    #x0[6:10] = np.random.random(4) - 0.5
    x0[2] = -1.5
    x0[6] = 1.
    x0[7] = 0.
    x0[6:10] /= np.linalg.norm(x0[6:10])
    x[0, :] = x0.copy()

    coro = start_server()
    asyncio.create_task(coro)
    await asyncio.sleep(3)

    i = 0
    while True:
        d = np.zeros((N, 4), dtype=np.float32)
        Controller(0, x[0], d[0])
        Dot(0, x[0], d[0], xDot[0])
        step(xDot[0], dt, x[0])

        if ws is not None and not i % int(0.1/dt):
            data = {"id": 0, "pos": list(x[0, :3]), "quat": list(x[0, 6:10])}
            await ws.send(json.dumps(data))

        i += 1
        print(i)
        await asyncio.sleep(0.1)

ts = 0

async def main():
#def main():
    x0 = np.random.random((N, 17)).astype(np.float32) - 0.5
    #x0[:, :3] = 0.
    #x0[:, 3:6] = 0.
    #x0[:, 6] = 1.
    #x0[:, 7:10] = 0.
    x0[:, 6:10] /= np.linalg.norm(x0[:, 6:10], axis=1)[:, np.newaxis]
    #x0[:, 10:13] = 0.
    #x0[:, 13:17] = 0.

    x[:] = x0.copy()
    xDot = np.zeros_like(x, dtype=np.float32)

    idxs = np.arange(N, dtype=np.int32)


    coro = start_server()
    asyncio.create_task(coro)
    print("initialized websocket")
    await asyncio.sleep(2)

    d = np.random.random((N, 4)).astype(np.float32)

    global ts
    ts = time()
    for i in range(int(T / dt)):
        tStart = time()

        Controller(idxs, x, d)
        Dot(idxs, x, d, xDot)
        step(xDot, dt, x)

        if ws is not None and not i % int(0.1/dt):
            j = 0
            for xD in x.astype(np.float64):
                data = {"id": j, "pos": list(xD[:3]), "quat": list(xD[6:10])}
                #data = {"id": i, "pos": [1,2,3.], "quat": [1., 0., 0., 0.]}
                j += 1;
                await ws.send(json.dumps(data))

        dtReal = time() - tStart
        e = dt - dtReal
        #sleep(max(e, 0))


#asyncio.run(mainDebug())
asyncio.run(main())
#main()
runtime = time() - ts

print(f"Achieved {N*T / dt / runtime / 1e6:.2f}M ticks per second.")