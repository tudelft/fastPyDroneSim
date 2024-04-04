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
from tqdm import tqdm
import asyncio
from time import time
from libs.wsInterface import wsInterface, dummyInterface
from crafts import QuadRotor, Rotor
GRAVITY = 9.80665


#%% sim config
gpu = True              # run on the GPU using cuda
viz = True              # stream pose to a websocket connection
log = True              # log states from GPU back to the CPU (setting False has no speedup on CPU)
position_control = True # use a simple position/attitude NDI controller
realtime = False         # wait every timestep to try and be real time

# length / number of parallel sims
dt = 0.01 # step time is dt seconds (forward Euler)
T = 10  # run for T seconds
if gpu: # number of simulations to run in parallel
    blocks = 128 # 128 or 256 seem best. Should be multiple of 32
    threads_per_block = 64 # depends on global memorty usage. 256 seems best without. Should be multiple of 64
    # dt 0.01, T 10, no viz, log_interval 0, no controller, blocks 256, threads 256, gpu = True --> 250M ticks/sec
    N = blocks * threads_per_block
else:
    N = 100 # cpu

# initial states: 0:3 pos, 3:6 vel, 6:10 quaternion, 10:13 body rates Omega, 13:17 motor speeds omega
x0 = np.random.random((N, 17)).astype(np.float32) - 0.5
x0[:, 6:10] /= np.linalg.norm(x0[:, 6:10], axis=1)[:, np.newaxis] # quaternion needs to be normalized

# other settings
viz_interval = 0.05 # visualize every viz_interval simulation-seconds
Nviz = 512 # max number of quadrotors to visualize
log_interval = 1    # log state every x iterations. Too low may cause out_of_memory on the GPU. False == 0


#%% drone data
G1s = np.empty((N, 4, 4), dtype=np.float32)
G2s = np.empty((N, 1, 4), dtype=np.float32)
omegaMaxs = np.empty((N, 4), dtype=np.float32)
taus = np.empty((N, 4), dtype=np.float32)

for i in tqdm(range(N), desc="Building crafts"):
    q = QuadRotor()
    q.setInertia(0.42, 1e-3*np.eye(3))
    q.rotors.append(Rotor([-0.1, 0.1, 0], dir='cw'))
    q.rotors.append(Rotor([0.1, 0.1, 0], dir='ccw'))
    q.rotors.append(Rotor([-0.1, -0.1, 0], dir='ccw'))
    q.rotors.append(Rotor([0.1, -0.1, 0], dir='cw'))

    q.fillArrays(i, G1s, G2s, omegaMaxs, taus)

# precompute stuff
itaus = 1. / taus


#%% controller data
# (FIXME: should be a weighted pseudoinverse!!)
G1pinvs = np.linalg.pinv(G1s) / (omegaMaxs*omegaMaxs)[:, :, np.newaxis]

# position setpoints --> uniform on rectangular grid
grid_size = int(np.ceil(np.sqrt(N)))
x_vals = np.linspace(-7, 7, grid_size)
y_vals = np.linspace(-7, 7, grid_size)
X, Y = np.meshgrid(x_vals, y_vals)
vectors = np.column_stack((X.ravel(), Y.ravel(), -1.5*np.ones_like(X.ravel())))
pSets = vectors[:N].astype(np.float32)

# position controller gains (attitude/rate hardcoded for now, sorry)
posPs = 2*np.ones((N, 3), dtype=np.float32)
velPs = 2*np.ones((N, 3), dtype=np.float32)


#%% import compute kernels
if gpu:
    from numba import cuda
    jitter = lambda signature: nb.cuda.jit(signature, fastmath=False, device=True, inline=False)
    kerneller = lambda signature: nb.cuda.jit(signature, fastmath=False, device=False)
    from libs.gpuKernels import step, controller
else:
    jitter = lambda signature: nb.jit(signature, nopython=True, fastmath=False)
    kerneller = lambda signature, map: nb.guvectorize(signature, map, target='parallel', nopython=True, fastmath=False)
    nb.set_num_threads(max(nb.config.NUMBA_DEFAULT_NUM_THREADS-4, 1))
    from libs.cpuKernels import step, controller


#%% allocate sim data
log_interval = log*5
iters = int(T / dt)
Nlog = int(iters / log_interval) if log_interval > 0 else 0

xs = x0.copy()
us = np.random.random((N, 4)).astype(np.float32)

xs_log = np.empty(
    (N, Nlog, 17), dtype=np.float32)
xs_log[:] = np.nan


if viz:
    print("initializing websocket. Awaiting connection... ")
    wsI = wsInterface(8765)
else:
    wsI = dummyInterface()

#%% loop
ep = 0
async def main():
    async with wsI as ws:
        tsAll = time()

        if gpu:
            d_us = cuda.to_device(us)
            d_xs = cuda.to_device(xs)
            d_xs_log = cuda.to_device(xs_log)
            d_itaus = cuda.to_device(itaus)
            d_omegaMaxs = cuda.to_device(omegaMaxs)
            d_G1s = cuda.to_device(G1s)
            d_G2s = cuda.to_device(G2s)

            d_posPs = cuda.to_device(posPs)
            d_velPs = cuda.to_device(velPs)
            d_pSets = cuda.to_device(pSets)
            d_G1pinvs = cuda.to_device(G1pinvs)
            cuda.synchronize()

        ts = time()
        ei = 0
        iters = int(T / dt)
        for i in tqdm(range(iters), desc="Running simulation"):
            if realtime:
                t = time()

            log_idx = -1
            if (log_interval > 0) and not (i % log_interval):
                log_idx = int(i / log_interval)

            if gpu:
                step[blocks,threads_per_block](d_xs, d_us, d_itaus, d_omegaMaxs, d_G1s, d_G2s, dt, log_idx, d_xs_log)
            else:
                step(xs, us, itaus, omegaMaxs, G1s, G2s, dt, log_idx, xs_log)

            if position_control:
                if gpu:
                    controller[blocks,threads_per_block](d_xs, d_us, d_posPs, d_velPs, d_pSets, d_G1pinvs)
                else:
                    controller(xs, us, posPs, velPs, pSets, G1pinvs)

            if viz  and  ws.ws is not None  and  not i % int(viz_interval/dt):
                # visualize every 0.1 seconds
                if gpu:
                    xs[:] = d_xs.copy_to_host()
                await ws.sendData(xs[::int(np.ceil(N/Nviz))].astype(np.float64))

            if realtime:
                global ep, e
                te = time()
                e = dt - ( te - t )
                ep = dt - ( te - ts ) / ( i + 1 )
                ei += ep
                #print(e, ep)
                await asyncio.sleep( max(0.01*ei + 10*ep + e, 0) )


        # make sure all threads complete before stopping the count
        if gpu:
            cuda.synchronize()

        runtime = time() - ts

        if gpu and (log_interval > 0):
            xs_log[:] = d_xs_log.copy_to_host()

        runtimeOverhead = time() - tsAll

    return runtime, runtimeOverhead

runtime, runtimeOverhead = asyncio.run(main())

print(f"Achieved {N*iters / runtime / 1e6:.2f}M ticks per second (sim only) across {runtime:.5f} seconds.")
print(f"Retrieved {Nlog*N / runtimeOverhead / 1e6:.2f}M datapoints per (sim plus overhead) second across {runtimeOverhead:.5f} seconds.")
