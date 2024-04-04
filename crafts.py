#!/usr/bin/env python3
"""
    Turns quadrotor physical data into a few matrices for fast simulations

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

class Rotor:
    def __init__(self, x=[0., 0., 0.], wmax=4000., Tmax=4., k=0., cm=0.01, tau=0.03, Izz=1e-6, dir='cw'):
        self.x = np.asarray(x)
        self.wmax = wmax
        self.Tmax = Tmax
        self.k = k
        self.cm = cm
        self.tau = tau
        self.Izz = Izz
        self.dir = -1. if dir=='cw' else +1.

class QuadRotor:
    def __init__(self):
        self.rotors = []
        self.M = np.eye(4)

    def setInertia(self, m, I):
        self.M[0,0] = m
        self.M[1:,1:] = I

    def fillArrays(self, idx, G1, G2, omegaMax, tau):
        omegaMax[idx][:] = [r.wmax for r in self.rotors]
        tau[idx][:] = [r.tau for r in self.rotors]
        G1[idx][0, :] = [-r.Tmax for r in self.rotors]
        G1[idx][1:, :] = np.array([r.Tmax*np.cross(r.x, np.array([0, 0, -1.])) for r in self.rotors]).T
        G1[idx][3, :] = [r.Tmax*r.cm*r.dir for r in self.rotors]
        G1[idx][:] = np.linalg.inv(self.M) @ G1[idx]
        G1[idx][:] /= (omegaMax[idx]*omegaMax[idx])[:, np.newaxis]

        G2[idx][0, :] = [r.Izz/self.M[3,3]*r.dir for r in self.rotors]

if __name__=="__main__":
    q = QuadRotor()
    q.setInertia(0.42, 1e-3*np.eye(3))
    q.rotors.append(Rotor([-0.1, 0.1, 0], dir='cw'))
    q.rotors.append(Rotor([0.1, 0.1, 0], dir='ccw'))
    q.rotors.append(Rotor([-0.1, -0.1, 0], dir='ccw'))
    q.rotors.append(Rotor([0.1, -0.1, 0], dir='cw'))

    G1 = np.empty((2, 4, 4))
    G2 = np.empty((2, 1, 4))
    wm = np.empty((2, 4))
    tau = np.empty((2, 4))

    q.fillArrays(0, G1, G2, wm, tau)
