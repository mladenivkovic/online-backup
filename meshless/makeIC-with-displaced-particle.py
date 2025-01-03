###############################################################################
# This file is part of SWIFT.
# Copyright (c) 2016 Matthieu Schaller (matthieu.schaller@durham.ac.uk)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

import h5py
import sys
import random
from numpy import *

# Generates a swift IC file containing a perturbed cartesian distribution of particles
# at a constant density and pressure in a cubic box

# an additional particle is displaced: it is set at (i-1+dx/100, j-1+dy/100)*IPD
# where IPD = InterParticleDistance
# dx, dy are cmd line args that should vary between 0 and 200 so that it may circle
# around the central particle

# Parameters
periodic = 1  # 1 For periodic box
boxSize = 1.0
L = int(sys.argv[1])  # Number of particles along one axis
rho = 1.0  # Density
P = 1.0  # Pressure
gamma = 5.0 / 3.0  # Gas adiabatic index
fileName = (
    "displacedUniformPlane-"
    + sys.argv[2].zfill(3)
    + "-"
    + sys.argv[3].zfill(3)
    + ".hdf5"
)

dx = float(sys.argv[2]) / 100
dy = float(sys.argv[3]) / 100

# ---------------------------------------------------
numPart = L ** 2 + 1
mass = boxSize ** 2 * rho / numPart
internalEnergy = P / ((gamma - 1.0) * rho)

# Generate particles
coords = zeros((numPart, 3))
v = zeros((numPart, 3))
m = zeros((numPart, 1))
h = zeros((numPart, 1))
u = zeros((numPart, 1))
ids = zeros((numPart, 1), dtype="L")

for i in range(L):
    for j in range(L):
        index = i * L + j
        x = i * boxSize / L + boxSize / (2 * L) + boxSize / (2.0 * L)
        y = j * boxSize / L + boxSize / (2 * L) + boxSize / (2.0 * L)
        z = 0
        coords[index, 0] = x
        coords[index, 1] = y
        coords[index, 2] = z
        v[index, 0] = 0.0
        v[index, 1] = 0.0
        v[index, 2] = 0.0
        m[index] = mass
        h[index] = 1.23485 * boxSize / L
        u[index] = internalEnergy
        ids[index] = index

# add additional particle at the end
i = L // 2 - 1
j = L // 2 - 1
index = numPart - 1
x = (i - 1 + dx) * boxSize / L + boxSize / (2 * L) + boxSize / (2.0 * L)
y = (j - 1 + dy) * boxSize / L + boxSize / (2 * L) + boxSize / (2.0 * L)
#  print('adding particle at x=', x, 'y=', y)
z = 0
coords[index, 0] = x
coords[index, 1] = y
coords[index, 2] = z
v[index, 0] = 0.0
v[index, 1] = 0.0
v[index, 2] = 0.0
m[index] = mass
h[index] = 1.23485 * boxSize / L
u[index] = internalEnergy
ids[index] = numPart - 1

# --------------------------------------------------

# File
file = h5py.File(fileName, "w")

# Header
grp = file.create_group("/Header")
grp.attrs["BoxSize"] = boxSize
grp.attrs["NumPart_Total"] = [numPart, 0, 0, 0, 0, 0]
grp.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
grp.attrs["NumPart_ThisFile"] = [numPart, 0, 0, 0, 0, 0]
grp.attrs["Time"] = 0.0
grp.attrs["NumFilesPerSnapshot"] = 1
grp.attrs["MassTable"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
grp.attrs["Flag_Entropy_ICs"] = [0, 0, 0, 0, 0, 0]
grp.attrs["NumPart_Total"] = numPart
grp.attrs["Dimension"] = 2

# Units
grp = file.create_group("/Units")
grp.attrs["Unit length in cgs (U_L)"] = 1.0
grp.attrs["Unit mass in cgs (U_M)"] = 1.0
grp.attrs["Unit time in cgs (U_t)"] = 1.0
grp.attrs["Unit current in cgs (U_I)"] = 1.0
grp.attrs["Unit temperature in cgs (U_T)"] = 1.0

# Particle group
grp = file.create_group("/PartType0")
ds = grp.create_dataset("Coordinates", (numPart, 3), "d")
ds[()] = coords
ds = grp.create_dataset("Velocities", (numPart, 3), "f")
ds[()] = v
ds = grp.create_dataset("Masses", (numPart, 1), "f")
ds[()] = m
ds = grp.create_dataset("SmoothingLength", (numPart, 1), "f")
ds[()] = h
ds = grp.create_dataset("InternalEnergy", (numPart, 1), "f")
ds[()] = u
ds = grp.create_dataset("ParticleIDs", (numPart, 1), "L")
ds[()] = ids + 1

file.close()
