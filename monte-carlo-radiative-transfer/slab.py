#!/usr/bin/env python3


#==================================================
# Monte Carlo RT in a slab.
# 
#==================================================



# globals

NPACK = 10
RMIN = 0
RMAX = 1
DRMAX = RMAX - RMIN
ALPHA = 10  # mean free path = 1/alpha = 1/(n*sigma) = 1/(kappa * rho)
TAU_MAX = ALPHA*DRMAX

# scattering and absorption cross sections
SIGMA_S = 0.001
SIGMA_A = 1


import numpy as np

np.random.seed(10)


def rand():
    return np.random.uniform(0,1)



#==============================
class packet():
#==============================
    """
    Class representing an MC packet
    """

    def __init__(self):
        self.r = RMIN
        self.theta = 0
        self.phi = 0
        self.nscatter = 0
        return


    def new_theta(self):
        """
        Compute theta after scattering
        """
        return np.arccos(rand())

    def new_phi(self):
        """
        Compute phi after scattering
        """
        return 2*np.pi*rand()

    def new_tau(self):
        """
        Compute new optical depth
        """
        return -np.log(rand())


    def is_absorbed(self):
        """
        Check whether the packet is absorbed
        """
        a = SIGMA_S / (SIGMA_S + SIGMA_A)
        if rand() < a:
            return True
        else:
            return False


    def update_position(self, L, theta, phi):
        """
        Move the packet along along distance L and angles theta and phi
        """

        x = self.r * np.sin(self.theta) * np.cos(self.phi)
        y = self.r * np.sin(self.theta) * np.sin(self.phi)
        z = self.r * np.cos(self.theta)

        dx = L * np.sin(theta) * np.cos(phi)
        dy = L * np.sin(theta) * np.sin(phi)
        dz = L * np.cos(theta)

        x += dx
        y += dy
        z += dy

        self.r = np.sqrt(x*x + y*y + z*z)
        self.phi = np.arctan(y/x)
        self.theta = np.arccos(z/self.r)

        return



    def write_output(self, is_outside=False):
        """
        Fill up output() object with data, return it
        """

        res = output()
        res.r = self.r
        res.theta = self.theta
        res.phi = self.phi
        res.nscatter = self.nscatter
        res.is_outside = is_outside

        return res


    def propagate(self):
        """
        propagate a packet for one scatter
        """

        # first move the packet to the new location
        new_tau = self.new_tau()
        L = new_tau / ALPHA
        new_theta = self.new_theta()
        new_phi = self.new_phi()

        self.update_position(L, new_theta, new_phi)


        # now check whether we need to end

        if self.r > RMAX:
            return self.write_output(is_outside=True)
        elif self.r < RMIN:
            # reset everything and restart
            self.__init__()
            return self.propagate()


        if self.is_absorbed():
            return self.write_output(is_outside=False)

        else:
            self.nscatter += 1
            return self.propagate()





#=======================
class output():
#=======================
    """
    A class to store all kinds of outputs
    so you can easily add stuff later
    """
    
    
    def __init__(self):

        self.r = None
        self.theta = None
        self.phi = None

        self.nscatter = None

        return







#====================
def main():
#====================


    # list where to store output
    results = [output() for pack in range(NPACK)]

    for pack in range(NPACK):

        p = packet()
        results[pack] = p.propagate()


    r = np.array([o.r for o in results])
    n = np.array([o.nscatter for o in results])
    print(r)
    print(n)
        

        
    
    return



if __name__ == "__main__":
    main()
