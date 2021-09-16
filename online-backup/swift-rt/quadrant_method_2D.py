#!/usr/bin/env python3

#----------------------------------------------------
# Test the correction method to make injection more
# isotropic by dividing up the space centered on the
# source into 4 quadrants and counting the weights
# inside those quadrants, to then add a correction
# factor such that the flux sum of diagonally
# opposed quadrants sums up to zero.
# Here for the 2D case with 4 different particle
# weights based on their distance from the source.
# The script creates plots on its own.
# The particles are drawn from a random distribution.
#----------------------------------------------------

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.rcParams['text.usetex'] = True

rng = np.random.default_rng(666)

# set coordinates of source.
source_x = 0.5
source_y = 0.5

# total quantity to distribute
quantity_tot = 1.
# how many times to repeat experiment for same
# number of neighbours
nrepeat = 14
# How many particles we have besides the source
nparts = 1000



class particle():
    '''
    A class for particles.
    '''
    def __init__(self, rng, x, y):
        self.x = x
        self.y = y
        self.quantity = None
        self.uncorrected_quantity = None
        self.weight = None 

def assign_neighbour_weight(total_neighbours, r, r2list_of_neighbours, neighbour_weight_method):
    """
    Assign the neighbours some weight.
    total_neighbours: total number of neighbours for this realisation
    r: distance from source
    r2list_of_neighbours: list of squares of distance for all neighbours.
            This isn't what we'd usually get in a simulation, but it
            allows a flexible choice of weight while keeping the rest
            of the code uniform.

    returns: weight
        a float for the weight of this particle that satisfies
        0 <= weight <= 1 and sum over all weights = 1
    """

    if neighbour_weight_method == "equal":
        weight = 1./total_neighbours
    else:

        r2list = np.array(r2list_of_neighbours)
        rlist = np.sqrt(r2list)

        if neighbour_weight_method == "inverse_distance":
            weight_sum = np.sum(1./rlist)
            weight = 1./ r / weight_sum

        elif neighbour_weight_method == "inverse_distance_squared":
            weight_sum = np.sum(1./rlist**2)
            weight = 1./ r**2 / weight_sum

        elif neighbour_weight_method == "inverse_distance_cubed":
            weight_sum = np.sum(1./rlist**3)
            weight = 1./ r**3 / weight_sum

        else:
            print("ERROR: neighbour_weight_method =", neighbour_weight_method, "unknown")
            quit(1)

    return weight



def generate_particles(rng, particle_method):
    """
    Generate the particle configuration and the
    source
    """

    if particle_method == "random":

    elif particle_method == "uniform":
        parts = None
        source = None

    elif particle_method == "glass":
        parts = None
        source = None

    return parts, source


def run_realisation(rng, number_of_neighbours, iteration, neighbour_weight_method):
    """
    Run a realisation for a given number of number of neighbours
    to include

    rng: instance of numpy random number generator
    number_of_neighbours: integer number of neighbours to use
    iteration: current iteration of the realisation
    neighbour_weight_method: which method to use to determine weight of neighbours
    """

    if number_of_neighbours > nparts:
        print("You are asking for more neighbours than there are particles:")
        print(number_of_neighbours, "vs", nparts)
        quit()

    # generate particles
    parts, source = generate_particles(rng, particle_method)

    #--------------------
    # Find neighbours
    #--------------------

    # brute force is good enough for this example with few particles
    r2list = []
    for p in parts:
        dx = p.x - source.x
        dy = p.y - source.y
        r2 = dx**2 + dy**2
        r2list.append(r2)
    sortind = np.argsort(r2list)
    neighbours = []
    r2list_of_neighbours = []
    for n in range(number_of_neighbours):
        neighbours.append(parts[sortind[n]])
        r2list_of_neighbours.append(r2list[sortind[n]])

    if len(neighbours) != number_of_neighbours:
        print("didn't find enough neighbours..??", len(neighbours), "found; expect", number_of_neighbours)


    #---------------------------------
    # Assign neighbours some weight
    #---------------------------------

    weightcheck = 0.

    for n in neighbours:
        dx = n.x - source.x
        dy = n.y - source.y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        n.weight = assign_neighbour_weight(number_of_neighbours, r, r2list_of_neighbours, neighbour_weight_method)
        weightcheck += n.weight
    
    if abs(1. - weightcheck) > 1e-5:
        print("Warning: weightcheck N_ngb {0:d} iteration {1:d}: {2:.6f}".format(number_of_neighbours, iteration, weightcheck))

    quadrant_count = [0, 0, 0, 0, 0, 0, 0, 0]

    #       dx      dy      dz      opposite
    # 0:    < 0     < 0     < 0     7
    # 1:    > 0     < 0     < 0     6
    # 2:    < 0     > 0     < 0     5
    # 3:    > 0     > 0     < 0     4
    # 4:    < 0     < 0     > 0     3
    # 5:    > 0     < 0     > 0     2
    # 6:    < 0     > 0     > 0     1
    # 7:    > 0     > 0     > 0     0


    #--------------------------------------------
    # First loop: Gather weights and directions
    #--------------------------------------------
    for n in neighbours:
        dx = n.x - source.x
        dy = n.y - source.y
        quadrant_index = 0
        if dx > 0:
            quadrant_index += 1
        if dy > 0:
            quadrant_index += 2

        quadrant_count[quadrant_index] += n.weight


    #--------------------------------------------
    # Second loop: Distribute quantities
    #--------------------------------------------
    renormalization_sum = 0.  # for checks: should be equal to number of neighbours
    for n in neighbours:
        dx = n.x - source.x
        dy = n.y - source.y
        r2 = dx**2 + dy**2
        r = np.sqrt(r2)
        unit_vector = np.array([dx/r, dy/r])
        quadrant_index = 0
        if dx > 0:
            quadrant_index += 1
        if dy > 0:
            quadrant_index += 2

        opposite_quadrant_index = 3 - quadrant_index # for 2D
        #  opposite_quadrant_index = 7 - quadrant_index # for 3D

        #  renormalization = quadrant_count[opposite_quadrant_index] / quadrant_count[quadrant_index]
        if quadrant_count[opposite_quadrant_index] == 0:
            renormalization = 1.
        else:
            diagonal_sum = quadrant_count[opposite_quadrant_index] + quadrant_count[quadrant_index]
            renormalization = diagonal_sum / quadrant_count[quadrant_index] * 0.5
            # in case we have empty quadrants:
            if renormalization == 0.:
                print("renormalization = 0???")
        renormalization_sum += renormalization
        n.quantity = quantity_tot * n.weight * renormalization * unit_vector
        n.uncorrected_quantity = quantity_tot * n.weight * unit_vector

    quantity_sum = 0.
    quantity_vector_sum = 0.
    quantity_uncorrected_sum = 0.
    quantity_vector_sum = 0.
    quantity_uncorrected_vector_sum = 0.

    for n in neighbours:
        quantity_sum += np.sqrt(np.sum(n.quantity**2))
        quantity_vector_sum += n.quantity
        quantity_uncorrected_sum += np.sqrt(np.sum(n.uncorrected_quantity**2))
        quantity_uncorrected_vector_sum += n.uncorrected_quantity

    # normalize sums by expected total sum
    quantity_sum /= quantity_tot
    quantity_uncorrected_sum /= quantity_tot

    quantity_vector_sum /= quantity_tot
    quantity_uncorrected_vector_sum /= quantity_tot

    #  print("{0:.3f} {1:.3f}".format(quantity_vector_sum, quantity_uncorrected_vector_sum))
    #  print(quantity_sum, quantity_uncorrected_sum, quantity_vector_sum, quantity_uncorrected_vector_sum)
    #  print(iteration, quantity_sum, quantity_uncorrected_sum)


    #  fig = plt.figure()
    #  xp = [p.x for p in parts]
    #  yp = [p.y for p in parts]
    #  xn = [n.x for n in neighbours]
    #  yn = [n.y for n in neighbours]
    #  ax1 = fig.add_subplot(121, aspect="equal")
    #  ax2 = fig.add_subplot(122, aspect="equal")
    #  for ax in [ax1, ax2]:
    #      ax.scatter(xp, yp, c='C0')
    #      ax.scatter(xn, yn, c="C1")
    #      ax.scatter(source.x, source.y, c="red")
    #      ax.set_xlim(0.35, 0.65)
    #      ax.set_ylim(0.35, 0.65)
    #  for n in neighbours:
    #      ax1.arrow(n.x, n.y, n.uncorrected_quantity[0], n.uncorrected_quantity[1])
    #      ax2.arrow(n.x, n.y, n.quantity[0], n.quantity[1])
    #
    #  ax1.set_title("without correction")
    #  ax2.set_title("with correction")
    #  plt.show()

    return quantity_sum, quantity_vector_sum, quantity_uncorrected_sum, quantity_uncorrected_vector_sum


class results():
    """
    encapsulates results
    """

    def __init__(self, quantity_list, vector_quantity_list, uncorrected_quantity_list, uncorrected_vector_quantity_list):
        vq = np.sqrt(np.sum(np.array(vector_quantity_list)**2, axis=1))
        self.vector_diff_min = vq.min()
        self.vector_diff_max = vq.max()
        self.vector_diff_mean = vq.mean()

        vquc = np.sqrt(np.sum(np.array(uncorrected_vector_quantity_list)**2, axis=1))
        self.vector_diff_uncorrected_min = vquc.min()
        self.vector_diff_uncorrected_max = vquc.max()
        self.vector_diff_uncorrected_mean = vquc.mean()

        ql = np.array(quantity_list)
        self.magnitude_min = ql.min()
        self.magnitude_max = ql.max()
        self.magnitude_mean = ql.mean()

        qluc = np.array(uncorrected_quantity_list)
        self.magnitude_uncorrected_min = qluc.min()
        self.magnitude_uncorrected_max = qluc.max()
        self.magnitude_uncorrected_mean = qluc.mean()



def plot_results(number_of_neighbours, results_particle_method, neighbour_weight_method_titles, particle_generation_methods):
    """
    Create a plot and save the figures.
    """
    print("plotting")


    minsymbol = "v"
    maxsymbol = "^"
    meansymbol = "o"

    ncols = len(neighbour_weight_method_titles)
    nrows = len(particle_generation_methods)

    fig = plt.figure(figsize=(5*ncols, 5*nrows))
    plotcounter = 1
    for row, partmethod in enumerate(particle_generation_methods):
        for col, method in enumerate(neighbour_weight_method_titles):
            ax = fig.add_subplot(nrows, ncols, plotcounter)
            plotcounter += 1
            resultlist = results_particle_method[row][col]

            corr_min = [r.magnitude_min for r in resultlist]
            corr_max = [r.magnitude_max for r in resultlist]
            corr_mean = [r.magnitude_mean for r in resultlist]

            uncorr_min = [r.magnitude_uncorrected_min for r in resultlist]
            uncorr_max = [r.magnitude_uncorrected_max for r in resultlist]
            uncorr_mean = [r.magnitude_uncorrected_mean for r in resultlist]

            ax.scatter(number_of_neighbours, corr_max, marker=maxsymbol, c="C0", label="Max $\sum_i |\mathbf{F}_i|$ with correction", alpha=0.6)
            ax.scatter(number_of_neighbours, corr_mean, marker=meansymbol, c="C0", label="Average $\sum_i |\mathbf{F}_i|$ with correction", alpha=0.6)
            ax.scatter(number_of_neighbours, corr_min, marker=minsymbol, c="C0", label="Min $\sum_i |\mathbf{F}_i|$ with correction", alpha=0.6)

            ax.scatter(number_of_neighbours, uncorr_max, marker=maxsymbol, c="C1", label="Max $\sum_i |\mathbf{F}_i|$", alpha=0.6)
            ax.scatter(number_of_neighbours, uncorr_mean, marker=meansymbol, c="C1", label="Average $\sum_i |\mathbf{F}_i|$", alpha=0.6)
            ax.scatter(number_of_neighbours, uncorr_min, marker=minsymbol, c="C1", label="Min $\sum_i |\mathbf{F}_i|$", alpha=0.6)

            ax.set_xscale("log")
            ax.set_xlabel("Number of neighbours")
            title = "Neighbour weights $\propto$ " + method
            ax.set_title(title)

            if col == ncols - 1 and row == 0:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="small")
            if col == 0:
                ax.set_ylabel(partmethod + " particle distribution")
            else:
                ax.set_ylabel("$\sum_i |\mathbf{F}_i|$ / expected $\sum_i |\mathbf{F}_i| $")

    plt.tight_layout()

    figname = "quadrant_correction_total_quantity-2D.png"
    plt.savefig(figname, dpi=300)
    #  plt.show()


    fig2 = plt.figure(figsize=(5*ncols, 5*nrows))
    plotcounter = 1
    for row, partmethod in enumerate(particle_generation_methods):
        for col, method in enumerate(neighbour_weight_method_titles):
            ax = fig2.add_subplot(nrows, ncols, plotcounter)
            plotcounter += 1
            resultlist = results_particle_method[row][col]

            corr_vec_min = [r.vector_diff_min for r in resultlist]
            corr_vec_max = [r.vector_diff_max for r in resultlist]
            corr_vec_mean = [r.vector_diff_mean for r in resultlist]

            uncorr_vec_min = [r.vector_diff_uncorrected_min for r in resultlist]
            uncorr_vec_max = [r.vector_diff_uncorrected_max for r in resultlist]
            uncorr_vec_mean = [r.vector_diff_uncorrected_mean for r in resultlist]

            ax.scatter(number_of_neighbours, corr_vec_max, marker=maxsymbol, c="C0", label="Max $|\sum_i \mathbf{F}_i|$ with correction", alpha=0.6)
            ax.scatter(number_of_neighbours, corr_vec_mean, marker=meansymbol, c="C0", label="Average $|\sum_i \mathbf{F}_i|$ with correction", alpha=0.6)
            ax.scatter(number_of_neighbours, corr_vec_min, marker=minsymbol, c="C0", label="Min $|\sum_i \mathbf{F}_i|$ with correction", alpha=0.6)

            ax.scatter(number_of_neighbours, uncorr_vec_max, marker=maxsymbol, c="C1", label="Max $|\sum_i \mathbf{F}_i|$", alpha=0.6)
            ax.scatter(number_of_neighbours, uncorr_vec_mean, marker=meansymbol, c="C1", label="Average $|\sum_i \mathbf{F}_i|$ ", alpha=0.6)
            ax.scatter(number_of_neighbours, uncorr_vec_min, marker=minsymbol, c="C1", label="Min $|\sum_i \mathbf{F}_i|$", alpha=0.6)
            
            ax.set_xscale("log")
            ax.set_xlabel("Number of neighbours")
            ax.set_ylabel("$\sum_i \mathbf{F}_i$ / expected $\sum_i |\mathbf{F}_i| $")
            title = "Neighbour weights $\propto$ " + method
            ax.set_title(title)

            if col == ncols - 1 and row == 0:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="small")
            if col == 0:
                ax.set_ylabel(partmethod + " particle distribution")
            else:
                ax.set_ylabel("$|\sum_i \mathbf{F}_i|$ / expected $\sum_i |\mathbf{F}_i| $")

    plt.tight_layout()

    figname = "quadrant_correction-vectorial_quantity-2D.png"
    plt.savefig(figname, dpi=300)
    #  plt.show()


    return


if __name__ == "__main__":

    neighbour_weight_methods = ["equal", "inverse_distance", "inverse_distance_squared", "inverse_distance_cubed"]
    neighbour_weight_method_titles = ["1", "$r^{-1}$", "$r^{-2}$", "$r^{-3}$"]
    nneigh = [10, 15, 20, 25, 40, 50, 60, 80, 100, 200, 500]
    particle_generation_methods = ["random", "uniform", "glass"]

    #  neighbour_weight_methods = ["equal"]
    #  neighbour_weight_method_titles = ["1"]
    #  nneigh = [20]#, 50, 100]
    #  particle_generation_methods = ["random"]

    results_particle_method = []
    for particle_method in particle_generation_methods:

        results_neighbour_weight_method = []
        for neighbour_weight_method in neighbour_weight_methods:
            resultlist = []

            for n in nneigh:
                # re-set the random number generator
                # so we re-generate the same particle
                # configurations
                rng = np.random.default_rng(666)
                sum_iter = []
                sum_iter_vector = []
                sum_iter_uncorrected = []
                sum_iter_vector_uncorrected = []
                for repeat in range(nrepeat):
                    qsum, qvecsum, qsum_nocorr, qvecsum_nocorr = run_realisation(rng, n, repeat, neighbour_weight_method, particle_method)
                    sum_iter.append(qsum)
                    sum_iter_vector.append(qvecsum)
                    sum_iter_uncorrected.append(qsum_nocorr)
                    sum_iter_vector_uncorrected.append(qvecsum_nocorr)

                res = results(sum_iter, sum_iter_vector, sum_iter_uncorrected, sum_iter_vector_uncorrected)
                resultlist.append(res)
            results_neighbour_weight_method.append(resultlist)
        results_particle_method.append(results_neighbour_weight_method)

    plot_results(nneigh, results_particle_method, neighbour_weight_method_titles, particle_generation_methods)
