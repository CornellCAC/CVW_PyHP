from __future__ import print_function, division
"""
An estimate of the numerical value of pi via Monte Carlo integration.
Computation is distributed across processors via MPI.
"""

import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


def throw_darts(n):
    """
    returns an array of n uniformly random (x,y) pairs lying within the 
    square that circumscribes the unit circle centered at the origin, 
    i.e., the square with corners at (-1,-1), (-1,1), (1,1), (1,-1)
    """
    darts = 2*np.random.random((n,2)) - 1
    return darts

def in_unit_circle(p):
    """
    returns a boolean array, whose elements are True if the corresponding 
    point in the array p is within the unit circle centered at the origin, 
    and False otherwise -- hint: use np.linalg.norm to find the length of a vector
    """
    return np.linalg.norm(p,axis=-1)<=1.0

def estimate_pi(n, block=100000):
    """
    returns an estimate of pi by drawing n random numbers in the square 
    [[-1,1], [-1,1]] and calculating what fraction land within the unit circle; 
    in this version, draw random numbers in blocks of the specified size, 
    and keep a running total of the number of points within the unit circle;
    by throwing darts in blocks, we are spared from having to allocate 
    very large arrays (and perhaps running out of memory), but still can get 
    good performance by processing large arrays of random numbers
    """
    total_number = 0
    i = 0
    while i < n:
        if n-i < block:
            block = n-i
        darts = throw_darts(block)
        number_in_circle = np.sum(in_unit_circle(darts))
        total_number += number_in_circle
        i += block
    return (4.*total_number)/n

def estimate_pi_in_parallel(comm, N):
    """
    on each of the available processes, 
    calculate an estimate of pi by drawing N random numbers;
    the master processes will assemble all of the estimates
    produced by all workers, and compute the mean and
    standard deviation across the independent runs
    """

    if rank == 0:
        data = [N for i in range(size)]
    else:
        data = None
    data = comm.scatter(data, root=0)
    #
    pi_est = estimate_pi(N)
    #
    pi_estimates = comm.gather(pi_est, root=0)
    if rank == 0:
        return pi_estimates


def estimate_pi_statistics(comm, Ndarts, Nruns_per_worker):
    results = []
    for i in range(Nruns_per_worker):
        result = estimate_pi_in_parallel(comm, Ndarts)
        if rank == 0:
            results.append(result)
    if rank == 0:
        pi_est_mean = np.mean(results)
        pi_est_std  = np.std(results)
        return pi_est_mean, pi_est_std

if __name__ == '__main__':
    """
    for N from 4**5 to 4**14 (integer powers of 4), 
    compute mean and standard deviation of estimates of pi
    by throwing N darts multiple times (Nruns_total times,
    distributed across workers)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        print("MPI size = {}".format(size))
        sys.stdout.flush()
    Nruns_total = 64
    Nruns_per_worker = Nruns_total // size
    #
    estimates = []
    for log4N in range(5,15):
        N = int(4**log4N)
        result = estimate_pi_statistics(comm, N, Nruns_per_worker)
        if rank == 0:
            pi_est_mean, pi_est_std = result
            estimates.append((N, pi_est_mean, pi_est_std))
            print(N, pi_est_mean, pi_est_std)
            sys.stdout.flush()
    if rank == 0:
        estimates = np.array(estimates)
        plt.figure()
        plt.errorbar(np.log2(estimates[:,0]), estimates[:,1], yerr=estimates[:,2])
        plt.ylabel('estimate of pi')
        plt.xlabel('log2(number of darts N)')
        plt.savefig('pi_vs_log2_N.png')
        plt.figure()
        plt.ylabel('log2(standard deviation)')
        plt.xlabel('log2(number of darts N)')
        plt.plot(np.log2(estimates[:,0]), np.log2(estimates[:,2]))
        plt.savefig('log2_std_vs_log2_N.png')
    MPI.Finalize()
