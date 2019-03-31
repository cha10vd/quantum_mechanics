import numpy as np
import argparse
import logging
from math import pi as PI
from scipy.special import comb

np.set_printoptions(formatter={'all': lambda x:'{:6.5f}'.format(x)} )

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser(description='Molecular overlap calculator')
parser.add_argument('input_file', default=None)

args = parser.parse_args()


""" Orbita coefficients are the alpha exponents, e.g. 
        exp^-alpha(r2-r1)^2
    Primitive coeffcients are the C1, c2, c3 in the STO-3G:
        c1\psi1 + c2\psi2 + c3\psi3 
    None of these are varied in the course of a quantum
    chemical calculation, helping to speed the process.
"""

coords = np.loadtxt(args.input_file+'.xyz')
prim_c = np.loadtxt(args.input_file+'.prim')
orbi_c = np.loadtxt(args.input_file+'.orb')
ang_mo = np.loadtxt(args.input_file+'.ang', dtype=int)

def f2(n): 
    n = int(n)
    res = 1;
    if n == -1 or n == 0:
        return 1
    for i in range(n, -1, -2): 
        if(i == 0 or i == 1): 
            return res; 
        else: 
            res *= i; 

def norm_coeff(alpha, ang_mo): # Done - 29.03.19
    ax, ay, az = ang_mo

    logging.debug('ANGULAR MOMENTA: {}, {}, {}'.format(ax, ay, az))
    logging.debug('CONTRACTION COEFF: {}'.format(alpha))
    logging.debug('TEST CALC: {}'.format(2*az-1))
    logging.debug('FACTORIAL TEST: 4!! = {}'.format(f2(4)))

    nc = ((2*alpha)/PI)**(3/4) * (4*alpha)**(0.5*(ax + ay + az)) / \
                                 (f2(2*ax-1)*f2(2*ay-1)*f2(2*az-1))**(0.5)
    return nc

def ov(alpha, beta, A, B, a, b):

    soma = [0.0, 0.0, 0.0]
    E_AB = np.exp(-((alpha*beta)/(alpha+beta)) * np.linalg.norm(A-B)**2)
    logging.debug('E_AB = {}'.format(E_AB))
    coeff = (PI/(alpha + beta))**0.5
    P = [0,0,0]
    i = 0
    j = 0

    for d in range(3):
        P[d] = (alpha*A[d] + beta*B[d])/(alpha+beta)
        logging.debug('product of the two gaussians is: {}'.format(P[d]))
        for i in range(max(1,a[d])):
            for j in range(max(1,b[d])):
                a_choose_i = comb(a[d], i)
                b_choose_j = comb(b[d], j)
                numerator = f2(i + j -1)
                denominator = (2*(alpha+beta))**(0.5*(i + j))
                coord_a = (P[d] - A[d])**(max((a[d]-i),0))
                coord_b = (P[d] - B[d])**(max((b[d]-j),0))

                current = a_choose_i * b_choose_j * (numerator/denominator) * coord_a * coord_b
                logging.debug(current)
                soma[d] += current
        soma[d] *= coeff
    soma = np.array(soma)
    prod = soma[0] * soma[1] * soma[2]
    return prod * E_AB

def calc_overlap(orb1, orb2): #, at1, at2):
    sum = 0
    for i in range(3):
        for j in range(3):
            sum += (norm_coeff(orbi_c[orb1,i], ang_mo[orb1]) * prim_c[orb1,i]) *\
                   (norm_coeff(orbi_c[orb2,j], ang_mo[orb2]) * prim_c[orb2,j]) *\
                    ov(orbi_c[orb1,i], orbi_c[orb2,j], coords[orb1], coords[orb2],\
                       ang_mo[orb1], ang_mo[orb2])

    return(sum)

#print(ov(orbi_c[0,0], orbi_c[6,0], coords[0], coords[2], ang_mo[0], ang_mo[6]))

overlap_mat = np.zeros((len(orbi_c),len(orbi_c)))

for a in range(0,7):
    for b in range(a,7):
        overlap_mat[a,b] = overlap_mat[b,a] = calc_overlap(a,b)

print(overlap_mat)
