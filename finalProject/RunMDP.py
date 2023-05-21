import sys
import argparse
import math

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from GridWorldMDP import *
SEED = 666
np.random.seed(SEED)


def compare_preference(V1, V2, true_p):
    '''
    :param V1: Value matrix for policy 1
    :param V2: Value matrix for policy 2
    :param true_p: User's true preference
    :return: True if the first one is preferred by the second one, abs(v1 - v2), difference
    '''

    v1 = np.average(V1 @ true_p)
    v2 = np.average(V2 @ true_p)

    print(f"v1 is {v1}, v2 is {v2}")
    if v1 > v2:
        return True, abs(v1 - v2)
    else:
        return False, abs(v1 - v2)


def gen_greedy_policy(p):
    '''
    :param p: Pi returned by mdp.valueIteration
    :return: numofstate * 5 matrix, each row is a one hot vector
    '''
    greedy_pi = np.zeros([len(p), 5])
    for i in range(len(p)):
        greedy_pi[i, int(p[i])] = 1.
    return greedy_pi


def find_greedy_policy(obj, args, pref):
    '''

    :param obj:
    :param args:
    :param pref:
    :return:
    '''
    V, Pi, n = obj.valueIteration(args["epsilon"], pref)
    return gen_greedy_policy(Pi)


if __name__ == '__main__':

    args = {"gamma": 0.99, "noise": 0.1, "epsilon": 0.01}
    mdp = GridWorldMDP(args["noise"], args["gamma"])
    K = 3  # No of objectives
    true_preference = np.array([0.7, 0.2, 0.1])
    C_alpha = 2*K  # Lemma 1
    indistinguishablity = 0.001  # Indistinguishablity

    # Step 0: Sanity check to make sure "do nothing" policy pi0 has value function 0
    Pi0 = np.zeros([mdp.numstates, 5])  # Pi0[state, :] is a probability vector
    Pi0[:, 4] = 1  # Deterministic action of doing nothing
    V0, n = mdp.policyEvaluation(args["epsilon"], Pi0)
    Pi0_drawing = np.zeros(mdp.numstates) + 4
    mdp.drawWorld(V0[:, 0], Pi0_drawing, f"C:/Users/hydep/robotics/policy_eval_0.png", savefig=True)

    # Step 1: Sanity check to make sure the estimation of MORL is correct
    for k in range(3):
        preference = np.zeros(3)
        preference[k] = 1
        V, Pi, n = mdp.valueIterationSanity(args["epsilon"], k)
        V1, Pi1, n1 = mdp.valueIteration(args["epsilon"], preference)
        mdp.drawWorld(V, Pi, f"C:/Users/hydep/robotics/sanity{k}.png", savefig=True)
        mdp.drawWorld(V1[:, k], Pi1, f"C:/Users/hydep/robotics/morl{k}.png", savefig=True)

    # Step 2: Identification of Basis Policies
    V_mat = np.zeros([K, K])  # [V^{\pi1}, V^{\pi2}, ...]
    policy_mat = np.zeros([K, mdp.numstates, 5])  # This is a probability vector
    preference_star = np.zeros(K); preference_star[0] = 1  # Initialize \pi^{e*}<- \pi^{e1}
    V_star, Pi_star, n_star = mdp.valueIteration(args["epsilon"], preference_star)
    # print(f"greedy policy is {gen_greedy_policy(Pi_star)}")
    policy_mat[0, :, :] = gen_greedy_policy(Pi_star)
    V_mat[:, 0] = np.average(V_star, axis=0)  #@TODO: need to change np.average
    # print(np.average(V_star, axis=0))
    # print(f"V star is {V_star}")

    for j in range(1, K, 1):
        preference_curr = np.zeros(K); preference_curr[j] = 1
        V_curr, Pi_curr, n_curr = mdp.valueIteration(args["epsilon"], preference_curr)
        V_mat[:, j] = np.average(V_curr, axis=0)
        res, delta = compare_preference(V_star, V_curr, true_preference)
        if not res:
            V_star = V_curr.copy()
            Pi_star = Pi_curr.copy()
            policy_mat[0, :, :] = gen_greedy_policy(Pi_star)
            preference_star = preference_curr.copy()

    pi_mat = np.zeros([K, K])  # (pi1, pi1, ...)  #@TODO: Policy matrix
    mu_mat = np.zeros([K, K])  # (mu1, mu2, ...)

    pi_mat[:, 0] = preference_star  #@TODO: It's wrong
    mu_mat[:, 0] = V_mat[:, 0] / np.linalg.norm(V_mat[:, 0], 2)
    # mdp.drawWorld(V_star@true_preference, Pi_star)
    # print(np.linalg.matrix_rank(V_mat))
    eigen_val, eigen_mat = np.linalg.eig(V_mat)
    for i in range(1, K, 1):
        orth_space = eigen_mat[:, (i-1):-1]
        v_j = np.zeros(K-i)
        w_j = np.zeros(K-i)
        for j in range(K-i):
            Vj1, _, _ = mdp.valueIteration(args["epsilon"], orth_space[:, j])
            Vj2, _, _ = mdp.valueIteration(args["epsilon"], -orth_space[:, j])
            if abs(np.average(Vj1 @ orth_space[:, j])) > abs(np.average(Vj2 @ (-orth_space[:, j]))):
                v_j[j] = abs(np.average(Vj1 @ orth_space[:, j]))
                w_j[j] = 0
            else:
                v_j[j] = abs(np.average(Vj2 @ (-orth_space[:, j])))
                w_j[j] = 1
        jmax = np.argmax(v_j)
        if v_j[jmax] > 0:
            if w_j[jmax] == 0:
                pi_mat[:, i] = orth_space[:, jmax]
                policy_mat[i, :, :] = find_greedy_policy(mdp, args, orth_space[:, jmax])
            else:
                pi_mat[:, i] = -orth_space[:, jmax]
                policy_mat[i, :, :] = find_greedy_policy(mdp, args, -orth_space[:, jmax])
            mu_mat[:, i] = orth_space[:, jmax]
        else:
            break

    # print(f"pi_mat is {pi_mat}")
    # print(f"mu_mat is {mu_mat}")
    # print(f"policy_mat is {policy_mat}")

    # Step 3: Computation of Basis Ratios
    V_mat = np.zeros([K, K])
    # Evaluate V based on estimated pi_hat
    for i in range(K):
        V, pi, n = mdp.valueIteration(args["epsilon"], pi_mat[:, i])
        V_mat[:, i] = np.average(V, axis=0)  #@TODO this might need adjustment

    d = np.linalg.matrix_rank(V_mat)  #@TODO Think about the case when d<K
    alpha_hat = np.zeros(d-1)

    if d < K:
        print(d)
        # print(np.linalg.matrix_rank(V_mat[0:d, ]))
        print(f"We are in the case when d is smaller than K, need to be careful")

    # alpha_hat[0] = C_alpha
    # policy1 = policy_mat[0, :, :]
    # policy2 = policy_mat[1, :, :] / alpha_hat[0] + Pi0 * (1 - 1 / alpha_hat[0])
    # print(f"policy 1 is {policy1}, policy2 is {policy2}")
    # mdp.policyEvaluation(args["epsilon"], policy1)
    # mdp.policyEvaluation(args["epsilon"], policy2)

    for i in range(d-1):
        print(f"counter {i}")
        l = 0; h = 2*C_alpha; alpha_hat[i] = C_alpha
        counter = 0
        while True:
            counter = counter + 1
            # print(f"Current counter is {counter}")
            if alpha_hat[i] > 1:  # Assume \pi_0 is a zero vector, need to statisfy V^{\pi_0} = 0, to see whether \pi_0  = 0 works
                policy1 = policy_mat[0, :, :]
                policy2 = policy_mat[i+1, :, :] / alpha_hat[i] + Pi0 * (1 - 1/alpha_hat[i])
                V1, _ = mdp.policyEvaluation(args["epsilon"], policy1)
                V2, _ = mdp.policyEvaluation(args["epsilon"], policy2)
                res, delta = compare_preference(V1, V2, true_preference)
                if delta < indistinguishablity:
                    break
                if res:
                    h = alpha_hat[i]; alpha_hat[i] = (l+h)/2
                else:
                    l = alpha_hat[i]; alpha_hat[i] = (l+h)/2
            else:
                policy1 = policy_mat[i+1, :, :]
                policy2 = policy_mat[0, :, :]*alpha_hat[i] + Pi0 * (1 - alpha_hat[i])
                V1, _ = mdp.policyEvaluation(args["epsilon"], policy1)
                V2, _ = mdp.policyEvaluation(args["epsilon"], policy2)
                res, delta = compare_preference(V1, V2, true_preference)
                if delta < indistinguishablity:
                    break
                if res:
                    l = alpha_hat[i];alpha_hat[i] = (l + h) / 2
                else:
                    h = alpha_hat[i];alpha_hat[i] = (l + h) / 2
            print(f"current delta is {delta}, counter is {counter}, alpha_hat[i] is {alpha_hat[i]}")

    print(f"alpha_hat is {alpha_hat}")

    # Step 4 Construct matrix A and find preference estimation
    A_hat = np.zeros([d, K])
    A_hat[0, :] = V_mat[:, 0]

    for i in range(1, d, 1):
        A_hat[i, :] = alpha_hat[i-1]*(V_mat[:, 0] - V_mat[:, i])

    print(A_hat)
    e1 = np.zeros([d, 1]); e1[0] = 1
    w_hat = np.linalg.pinv(A_hat) @ e1
    w_hat = w_hat / np.linalg.norm(w_hat, 2)

    print(w_hat)

    # Step 5 Evaluate error
    V_star, Pi_star, n_star = mdp.valueIteration(args["epsilon"], true_preference)
    V_hat, Pi_hat, n_hat = mdp.valueIteration(args["epsilon"], w_hat)
    res, delta = compare_preference(V_star, V_hat, true_preference)
    print(f"delta is {delta}")
    threshold = np.sqrt(K)**(d + 14/3) * indistinguishablity**(1/3)
    print(f"bound is {threshold}")
