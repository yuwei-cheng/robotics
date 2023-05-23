import pickle
import matplotlib.pyplot as plt
import numpy as np
from GridWorldMDP import *
SEED = 666
np.random.seed(SEED)

if __name__ == "__main__":
    # Load saved data
    with open('./true_preference_list.pkl', 'rb') as f:
        true_preference_list = pickle.load(f)
    with open('./est_preference_list.pkl', 'rb') as f:
        est_preference_list = pickle.load(f)
    with open('./delta_list.pkl', 'rb') as f:
        delta_list = pickle.load(f)
    with open('./threshold_list.pkl', 'rb') as f:
        threshold_list = pickle.load(f)
    with open('./num_query_list.pkl', 'rb') as f:
        num_query_list = pickle.load(f)
    with open('./performance_gap_list.pkl', 'rb') as f:
        performance_gap_list = pickle.load(f)

    sign_align_table = np.zeros([8, 3])
    for k in range(3, 11, 1):
        counts = np.apply_along_axis(lambda x: np.sum(x>0), 1, true_preference_list[k-3]*est_preference_list[k-3])
        # print(counts)
        sign_align_table[k-3, 0] = np.average(counts)
        sign_align_table[k-3, 1] = np.min(counts)
        sign_align_table[k-3, 2] = np.max(counts)

    print(sign_align_table)
    # Preference estimation
    for k in range(3, 4, 1):  # Can adjust the range based on your needs
        # np.apply_along_axis(lambda x: np.all(x > 0), 0,
        fig, axes = plt.subplots(nrows=1, ncols=2)
        im = axes.flat[0].imshow(true_preference_list[k-3])
        axes.flat[0].set_title("True Preference")
        axes.flat[0].set_xlabel("K")
        axes.flat[0].set_ylabel("Trials")
        im = axes.flat[1].imshow(est_preference_list[k - 3])
        axes.flat[1].set_title("Estimated Preference")
        axes.flat[1].set_xlabel("K")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig("./preferece_compare.png", dpi=300)
        plt.show()
    
    # Number of queries
    plt.plot([k for k in range(3, 11, 1)], list(map(np.average, num_query_list)),
             "k--", label="Experiments")
    plt.fill_between([k for k in range(3, 11, 1)],
                     list(map(np.min, num_query_list)),
                     list(map(np.max, num_query_list)))
    plt.plot([k for k in range(3, 11, 1)], [k * np.log(k / 0.01) for k in range(3, 11, 1)],
             color="red", label = "Theory", linestyle="dashed")
    plt.xlabel("K")
    plt.ylabel("Number of Queries")
    plt.legend()
    plt.savefig("./number_of_queries.png", dpi=300)
    plt.show()
    
    # Delta
    plt.plot([k for k in range(3, 11, 1)], list(map(np.average, delta_list)),
             "k--", label="Experiments")
    plt.fill_between([k for k in range(3, 11, 1)],
                     list(map(np.min, delta_list)),
                     list(map(np.max, delta_list)))
    plt.ylim([0.0, 0.35])
    plt.xlabel("K")
    plt.ylabel(r"$\nu^{*} - \omega^{*T}V^{\hat{\pi}}$")
    plt.legend()
    plt.savefig("./delta.png", dpi=300)
    plt.show()
    
    #Relative Performance
    plt.plot([k for k in range(3, 11, 1)], list(map(np.average, performance_gap_list)),
             "k--", label="Experiments")
    plt.fill_between([k for k in range(3, 11, 1)],
                     list(map(np.min, performance_gap_list)),
                     list(map(np.max, performance_gap_list)))
    plt.xlabel("K")
    plt.ylabel(r"$\frac{\nu^{*} - \omega^{*T}V^{\hat{\pi}}}{\nu^{*}}$")
    plt.legend()
    plt.savefig("./relative_performance.png", dpi=300)
    plt.show()
    
    # Sanity Check
    args = {"gamma": 0.99, "noise": 0.1, "epsilon": 0.01, "K": 3}
    mdp = GridWorldMDP(args["noise"], args["gamma"], args["K"], sanity=True)
    K = 3
    
    # Step 0: Sanity check to make sure "do nothing" policy pi0 has value function 0
    Pi0 = np.zeros([mdp.numstates, 5])  # Pi0[state, :] is a probability vector
    Pi0[:, 4] = 1  # Deterministic action of doing nothing
    V0, n = mdp.policyEvaluation(args["epsilon"], Pi0)
    Pi0_drawing = np.zeros(mdp.numstates) + 4
    mdp.drawWorld(V0[:, 0], Pi0_drawing, f"C:/Users/hydep/robotics/policy_eval_0.png", savefig=True, p="Doing Nothing")
    
    #Step 1: Sanity check to make sure the estimation of MORL is correct
    preference_name = ["green", "blue", "yellow"]
    for k in range(mdp.K):
        preference = np.zeros(mdp.K)
        preference[k] = 1
        V, Pi, n = mdp.valueIterationSanity(args["epsilon"], k)
        V1, Pi1, n1 = mdp.valueIteration(args["epsilon"], preference)
        mdp.drawWorld(V, Pi, f"C:/Users/hydep/robotics/sanity{k}.png", savefig=True, p=preference_name[k], setting="Sanity")
        mdp.drawWorld(V1[:, k], Pi1, f"C:/Users/hydep/robotics/morl{k}.png", savefig=True, p=preference_name[k], setting="MORL")
