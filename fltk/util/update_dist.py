import math


# Inputs: 1. dist, the distribution of configurations
# 2. configs, the possible configurations
# 3. chosen_configs, the configurations chosen by clients in one epoch
# 4. losses: loss sent by each client
# 5. size of the validation set of each client
def update_dist(dist, configs, chosen_configs, losses, V):
    new_dist = []
    for p in dist:
        new_dist.append(p)
    learning_rate = math.sqrt(2*math.log10(len(configs))) # Learning rate for the distribution
    factor = 0.005    # A multiplying factor to reduce the variance of grads
    grads = [0]*len(configs)

    # Calculate the sum of the length of the validation set
    sum_v = 0
    for v in V:
        sum_v += v

    # Calculate gradient
    for j in range(len(configs)):
        for i in range(len(chosen_configs)):
            if chosen_configs[i] == configs[j]:
                grads[j] += factor*losses[i]*V[i]/(dist[j]*sum_v)

    # Update distribution
    sum_p = 0
    for j in range(len(dist)):
        new_dist[j] *= math.exp(-learning_rate*grads[j])
        sum_p += new_dist[j]

    # Normalization
    for j in range(len(dist)):
        new_dist[j] = new_dist[j]/sum_p
    return new_dist


if __name__ == "__main__":
    dist = [0.2, 0.2, 0.2, 0.2, 0.2]
    configs = [10, 16, 32, 64, 128]
    chosen_configs = [10, 128, 64, 32, 16]
    losses = [600, 200, 350, 400, 500]
    V = [10, 10, 10, 10, 10]
    new_dist = update_dist(dist, configs, chosen_configs, losses, V)
    print(f"dist: {dist}")
    print(f"New dist: {new_dist}")
