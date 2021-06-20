from distutils.command.config import config
import random

# Return the index of the chosen configuration
def choose_from_dist(dist, configs):
    counter = 0
    total = 0   # Sum of probabilities
    r = random.random()
    for p in dist:
        total += p
        if r > total:
            counter += 1
        else: 
            return counter
    return -1

# Return a random value from within a range
def setup_configs(lastdist, lastconfigs, configs):

    newdist = []
    newconfigs = []
    if not lastdist:
        for c in configs:
            entry = [c]
            newconfigs.append(entry)
            newdist.append(1/len(configs))
        return newdist, newconfigs
    else:
        for i in range(len(lastdist)):
            for c in configs:
                previous = []
                for l in lastconfigs[i]:
                    previous.append(l)
                previous.append(c)
                newconfigs.append(previous)
                newdist.append(lastdist[i] * 1/len(configs))
        return newdist, newconfigs



if __name__ == "__main__":
    dist = [0.2, 0.3, 0.1, 0.25, 0.15]
    configs = [8, 16, 32, 64, 128]
    for i in range(10):
        config = choose_from_dist(dist, configs)
        print(f"Sample {i+1}: {config}")
