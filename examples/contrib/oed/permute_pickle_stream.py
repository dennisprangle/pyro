import pickle
import torch

if __name__ == '__main__':
    l = []
    with open('./run_outputs/turk_simulation/0run_7.result_stream.pickle', 'rb') as results_file:
        try:
            while True:
                results = pickle.load(results_file)
                l.append(results)
        except EOFError:
            pass
    for i in range(5):
        perm = torch.randperm(8)
        with open(f'./run_outputs/turk_simulation/0permuterand{i}.result_stream.pickle', 'wb') as output:
            for j in perm:
                for k in range(10):
                    pickle.dump(l[10*j+k], output)
