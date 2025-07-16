


# export data 
# export_policy 
for itr in range(params['n_iter']):
    for _ in range(params['num_agent_train_steps_per_iter']):
        # TODO: sample some data from replay_buffer
        # HINT1: how much data = params['train_batch_size']
        # HINT2: use np.random.permutation to sample random indices
        # HINT3: return corresponding data points from each array (i.e., not different indices from each array)
        # for imitation learning, we only need observations and actions.  
        ob_batch, ac_batch = TODO

        # use the sampled data to train an agent
        train_log = actor.update(ob_batch, ac_batch)
        # action mseloss 
        


        








