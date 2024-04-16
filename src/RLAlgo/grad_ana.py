# python3
# Create Date: 2024-04-12
# Author: Scc_hy
# Func: 梯度分析
# ===========================================================================================

import torch 
import numpy as np
import matplotlib.pyplot as plt


class gradCollecter:
    def __init__(self):
        self.collected_grad = []
    
    def collect_grad(self, params_tensor):
        norms = []
        for p in params_tensor:
            if p.grad is not None:
                norms.append(torch.linalg.vector_norm(p.grad.data.cpu().detach()))

        self.collected_grad.append(
            torch.linalg.vector_norm(torch.stack(norms)).numpy()
        )
    
    def __call__(self, params_tensor):
        self.collect_grad(params_tensor)
        return 
    
    def dump(self, file_name):
        total = np.stack(self.collected_grad)
        with open(file_name, 'wb') as f:
            np.save(f,  total)

    def describe(self, plot_flag=False):
        total = np.stack(self.collected_grad)
        pct_list = [25, 50, 75, 95, 99]
        res = np.percentile(total, np.array(pct_list))
        res_dict = dict(zip([f'P{p}' for p in pct_list], res))
        if plot_flag:
            plt.hist(total)
            plt.title(f'total norm distribution\n{str(res_dict)}')
            plt.show()
        return res_dict



        

