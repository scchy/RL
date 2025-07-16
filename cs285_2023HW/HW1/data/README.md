
# 数据下载地址

- [expert_data 地址: hw1/cs285/expert_data](https://github.com/berkeleydeeprlcourse/homework_fall2023/tree/main/hw1/cs285/expert_data)
- [expert_policy 地址: hw1/cs285/policies/experts](https://github.com/berkeleydeeprlcourse/homework_fall2023/tree/main/hw1/cs285/policies/experts)


# shell 下载

expert_data
```shell
cd ./cs285_2023HW/HW1/data/
# expert_data
expert_data_p=hw1/cs285/expert_data
git clone --depth 1 --filter=blob:none --sparse https://github.com/berkeleydeeprlcourse/homework_fall2023.git
cd homework_fall2023
git sparse-checkout set ${expert_data_p}
# homework_fall2023/hw1/cs285/policies/experts/
cd ..
mv homework_fall2023/${expert_data_p}/*.pkl ./
rm -rf homework_fall2023

# RL项目恢复
cd ../../../
git config core.sparseCheckout 
git sparse-checkout disable 
```

expert_policy
```shell
cd ./cs285_2023HW/HW1/data/
# expert_policy
expert_policy_p=hw1/cs285/policies/experts 
git clone --depth 1 --filter=blob:none --sparse https://github.com/berkeleydeeprlcourse/homework_fall2023.git
cd homework_fall2023
git sparse-checkout set ${expert_policy_p}
# homework_fall2023/hw1/cs285/policies/experts/
cd ..
mv homework_fall2023/${expert_policy_p}/*.pkl ./
rm -rf homework_fall2023

# RL项目恢复
cd ../../../
git config core.sparseCheckout 
git sparse-checkout disable 
```
