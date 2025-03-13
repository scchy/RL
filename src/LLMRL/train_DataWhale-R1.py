# python3
# Reference Github: https://github.com/datawhalechina/unlock-deepseek 
# Reference Article: DeepSeek R1 Zero中文复现教程来了！ https://mp.weixin.qq.com/s/Z7P61IV3n4XYeC0Et_fvwg
#                    DeepSeek GRPO Trainer简明教程 http://www.hubwiz.com/blog/deepseek-grpo-trainer-concise-tutorial/
#                    Deepseek R1 Zero成功复现, 三阶段RL   https://zhuanlan.zhihu.com/p/21290410831
#                    DeepSeek R1技术报告关键解析 https://juejin.cn/post/7467830784449937434
#                    大模型KV Cache节省神器MLA学习笔记 https://zhuanlan.zhihu.com/p/703862723?utm_psn=1786086877884870656
#                    蒙特卡洛采样之拒绝采样（Reject Sampling） http://www.twistedwg.com/2018/05/30/MC-reject-sampling.html
# Reference Model: https://www.modelscope.cn/models/qwen/Qwen2.5-3B-Instruct
# 复现 rl stage2部分
# Create Date: 2025-02-07
# ======================================================================================================
import logging
import os 
import random
import re 
from dataclasses import dataclass
from datetime import datetime 
from typing import List, AnyStr, Dict

from datasets import load_dataset
# swanlab==0.4.6
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
# trl==0.14.0
from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser


_deepseek_rl_4stage = """ 
paper: https://arxiv.org/pdf/2501.12948
rule-based reward system
 
1. 冷启动 SFT: 推理数据 (高质量思维链数据)   一般认为 Qwen2.5-3B-Instruct 已经做过该阶段
    目的：增强模型的推理能力，解决强化学习冷启动问题
2. 以推理为导向的强化学习  GRPO(Group Relative Policy Optimization): 推理数据 -> 数学、编程、逻辑推理等明确有标准答案的任务
    目的：提高模型在推理任务上的准确性和能力
3. 拒绝采样与再监督学习 STF: 
    推理数据(-> 拒绝采样方法，收集高质量的推理任务数据) + 非推理数据（写作、问答、翻译、自我认知等）
        - 拒绝采样对于概率分布函数难以求解的数据进行采样是有效的，现在计算机的计算能力如此发达的情况下，更是有利于蒙特卡罗采样的发展
    目的：提升推理任务 + 扩展在非推理任务上的能力，使其能够有效处理这些任务
4. 全面场景的强化学习 GRPO: 推理数据 + 非推理数据(Reward Model)
    目的：提升推理任务 + 非推理任务-对齐人类偏好, 提高在非推理任务上的 有用性 和 安全性

对于ollama等一般做 2-4的三阶段的训练
"""


_vllm_doc_ = """ 
vllm 推理优化 from vllm import LLM, SamplingParams
由加州大学伯克利分校团队开发的高性能LLM推理和服务框架，通过多项创新技术显著提升大模型推理速度（最高可达传统方案的24倍），
尤其在高并发、长文本生成场景下表现突出。以下是其核心加速原理的深度解析：

通过PageAttention内存管理+连续批处理+硬件协同优化的三重创新，vLLM实现了LLM推理领域的革命性加速，已成为AI大模型部署的行业标准工具。

- 核心技术：PageAttention（分页注意力机制）
    - KV Cache（键值缓存）进行优化
        - 一般KV Cache:  显存浪费严重, 无法动态扩展
        - PageAttention的解决方案： 分块存储、按需分配、逻辑连续，物理离散
- 连续批处理
- 内存与计算优化
    - 共享内存池（Memory Pool）
    - 零冗余权重加载
    - 计算内核优化： 使用FlashAttention-2加速注意力计算，降低显存带宽需求
- 硬件协同加速
- 当前限制
    - 对非Transformer架构模型支持有限（如RNN-based模型）
    - 需要GPU显存≥24GB（FP16加载70B模型）
"""

_MLA_doc = """ 

DeepSeek2针对KV Cache使用了6Bit量化，节省比例为 6 / 16
这个修补过的DeepseekV2Model包含了对DeepseekV2Attention的以下修改，以减少VRAM消耗并提高效率：

1. 不再缓存解压缩的Key/Value状态，而仅缓存低秩Key-Value联合压缩以及Key的解耦RoPE部分。 
    为了重用transformers库的缓存实用程序，我们将k_pe视为key_states，将compressed_kv视为value_states。
2. 采用DeepseekV2论文中描述的吸收技术，通过改变计算Query和Output向量时的乘法顺序。
    这不仅节省了中间张量的内存消耗，还减少了浮点运算的次数。
3. 分别计算RoPE部分和非RoPE部分的注意力分数，然后将它们相加。
    原始实现将Query/Key向量的两部分连接起来，但在缓存压缩Key/Value状态时由于不必要的数据广播和内存往返而被证明效率较低。

"""

_grpo_doc = """ 
GRPO: on policy  和 PPO的唯一差异是adv的计算
1. 生成完成
    prompts -> completions -> rewards 
2. 计算优势: adv = (r - mean)/std 
    组相对策略优化 (GRPO)。
3. KL + loss
    - D_kl(\pi||\pi_ref) = \pi_ref / \pi - log(\pi_ref / \pi) - 1  # Schulman 等人 (2020) 引入的近似器来估计
    - loss = mean(\pi(o|q)/\pi_freeze(o|q) adv - \beta * D_kl(\pi||\pi_ref))


GRPOTrainer.compute_loss
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    per_token_loss = -(per_token_loss - self.beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()


self.ref_model = create_reference_model(model)

"""


_result_think = """ 
其他社区多次报告，小于 3B 的模型无法学会推理，经过我们的测试，确实！


1. 模型前期学习输出格式的速度很快，大概 20 到 30 步就能学得很好。
但是后来由于我们的思考长度奖励函数，模型的输出长度被拉长，发生严重的重复现象，导致超出 4096 的输出被截断，格式不完整，格式奖励函数的奖励值就大幅下降，
后面模型又开始缩短输出，稳定在 300 到 400，又恢复到正确格式。

2. 模型被鼓励拉长输出的时候，计算正确率也在提升，所以我们有个不严谨的判断，似乎拉长模型输出，能带一定的计算正确率的提升。
观察下图可以发现，在 120 步时，模型的输出在越变越长，平均输出长度已经被拉到 400 左右，
越来越多的输出已经超过 1000，方程计算正确率也在逐步升高，但是这时已经发生一些重复问题导致格式错误

3. GRPO 已经意识到重复问题带来的奖励值下降，它在 200 步左右开始逐步限制模型输出长度，而这时模型的计算正确率也保持在 0.3 到 0.4 左右

4. 在训练初期，你会看到比较明显的方程奖励提升，而输出长度不断减小。模型似乎有一种趋向于缩短思考长度的趋势，所以我们引入思考长度奖励函数来对抗这种趋势，
我们把它解释为模型计算能力提升之后，就像学霸一眼秒杀题目一样，模型不想输出更多“废话”来解释解题过程

- Qwen 2.5 很喜欢反复试错、验算，反复试错很容易导致上文提及的重复输出问题

"""


@dataclass
class DatasetArguments:
    """数据集参数的数据类"""
    dataset_id_or_path: AnyStr = "Jiayi-Pan/Countdown-Tasks-3to4"
    dataset_split: AnyStr = "train"
    tokenizer_name_or_path: AnyStr = None 


@dataclass
class SwanlabArguments:
    swanlab: bool 
    # SwanLab 用户名
    workspace: AnyStr
    # SwanLab 的项目名
    project: AnyStr
    # SwanLab 的实验名
    experiment_name: AnyStr


# log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)  # 设置日志格式
logger.addHandler(handler)


def format_reward_func(completions: List[AnyStr], **kwargs) -> List[float]:
    """ 
    格式奖励函数，检查模型输出格式是否匹配：<think>...</think><answer>...</answer>
    
    args:
        completions: 生成的输出
    return:
        奖励分数
    """
    rewards = []
    # 遍历生成的输出
    for cmp in completions:
        try:
            cmp = f"<think>{cmp}"
            if random.random() < 0.1: # 1% 的概率生成输出写入文件
                # 创建生成输出目录
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(completion)  # 写入生成的输出

            
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
            match = re.search(regex, cmp, re.DOTALL)  # 使用正则表达式进行匹配
            
            if match is None or len(match.groups()) !=2:
                rewards.append(0.0)  # 如果格式不正确，奖励为 0
            else:
                rewards.append(1.0)  # 如果格式正确，奖励为 1
            
        except Exception:
            rewards.append(0.0)
    return rewards


def equation_reward_func(completions: List[AnyStr], target: List[AnyStr], nums: List[AnyStr], **kwargs) -> List[float]:
    """
    方程奖励函数，检查计算结果是否正确，数字是否符合使用要求（每个数字只用一次，只使用所提供的数字）
    
    args:
        completions: 生成的输出
        target: 预期的答案
        nums: 可用的数字
    return:
        奖励分数
    """
    # 初始化奖励列表
    rewards = []
    # 遍历生成的输出、预期的答案和可用的数字
    for cmp, gt, numbers in zip(completions, target, nums):
        try:
            # 在生成的输出前添加 <think> 标签，便于后续正则表达式匹配
            cmp = f"<think>{cmp}"
            # 定义正则表达式模式，用于匹配 <answer> 标签
            match = re.search(r"<answer>(.*?)<\/answer>", cmp)
            if match is None:
                rewards.append(0.0)  # 如果没有匹配到 <answer> 标签，奖励为 0
                continue
            equation = match.group(1).strip()  # 提取 <answer> 标签中的内容
            # 提取方程中的所有数字
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

            # 检查所有数字是否被使用且只使用一次
            if sorted(used_numbers) != sorted(numbers):
                rewards.append(0.0)
                continue
            
            # 定义允许的字符模式，只允许数字、运算符、括号和空白字符
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation):
                rewards.append(0.0)   
                continue
            
            # 计算方程的结果
            res = eval(equation, {"__builtins__": None}, {})
            # 检查结果
            if abs(float(res) - float(gt)) < 1e-5:
                rewards.append(1.0)
                # 10% 的概率将成功的样本写入文件
                if random.random() < 0.10:
                    # 创建生成输出目录（如果不存在）
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join(
                        "completion_samples", "success_completion_samples.txt"
                    )
                    with open(log_file, "a") as f:
                        f.write(f"\n\n==============\n")
                        f.write(completion)  # 写入生成的输出
            else:
                rewards.append(0.0)  # 如果不正确，奖励为 0
        except Exception:
            rewards.append(0.0)  # 如果评估失败，奖励为 0

    return rewards


def thought_len_reward_func(completions: List[AnyStr], **kwargs) -> List[float]:
    """
    思考长度奖励函数，检查<think>标签的长度是否大于1000
    
    args:
        completions: 生成的输出
    return:
        奖励分数
    """
    rewards = []
    # 遍历生成的输出
    for cmp in completions:
        try:
            cmp = f"<think>{cmp}"
            match = re.search(r"<think>(.*?)</think>", cmp)
            if match:
                thought_process = match.group(1).strip()
                if len(thought_process) > 1000:
                    rewards.append(1.0)  # 如果思考过程长度大于 1000，奖励为 1
                else:
                    rewards.append(0.0)  # 否则奖励为 0
            else:
                rewards.append(0.0)  # 如果没有匹配到 <think> 标签，奖励为 0
                continue
        except Exception:
            rewards.append(0.0)
    return rewards

    
def get_checkpoint(training_args: GRPOConfig):
    """ 
    获取最后一个检查点
    args:
        training_args: 训练参数
    return:
        str: 最后一个检查点的路径，如果没有检查点，则返回 None
    """
    last_ck = None 
    if os.path.isdir(training_args.output_dir):  # 如果输出目录存在
        # 获取最后一个检查点
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


# GRPO 训练函数
def grpo_function(
    model_args: ModelConfig,
    dataset_args: DatasetArguments,
    training_args: GRPOConfig,
    callbacks: List,
):
    # 记录模型参数
    logger.info(f"Model parameters {model_args}")
    # 记录训练/评估参数
    logger.info(f"Training/evaluation parameters {training_args}")
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        dataset_args.tokenizer_name_or_path if dataset_args.tokenizer_name_or_path else model_args.model_name_or_path,
        revision=model_args.model_revision, # 指定模型版本
        trust_remote_code=model_args.trust_remote_code, # 允许使用远程代码  
    )
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token 
    
    # 加载数据集
    dataset = load_dataset(
        dataset_args.dataset_id_or_path, split=dataset_args.dataset_split
    )
    # 随机选择 50K 个样本，看你喜好定数字，但是数据集有 409K 个样本
    dataset = dataset.shuffle(seed=training_args.seed).select(range(50000))
    
    def generate_rl_prompt(numbers: List[int], target: int) -> Dict:
        """
        生成R1 Countdown 游戏提示词
        
        args:
            numbers: 数字列表 
            target: 目标值
        return:
            dict: 生成的一个数据样本
        """
        r1_prefix = [
            {
                "role": "user",
                "content": f"使用给定的数字 {numbers}，创建一个等于 {target} 的方程。你可以使用基本算术运算（+、-、*、/）一次或多次，但每个数字只能使用一次。在 <think> </think> 标签中展示你的思考过程，并在 <answer> </answer> 标签中返回最终方程，例如 <answer> (1 + 2) / 3 </answer>。在 <think> 标签中逐步思考。",
            },
            {
                "role": "assistant",
                "content": "让我们逐步解决这个问题。\n<think>",  # 结尾使用 `<think>` 促使模型开始思考
            },
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                r1_prefix, tokenizer=False, continue_final_message=True
            ),
            "target": target,
            "nums": numbers
        }
    
    # 将数据集转换为 R1 Countdown 游戏提示词
    dataset = dataset.map(lambda x: generate_rl_prompt(x['nums'], x['target']))
    # 将数据集拆分为训练和测试集，拆分比例为9:1
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split["test"]  # 获取测试集
    
    # 设置 GRPOTrainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        # 奖励函数列表
        reward_funcs = [
            format_reward_func, # 格式奖励函数
            equation_reward_func, # 方程奖励函数
            thought_len_reward_func, # 思考长度奖励函数
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=callbacks
    )
    
    last_check_point = get_checkpoint(training_args)
    # 断点恢复
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    
    # 训练模型
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # 记录和保存指标
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.model.config.use_cache = True 
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # 等待所有进程加载
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    logger.info("*** Training complete! ***")
    
    
def mian():
    # 去获取传入的 Datawhale-R1.yaml 里面的参数
    parser = TrlParser((ModelConfig, DatasetArguments, GRPOConfig, SwanlabArguments))
    model_args, dataset_args, training_args, swanlab_args = (
        parser.parse_args_and_config()
    )
    
    if swanlab_args.swanlab:
        swanlab_callback = SwanLabCallback(
            workspace=swanlab_args.workspace,
            project=swanlab_args.project,
            experiment_name=swanlab_args.experiment_name,
        )
        callbacks = [swanlab_callback]
    else:
        callbacks = None
    
    # run
    grpo_function(model_args, dataset_args, training_args, callbacks=callbacks)


if __name__ == "__main__":
    main()

