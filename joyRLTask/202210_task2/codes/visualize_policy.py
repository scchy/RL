import emoji
import numpy as np
import pandas as pd

ACTION_TO_EMOJI = {0: "👈", 
    1: "👇", 
    2: "👉", 
    3: "👆"}

MAPS = {
    "theAlley": [
        "S...H...H...G"
    ],
    "walkInThePark": [
        "S.......",
        ".....H..",
        "........",
        "......H.",
        "........",
        "...H...G"
    ],
    "1Dtest": [

    ],
    "4x4": [
        "S...",
        ".H.H",
        "...H",
        "H..G"
    ],
    "8x8": [
        "S.......",
        "........",
        "...H....",
        ".....H..",
        "...H....",
        ".HH...H.",
        ".H..H.H.",
        "...H...G"
    ],
}

def visualize_policy(policy_list, map, policy_file="./policy.txt"):
    row = len(map)
    col = len(map[0])

    # with open(policy_file, "w") as f:
    #     for i in range(row):
    #         for j in range(col):
    #             p = policy_list[i * col + j]
    #             f.write(emoji.emojize(ACTION_TO_EMOJI[p]))
    #         f.write("\n")

    fig = np.zeros((row, col), dtype=np.unicode_)
    for i in range(row):
        for j in range(col):
            p = policy_list[i * col + j]
            fig[i, j] = emoji.emojize(ACTION_TO_EMOJI[p])
    fig_df = pd.DataFrame(fig)
    fig_df.to_csv(policy_file, header=False, index=False, encoding="utf-16")

if __name__ == "__main__":
    policy = [2, 2, 1, 0, 1, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 3, 0, 0,
    2, 1, 1, 1, 1, 1, 0, 2, 3, 2, 2, 2, 1, 0, 2, 1, 2, 3, 3, 3, 2, 2, 2, 0]
    policy = [1,1,2,2,1,0,0,0,2,2,2,1,0,1,3,0,3,3,2,2,1,1,1,0,2,2,1,2,1,1,1,1,2,3,3,3,2,2,2,1,0,0,3,3,3,3,2,0]

    visualize_policy(policy_list=policy, map=MAPS["walkInThePark"], 
        policy_file="./policy1.csv")

    policy2 = list("222113003231021121211001232210013233211102033220")
    policy2 = [int(i) for i in policy2]
    visualize_policy(policy_list=policy2, map=MAPS["walkInThePark"], 
        policy_file="./policy2.csv")
