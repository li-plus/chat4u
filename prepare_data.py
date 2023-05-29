import sqlite3
import pandas as pd

# %% Prepare training data

# load chat history from the decrypted database
# NOTE: open plain_msg_x.db with https://sqliteviewer.app/ and find the chat you want to train on
DB_NAME = "plain_msg_5.db"
TABLE_NAME = "Chat_58cb09164344b36c29a74aa1f7c24205"
with sqlite3.connect(DB_NAME) as conn:
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

# sort by timestamp
df = df.sort_values(by="msgCreateTime")

# filter out non-text messages
df = df[df["messageType"] == 1]

# drop unused columns
df = df[["msgCreateTime", "msgContent", "mesDes"]]

# %%

# merge adjacent sentences
merged_history = [df.iloc[0].to_dict()]
for row in df.to_dict(orient="records")[1:]:
    prev_data = merged_history[-1]
    if (
        row["mesDes"] == prev_data["mesDes"]
        and row["msgCreateTime"] - prev_data["msgCreateTime"] < 5 * 60
    ):
        prev_data["msgContent"] += "\n" + row["msgContent"]
    else:
        merged_history.append(row)

# single-round data
single_round_data = []
for i in range(1, len(merged_history)):
    prev_row, row = merged_history[i - 1], merged_history[i]
    if row["mesDes"] == 0 and prev_row["mesDes"] == 1:
        single_round_data.append(
            dict(instruction=prev_row["msgContent"], output=row["msgContent"])
        )


# %%

train_df = pd.DataFrame(single_round_data)

# filter out long input
MAX_PROMPT_LENGTH = 64
train_df = train_df[train_df["instruction"].str.len() <= MAX_PROMPT_LENGTH]

train_df.to_json("train.json", orient="records", force_ascii=False)
