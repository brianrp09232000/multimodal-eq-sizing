class OfflineRLDataset(torch.utils.data.Dataset):
    def __init__(self, rl_df):
        state_cols = [c for c in rl_df.columns if c.startswith("state_")]
        next_state_cols = [c for c in rl_df.columns if c.startswith("next_state_")]

        self.states = torch.from_numpy(rl_df[state_cols].values).float()
        self.actions = torch.from_numpy(rl_df["action"].values).float().unsqueeze(-1)
        self.rewards = torch.from_numpy(rl_df["reward"].values).float().unsqueeze(-1)
        self.next_states = torch.from_numpy(rl_df[next_state_cols].values).float()
        self.dones = torch.from_numpy(rl_df["done"].values.astype("float32")).unsqueeze(-1)

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
