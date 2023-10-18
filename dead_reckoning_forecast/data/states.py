import pandas as pd
import numpy as np
import os
from .. import constants

def add_weights(df):
    df["weight"] = 1
    movement_disrupted = ~(
        (
            df["last_movement_x"].eq(df["delta_position_x"]) & 
            df["last_movement_y"].eq(df["delta_position_y"])
        ) |
        (
            np.isclose(df["last_movement_x"], df["delta_position_x"]) & 
            np.isclose(df["last_movement_y"], df["delta_position_y"])
        )
    )
    df.loc[movement_disrupted, "weight"] += 1
    movement_changed = df[constants.deltas].diff(axis=0).sum(axis=1).ne(0)
    df.loc[movement_changed, "weight"] += 1
    begin_dash = (df[constants.dashes].diff(axis=0).sum(axis=1) > 0) & (df[constants.dashes].sum(axis=1) > 0)
    df.loc[begin_dash, "weight"] += 1
    end_dash = (df["dash"].diff().diff() > 0) & (df["dash"].eq(0) | np.isclose(df["dash"], 0))
    df.loc[end_dash, "weight"] += 1
    hurt = df["health"].diff() < 0
    df.loc[hurt, "weight"] += 1
    return df

def read_states(match_dir, player, match_type):
    path = os.path.join(match_dir, str(player), "states.csv")
    df = pd.read_csv(path, index_col=0)[1:]#.copy()
    df[np.isclose(df, 0)] = 0
    dash_enabled = "dash" in match_type
    df["dash_enabled"] = 1 if dash_enabled else 0
    if not dash_enabled:
        if "dash_cooldown" in df.columns:
            df["dash_cooldown"] = -1
        if "dash" in df.columns:
            df["dash"] = -1
    if "health" not in df:
        df["health"] = 100
    df = add_weights(df)
    #df = normalize(df, dash_enabled)
    return df

def combine_player_states(self_states, enemy_states, match_type, self_attrs=constants.self_attrs, normalizer=None):
    dash_enabled = "dash" in match_type

    states = self_states[self_attrs].copy()
    states["enemy_relative_position_x"] = enemy_states["position_x"] - self_states["position_x"]
    states["enemy_relative_position_y"] = enemy_states["position_y"] - self_states["position_y"]
    states["enemy_health"] = enemy_states["health"]
    
    #if normalizer is None:
    #    normalizer = Normalizer(dash_enabled=dash_enabled)
    #    normalizer.fit(states)
    #states = normalizer.transform(states)
    #states = remove_leading_trailing_zeros(states)
    return states
