import torchvision.transforms as transforms
from .. import constants
from ..util import max_vector_magnitude

def create_transformer(img_size=(224,224), mean_std=None, aug=False):
    augmentations = []
    if aug:
        augmentations = [
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),    
        ]
    normalizations = []
    if mean_std is not None:
        normalizations = [transforms.Normalize(*mean_std)]
    return transforms.Compose([
        transforms.Resize(img_size),
        *augmentations,
        transforms.ToTensor(),
        *normalizations
    ])

class Normalizer:
    def __init__(self, dash_enabled=None, delta_mag=0, enemy_mag=0, negative_dash=False):
        self.dash_enabled = dash_enabled
        self.dash_cooldown = 2
        self.dash_length = 10
        self.max_health = 100
        self.max_bullets = 5
        self.delta_mag = delta_mag
        self.enemy_mag = enemy_mag
        self.n = 0
        self.negative_dash = negative_dash
        
    def fit(self, df):
        if self.dash_enabled is None and "dash_enabled" in df.columns:
            dash_enabled = bool(df.iloc[-1]["dash_enabled"])
            
        if self.delta_mag is None:
            if "last_movement_x" in df.columns:
                delta_mag = max(max_vector_magnitude(df[constants.last_movements]), max_vector_magnitude(df[constants.deltas]))
            else:
                delta_mag = max_vector_magnitude(df[constants.deltas])
        if self.enemy_mag is None:
            if "enemy_relative_position_x" in df.columns:
                enemy_mag = max_vector_magnitude(df[constants.enemy_positions])

        n = len(df)

        self.dash_enabled = self.dash_enabled if not dash_enabled else True
        self.delta_mag = max(self.delta_mag, delta_mag)
        self.enemy_mag = max(self.enemy_mag, enemy_mag)
        self.n += n

            
    def get_dash_enabled(self, df, dash_enabled=None):
        if dash_enabled is None:
            dash_enabled = self.dash_enabled
        if dash_enabled is None:
            if "dash_enabled" in df.columns:
                dash_enabled = bool(df.iloc[-1]["dash_enabled"])
            else:
                dash_enabled = True
        return dash_enabled
        
    def transform(self, df, dash_enabled=None):
        dash_enabled = self.get_dash_enabled(df, dash_enabled=dash_enabled)
        
        df = df.copy()

        if dash_enabled or not self.negative_dash:
            df["dash_cooldown"] /= self.dash_cooldown
            df["dash"] /= self.dash_length

        healths = [x for x in df.columns if "health" in x]
        df[healths] /= self.max_health
        df["bullets"] /= self.max_bullets

        if "last_movement_x" in df.columns:
            df[constants.deltas+constants.last_movements] /= self.delta_mag
        else:
            df[constants.deltas] /= self.delta_mag

        if "enemy_relative_position_x" in df.columns:
            df[constants.enemy_positions] /= self.enemy_mag
        return df
    
    def inverse_transform(self, df):
        dash_enabled = self.get_dash_enabled(df, dash_enabled=dash_enabled)
        
        df = df.copy()

        if dash_enabled or not self.negative_dash:
            if "dash_cooldown" in df.columns:
                df["dash_cooldown"] *= self.dash_cooldown
            if "dash" in df.columns:
                df["dash"] *= self.dash_length

        if "health" in df.columns:
            healths = [x for x in df.columns if "health" in x]
            df[healths] *= self.max_health
        if "bullets" in df.columns:
            df["bullets"] *= self.max_bullets

        if "last_movement_x" in df.columns:
            df[constants.deltas+constants.last_movements] *= self.delta_mag
        else:
            df[constants.deltas] *= self.delta_mag

        if "enemy_relative_position_x" in df.columns:
            df[constants.enemy_positions] *= self.enemy_mag
        return df