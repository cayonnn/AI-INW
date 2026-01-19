import pandas as pd

def compute_dd_avoided(df: pd.DataFrame):
    """
    Estimate drawdown avoided by Guardian blocks
    """
    if "potential_dd" not in df.columns:
        return pd.Series(0, index=df.index)
        
    blocked = df[df["action"].isin(["FORCE_HOLD", "EMERGENCY_FREEZE", "HARD_BLOCK", "BLOCK"])]
    if blocked.empty:
        return pd.Series(0, index=df.index)
        
    return blocked.groupby("timestamp")["potential_dd"].sum().cumsum()

def compute_freeze_cost(df: pd.DataFrame):
    """
    Opportunity cost while Guardian froze trading
    """
    if "missed_profit" not in df.columns:
        return pd.Series(0, index=df.index)

    frozen = df[df["action"].isin(["FORCE_HOLD", "EMERGENCY_FREEZE", "HARD_BLOCK", "BLOCK"])]
    if frozen.empty:
        return pd.Series(0, index=df.index)
        
    return frozen.groupby("timestamp")["missed_profit"].sum().cumsum()
