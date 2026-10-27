import functools
import logging
import pandas as pd
from typing import Callable, Any
from .self_healing_df import SmartBIFrame

logger = logging.getLogger(__name__)

def SchemaGuard(func: Callable) -> Callable:
    """
    Decorator for BI functions: Auto-wrap df in SelfHealingDataFrame, catch errors, self-heal, retry.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Find df arg (assume first positional or 'df' kwarg)
        df = None
        if args:
            df = args[0] if isinstance(args[0], (pd.DataFrame, SmartBIFrame)) else None
        if not df and 'df' in kwargs:
            df = kwargs['df']
        
        if df is not None and not isinstance(df, SmartBIFrame):
            df = SmartBIFrame(df)
            # Update args/kwargs if df found
            if args and isinstance(args[0], pd.DataFrame):
                args = (df,) + args[1:]
            if 'df' in kwargs:
                kwargs['df'] = df
        
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return result
            except (KeyError, AttributeError) as e:
                if attempt == max_retries:
                    raise
                missing = str(e).split("'")[1] if "'" in str(e) else None
                if missing and df is not None:
                    logger.info(f"SchemaGuard caught {type(e).__name__}: '{missing}', attempting heal...")
                    try:
                        match, score = df.find_closest_attribute(missing)
                        if match:
                            logger.info(f"SchemaGuard healed: {match} -> {missing} ({score:.2%})")
                            # Re-run
                            continue
                    except Exception as heal_e:
                        logger.warning(f"Heal failed: {heal_e}")
                raise
        return None
    return wrapper

