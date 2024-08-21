from cat.mad_hatter.decorators import plugin
from pydantic import BaseModel

# from: https://github.com/nickprock/ccat_semantic_chunking/blob/main/semantic_chunking.py
class MySettings(BaseModel):
    """
    breakpoint_threshold_type must be one between ["percentile", "standard_deviation", "interquartile", "gradient"]
    breakpoint_threshold_amount:
    recommended values by langchain
        "percentile": 95,
        "standard_deviation": 3,
        "interquartile": 1.5,

    """
    breakpoint_threshold_type: str = "percentile"
    breakpoint_threshold_amount: float = 95
    answer_language: str = "English"


@plugin
def settings_model():
    return MySettings