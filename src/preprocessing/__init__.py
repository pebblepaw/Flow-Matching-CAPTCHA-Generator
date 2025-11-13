from enum import Enum
from .hairline_removal import inpainting_remove_black_hairline, color_voting_remove_black_hairline, color_voting_inpaint, color_voting_inpaint_dilate, color_voting_brighten_inpaint_dilate

class PreprocessingMethod(Enum):
    DEFAULT = "default_color_voting_inpaint_dilate"
    COLOR_VOTING = "color_voting"
    INPAINTING = "inpainting"
    COLOR_VOTING_INPAINT = "color_voting_inpaint"
    COLOR_VOTING_INPAINT_DILATE = "color_voting_inpaint_dilate"
    COLOR_VOTING_BRIGHTEN_INPAINT_DILATE = "color_voting_brighten_inpaint_dilate"

    @property
    def function(self):
        return {
            PreprocessingMethod.DEFAULT: color_voting_inpaint_dilate,
            PreprocessingMethod.COLOR_VOTING: color_voting_remove_black_hairline,
            PreprocessingMethod.INPAINTING: inpainting_remove_black_hairline,
            PreprocessingMethod.COLOR_VOTING_INPAINT: color_voting_inpaint,
            PreprocessingMethod.COLOR_VOTING_INPAINT_DILATE: color_voting_inpaint_dilate,
            PreprocessingMethod.COLOR_VOTING_BRIGHTEN_INPAINT_DILATE: color_voting_brighten_inpaint_dilate
        }[self]

__all__ = ["PreprocessingMethod"]
