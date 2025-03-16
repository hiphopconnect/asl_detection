from enum import Enum


class CameraType(str, Enum):
    """
    An enumeration to represent different types of cameras used.
    Attributes:
        RGB (str): Represents a standard Red-Green-Blue color camera.
        IR (str): Represents an Infrared camera.
    """

    RGB = "RGB"


class ModelName(str, Enum):
    """
    An enumeration to represent different names of models used.

    Attributes:
        MediaPipeHolistics (str): Represents a MediaPipe Holistics model.
    """

    ASLWORDDETECTION = "ASLWORDDETECTION"
    ASLFINGERSPELLING = "ASLFINGERSPELLING"
    NONE = "NONE"
