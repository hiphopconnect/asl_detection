from enum import Enum


class AlgorithmType(str, Enum):
    """
    An enumeration to represent different types of vision algorithms.

    Attributes:
        DETECTION (str): Placeholder type for object detection
    """

    NONE = "NONE"
    DETECTION = "DETECTION"


class CameraType(str, Enum):
    """
    An enumeration to represent different types of cameras used.

    Attributes:
        RGB (str): Represents a standard Red-Green-Blue color camera.
        IR (str): Represents an Infrared camera.
    """

    RGB = "RGB"
    IR = "IR"
