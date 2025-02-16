from typing import Any

import numpy as np
from backend.custom_types import AlgorithmType, ModelName

class NoModelFoundException(Exception):
    """
    Exception raised when no model is available.
    """

    pass



class Algorithm:
    """
    A base class for all algorithms that can be applied to frames in the form of numpy arrays.

    Attributes:
        type (AlgorithmType): An enum indicating the specific type of algorithm implemented by the subclass.
        name (ModelName): An enum indicating the specific name of the model used by the subclass.
    """

    type: AlgorithmType
    name: ModelName

    def __init__(self) -> None:
        """
        Initialization mehtod for an Algorithm instance.
        This method should be overridden by subclasses to implement specific algorithm functionality.
        """

        pass

    def __call__(self, frame: np.ndarray) -> Any:
        """
        Processes a frame and returns the result.
        This method should be overridden by subclasses to implement specific algorithm functionality.

        Args:
            frame (ndarray): The input frame to process as a numpy array.

        Returns:
            The relevant result of the algorithm.
        """

        pass

    def __del__(self) -> None:
        """
        Destructor method for the class.

        This method is called when the instance is about to be destroyed. It can be used to perform
        any cleanup operations such as closing files, releasing resources, or other similar tasks.
        """

        pass

    def __str__(self) -> str:
        """
        Returns a string representation of the algorithm.

        Returns:
            str: A string representation of the algorithm.
        """

        return self.name.split(".")[-1]



class AlgorithmManager:
    """
    A manager class for handling the lifecycle and retrieval of algorithm instances.
    This manager acts as a registry that associates `AlgorithmType` with their respective `Algorithm` instances.

    Attributes:
        _algo (dict[AlgorithmType, Algorithm]): Private dictionary to hold algorithm types and their instances.
    """

    _algo: dict[ModelName, Algorithm] = {}

    def __init__(self, algorithms: list[Algorithm] = []) -> None:
        """
        Initializes the AlgorithmManager with optional pre-defined algorithms.

        Args:
            algorithms (List[Algorithm], optional): A list of algorithm instances to be managed.
        """

        for algo in algorithms:
            self._algo[algo.name] = algo

    def add_algorithm(self, algorithm: Algorithm) -> None:
        """
        Adds an algorithm instance to the manager.
        If a algorithm with the same type already exists, it will be replaced by the new one.

        Args:
            algorithm (Algorithm): The algorithm instance to be added. Its type will be used as a key.
        """

        self._algo[algorithm.name] = algorithm

    def remove_algorithm_by_name(self, name: ModelName) -> None:
        """
        Removes an algorithm instance from the manager by its type.

        Args:
            type (AlgorithmType): The type of the algorithm to be removed from the manager.
        """

        if name in self._algo:
            self._algo.pop(name)


    def get_algorithm_by_name(self, name: ModelName) -> Algorithm:
        """
        Retrieves an algorithm instance by its type.

        Args:
            type (ModelName): The name of the algorithm to retrieve.

        Returns:
            Algorithm: The algorithm instance associated with the given name.

        Raises:
            TypeError: If the name is not recognized or no algorithm is associated with it.
        """

        if name in self._algo:
            return self._algo[name]
        return TypeError(f"{name} isn't a valid item of this Manager!")

