from typing import List, Tuple, Dict, Any
from ..data.field import Field


class Example:
    """Defines a single example or sample

    Attributes:
        Each column of the example is an attribute.
    """

    @classmethod
    def fromlist(cls, data: List[str], fields: Dict[str, Field]) -> "Example":
        """Create an example from a list of attrbutes.

        Args:
            data: List of attribute values of that example
            fields: List of Field objects correspond to each
                attribute.
        Returns:
            The Example object
        """
        ex = cls()
        if len(data) != len(fields):
            raise ValueError(
                "The number of attributes doesn't match the number of fields. "
                f"{len(fields)} fields but the data has {len(data)} columns"
            )
        for (name, field), val in zip(fields.items(), data):
            setattr(ex, name, val)
        return ex

    @classmethod
    def fromdict(
        cls, data: Dict[str, Any], fields: Dict[str, Field]
    ) -> "Example":
        """Create an example from a dict of attributes.

        Args:
            data: Dict of attribute name to values of that example.
            fields: Dict of attirbute name to Field object.
        Returns:
            The Example object
        """
        ex = cls()
        for name, field in fields.items():
            if name not in data:
                raise ValueError(
                    f"Specified key {name} was not found in the input data."
                )
            setattr(ex, name, data[name])
        return ex

    def preprocess(self, fields: Dict[str, Field]):
        """Preprocess the fields of this example.

        Args:
            fields: The dict containing the mapping of attributes to
                Field object

        Returns:
            The Example object containing the processed fields.
        """
        for name, field in fields.items():
            setattr(self, name, field.preprocess(getattr(self, name)))
        return self

    def __eq__(self, obj):
        return self.__dict__ == obj.__dict__
