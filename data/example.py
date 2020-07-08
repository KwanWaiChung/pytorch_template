from typing import List, Tuple, Dict, Any
from ..data.field import Field


class Example:
    """Defines a single example/ data row.

    Attributes:
        Each feature of the example is an attribute.
    """

    @classmethod
    def fromlist(cls, data: List[str], fields: Dict[str, Field]) -> "Example":
        """Create an example from a list of attrbutes.

        Args:
            data: List of attribute values of that example.
            fields: List of `Field` objects correspond to each
                attribute.

        Returns:
            The Example object.

        Raises:
            ValueError: If len(`data`) != len(`fields`).

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
            fields: Dict of attirbute name to `Field` object.

        Returns:
            The Example object.

        Raises:
            ValueError: If a key in `data` not in `fields`.

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
        """Preprocess the attributes in this example. This will be
            ran by the constructor of a `Dataset` object. Refer to the
            `preprocess` method of `Field` for more details on what it does to
            the attribute.

        Args:
            fields: The dict containing the mapping of attribute name to
                the `Field` object

        Returns:
            The Example object containing the precessed attributes.

        """
        for name, field in fields.items():
            setattr(self, name, field.preprocess(getattr(self, name)))
        return self

    def __eq__(self, obj):
        return self.__dict__ == obj.__dict__
