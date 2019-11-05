from typing import List, Tuple, Dict
from ..data.field import Field


class Example:
    @classmethod
    def fromlist(cls, data: List[str], fields: List[Tuple[str, Field]]):
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
        for (name, field), val in zip(fields, data):
            setattr(ex, name, val)
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