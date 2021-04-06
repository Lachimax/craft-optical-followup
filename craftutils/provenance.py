import json
import datetime as dt
from typing import Union


# Intended to implement the IVOA Provenance Data Model 1.0 (IVOA 1.0 henceforth);
# see https://www.ivoa.net/documents/ProvenanceDM/20200411/index.html

class Entity:
    def __init__(self, i_d: str = None, name: str = None, location: str = None,
                 generated_at_time: Union[str, dt.datetime] = None, invalidated_at_time=None,
                 comment: str = None):
        """

        :param i_d: a unique identifier for this entity;
            "id" in IVOA 1.0
        :param name: a human-readable name for the entity
        :param location: a path or spatial coordinates, e.g., a URL/URI, latitude-longitude coordinates on Earth, the
        name of a place.
        :param generated_at_time: date and time at which the entity was created (e.g., timestamp of a file);
            "generatedAtTime" in IVOA 1.0;
            Formatted in ISO
        :param invalidated_at_time: date and time of invalidation of the entity. After that date, the entity is no
        longer available for any use.
            "invalidatedAtTime" in IVOA 1.0
        :param comment
        """
        self.id = str(i_d)
        self.name = str(name)
        self.location = str(location)
        self.generated_at_time = dt.datetime.fromisoformat(generated_at_time)
        self.invalidated_at_time = invalidated_at_time
        self.comment = comment


class Activity:
    def __init__(self, i_d: str, name: str):
        """
        
        :param i_d:
        :param name:
        """
