import json
import datetime as dt
from typing import Union


# Intended to implement the IVOA Provenance Data Model;
# see https://www.ivoa.net/documents/ProvenanceDM/20200411/index.html

class Entity:
    def __init__(self, i_d: str = None, name: str = None, location: str = None,
                 generated_at_time: Union[str, dt.datetime] = None, invalidated_at_time=None):
        self.id = id
        self.name = name
        self.location = location
        self.generated_at_time = generated_at_time
        self.invalidated_at_time = invalidated_at_time
