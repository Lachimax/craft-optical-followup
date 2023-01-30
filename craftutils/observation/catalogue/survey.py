from craftutils.observation.catalogue import Catalogue


class SurveyCatalogue(Catalogue):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
