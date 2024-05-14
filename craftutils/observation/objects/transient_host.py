import craftutils.utils as u

from .galaxy import Galaxy


@u.export
class TransientHostCandidate(Galaxy):
    def __init__(
            self,
            transient: 'Transient',
            z: float = 0.0,
            **kwargs
    ):
        super().__init__(
            z=z,
            **kwargs
        )
        self.transient = transient
        if isinstance(self.transient, str) and self.field is not None:
            transient = self.field.get_object(self.transient, allow_missing=True)
            if transient is not None:
                self.transient = transient

        self.P_O = None
        if "P_O" in kwargs:
            self.P_O = kwargs["P_O"]
        self.p_xO = None
        if "p_xO" in kwargs:
            self.p_xO = kwargs["p_xO"]
        self.P_Ox = None
        if "P_Ox" in kwargs:
            self.P_Ox = kwargs["P_Ox"]
        self.P_U = None
        if "P_U" in kwargs:
            self.P_U = kwargs["P_U"]
        self.P_Ux = None
        if "P_U" in kwargs:
            self.P_Ux = kwargs["P_Ux"]
        self.probabilistic_association_img = None
        if "probabilistic_association_img" in kwargs:
            self.probabilistic_association_img = kwargs["probabilistic_association_img"]

    def get_transient(self, tolerate_missing: bool = False):
        from .transient import Transient
        if self.transient is None:
            self.transient = Transient(
                host_galaxy=self,
                z=self.z,
                z_err=self.z_err
            )
        elif isinstance(self.transient, str):
            self.transient = self._get_object(self.transient, tolerate_missing=tolerate_missing)
        elif not isinstance(self.transient, Transient):
            raise ValueError(f"{self.name}.transient is not set correctly ({self.transient})")

        return self.transient

    @classmethod
    def default_params(cls):
        default_params = super().default_params()
        default_params.update({
            "type": "TransientHostCandidate",
            "transient": None,
            "P_O": None,
            "p_xO": None,
            "P_Ox": None,
            "probabilistic_association_img": None
        })
        return default_params

    def to_param_dict(self):
        dictionary = self.default_params()
        dictionary.update(super().to_param_dict())
        dictionary.update({
            "P_O": self.P_O,
            "p_xO": self.p_xO,
            "P_Ox": self.P_Ox,
            "P_U": self.P_U,
            "P_Ux": self.P_Ux,
            "probabilistic_association_img": self.probabilistic_association_img
        })
        if isinstance(self.transient, str):
            dictionary["transient"] = self.transient
        else:
            dictionary["transient"] = self.transient.name
        return dictionary
