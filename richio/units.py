"""
Unit conversion. Update this when there's any change in the code.

By: Yujie He, Paola Martire
"""

import warnings

import numpy as np
import unyt as u
from unyt import Unit, UnitRegistry
from unyt.dimensions import length, mass, time
from unyt.unit_systems import UnitSystem


class Units:
    """
    Container for RICH simulation units. Mass unit is solar mass, length solar
    radius, and gravitational constant G = 1 (such that the time unit is also fixed).
    """

    def __init__(self):
        reg = UnitRegistry(unit_system="cgs")

        # base_value default in mks
        reg.add("code_mass", base_value=2e30, dimensions=mass, tex_repr=r"M_\odot")
        reg.add("code_length", base_value=7e8, dimensions=length, tex_repr=r"R_\odot")
        reg.add("code_time", base_value=1603.0, dimensions=time, tex_repr=r"t_\text{code}")

        # Base units
        self.mscale = Unit("code_mass", registry=reg)  # 2e33 * u.g    # ~ solar mass 1.988e33 g
        self.lscale = Unit(
            "code_length", registry=reg
        )  # 7e10 * u.cm   # ~ solar radius 6.955e10 cm
        self.tscale = Unit(
            "code_time", registry=reg
        )  # such that G ~ 1. would be ~ 1592 s if use more precise solar mass and solar radius

        # The rich unit system                  # convert by (1*)
        rus = UnitSystem(
            "rich",
            mass_unit=self.mscale,  # mass: Msun
            length_unit=self.lscale,  # length: Rsun
            time_unit=self.tscale,  # time chosen so that G=1
            registry=reg,
        )
        self.system = rus
        self.registry = reg

        self._unit_per_field = {
            "Box": self.lscale,
            "Time": self.tscale,
            "Cycle": u.Dimensionless,
            "CMx": self.lscale,
            "CMy": self.lscale,
            "CMz": self.lscale,
            "X": self.lscale,
            "Y": self.lscale,
            "Z": self.lscale,
            "Density": rus["density"],
            "Dissipation": rus["energy"] / self.lscale**3 / self.tscale,  # confirm this?
            "DpDx": rus["pressure"] / self.lscale,
            "DpDy": rus["pressure"] / self.lscale,
            "DpDz": rus["pressure"] / self.lscale,
            "DrhoDx": rus["density"] / self.lscale,
            "DrhoDy": rus["density"] / self.lscale,
            "DrhoDz": rus["density"] / self.lscale,
            "DsieDx": rus["energy"] / self.mscale / self.tscale,  # internal energy gradient?
            "DsieDy": rus["energy"] / self.mscale / self.tscale,  # internal energy gradient?
            "DsieDz": rus["energy"] / self.mscale / self.tscale,  # internal energy gradient?
            "Erad": rus["energy"] / self.mscale,
            "ID": u.Dimensionless,
            "InternalEnergy": rus["energy"] / self.mscale,
            "Pressure": rus["pressure"],
            "Temperature": rus["temperature"],
            "Volume": rus["volume"],
            "Vx": rus["velocity"],
            "Vy": rus["velocity"],
            "Vz": rus["velocity"],
            "divV": rus["velocity"] / self.lscale,
            "tracers/Entropy": rus["energy"]
            / rus["temperature"]
            / self.mscale,  # entropy per unit mass
            "tracers/Star": u.Dimensionless,  # proportion of matter that belongs to star, dimensionless, 0-1
            "Eg_0": 1,  # Elad: All of the things that are zero are not used in this simulation
            "stickers": 1,  #
            "tracers/WasRemoved": 1,  #
        }

        # Alias for npy files
        self._unit_per_field.update(
            {
                "box": self._unit_per_field["Box"],
                "Den": self._unit_per_field["Density"],
                "Diss": self._unit_per_field["Dissipation"],
                "Entropy": self._unit_per_field["tracers/Entropy"],
                "IE": self._unit_per_field["InternalEnergy"],
                "Mass": self.mscale,
                "P": self._unit_per_field["Pressure"],
                "Rad": self._unit_per_field["Erad"],
                "Star": self._unit_per_field["tracers/Star"],
                "T": self._unit_per_field["Temperature"],
                "tfb": 2.577726*u.day,  # fallback time for Mbh=10^4, Mstar=0.5, Rstar=0.47 temporary patch #TODO: think of a better solution...
                "Vol": self._unit_per_field["Volume"],
                "DivV": self._unit_per_field["divV"],
            }
        )

    def get_unit(self, key: str):
        """
        Return the unit associated with a RICH output field.
        """
        if key in self._unit_per_field.keys():
            unit = self._unit_per_field[key]

            if unit == 1:
                warnings.warn(f"'{key}' is in the data output but not used in the simulation.")

            return unit

        else:
            raise ValueError(f"Unknown key '{key}'. Supported keys: {list(self._unit_per_field)}")


# Singleton instance for convenience
units = Units()


def to_rich_units(qty):
    """
    Convert a unyt_array or unyt_quantity object to using the custom RICH unit
    registry. This does NOT change the unit, only the registry, such that 
    `qty.in_base('rich')` is allowed afterwards. 
    
    :param qty: unyt_array or unyt_quantity object.

    :returns: a unyt_array or unyt_quantity object but in RICH unit registry. 
    """
    
    if isinstance(qty, u.unyt_quantity):
        qty = u.unyt_quantity(
            qty.value,
            qty.units,
            registry=units.registry
        )
    elif isinstance(qty, u.unyt_array):
        qty = u.unyt_array(
            qty.value,
            qty.units,
            registry=units.registry
        )
    else:
        raise Exception("Quantity is neither an unyt_quantity nor an unyt_array.")

    qty = qty.in_base('rich')

    return qty

