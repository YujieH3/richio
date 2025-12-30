Getting started
===============

<!-- This is where you describe how to get set up on a clean install, including the -->
<!-- commands necessary to get the raw data (using the `sync_data_from_s3` command, -->
<!-- for example), and then how to make the cleaned, final data sets. -->

## Installation

Currently, to install RICHIO, run under the root directory of this repository:

```bash
pip install -e .
```

and one can import by

```python
import richio as rio
```

## Load a snapshot

Suppose you have a RICH snapshot `./snap_32.h5`, you can load it simply by

```python
>>> import richio as rio
>>> snap = rio.load('./snap_32.h5')
```

Let's take a look at the quantities we have
```python
>>> snap.keys()
['CMx',
 'CMy',
 'CMz',
 'Density',
 'Dissipation',
 'DpDx',
 'DpDy',
 'DpDz',
 'DrhoDx',
 'DrhoDy',
 'DrhoDz',
 'DsieDx',
 'DsieDy',
 'DsieDz',
 'Eg_0',
 'Erad',
 'ID',
 'InternalEnergy',
 'Pressure',
 'Temperature',
 'Volume',
 'Vx',
 'Vy',
 'Vz',
 'X',
 'Y',
 'Z',
 'divV',
 'stickers',
 'tracers']
```

Suppose we want the density field, we can use the key

```python
>>> density = snap['Density']
>>> density
unyt_array([4.85566085e-17, 4.94904353e-17, 4.84095341e-17, ...,
       2.43368753e-16, 1.48725291e-16, 1.37720716e-16], shape=(759004,), units='1988415860000000000000000000000*kg/Rsun**3')
```

The units is in RICH's default unit system, where mass unit is the solar mass, length unit the solar length, and time unit is set by fixing the gravitational constant G=1. 

We use [unyt](https://unyt.readthedocs.io/en/stable/) for our unit system, `density` is a `unyt_array`. You can easily convert to cgs unit by

```python
>>> density = density.in_cgs()
>>> density
unyt_array([2.86988266e-16, 2.92507542e-16, 2.86119000e-16, ...,
       1.43840311e-15, 8.79023777e-16, 8.13982497e-16], shape=(759004,), units='g/cm**3')
```

or the mks (m, kg, s) unit

```python
>>> density = density.in_mks()
>>> density
unyt_array([2.86988266e-13, 2.92507542e-13, 2.86119000e-13, ...,
       1.43840311e-12, 8.79023777e-13, 8.13982497e-13], shape=(759004,), units='kg/m**3')
```

And you can easily convert back to the RICH unit system by

```python
>>> density.in_base('rich')
>>> density
unyt_array([4.85566085e-17, 4.94904353e-17, 4.84095341e-17, ...,
       2.43368753e-16, 1.48725291e-16, 1.37720716e-16], shape=(759004,), units='1988415860000000000000000000000*kg/Rsun**3')
```

See [unyt's documentation](https://unyt.readthedocs.io/en/stable/) for other things you can do with an `unyt_array`.

