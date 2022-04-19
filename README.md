# lieops: Lie operator tools

Perform calculations with Lie operators of the form exp(:f:)

## Features

- Hamiltonians expressed in form of polynomials and njet.functions supported.
- Calculation of higher-order Birkhoff normal form.
- Custom phase space dimension and ordering.
- Contains a linear algebra package with several symplectic diagonalization routines, 
  as well as (linear) first-order normal form.

## Installation

Install this module with pip

```sh
pip install lieops
```

## Quickstart

An example
```python
from lieops import *
```

## Further reading

https://lieops.readthedocs.io/en/latest/index.html

## License

Copyright (C) 2021, 2022  Malte Titze

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
