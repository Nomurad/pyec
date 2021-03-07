# PyEC  

PyEC is a framework for evolutionary computing in Python3.  

## Algorithms

* MOEA/D
* MOEA/D-DE
* Constraint-MOEA/D (開発中)

## Install  

    python -m pip install -e .

## How to Use

* Optimize

        python example.py -i calc_input.json

* Constraint-Optimize

        python const_example.py -i calc_input.json

## requirement  

* python 3.6 >=
* numpy
* matplotlib
* dill
* icecream  (for debug)

## License  

pyec is released under the MIT License.  
