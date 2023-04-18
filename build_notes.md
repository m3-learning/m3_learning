# building the wheel
python setup.py sdist bdist_wheel

# install from local build
pip install .

# Build Jupyter Notebook
conda run -n m3_learning jupyter-book build "C:\Users\jca92\Documents\codes\m3_learning\m3_learning"

# Upload to pypi
python -m twine upload --repository pypi dist/*
