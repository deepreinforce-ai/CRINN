cd python
rm -rf build tmp
rm -rf dist crinn.egg-info crinn_high.egg-info
python setup.py bdist_wheel
pip uninstall crinn -y
pip uninstall crinn_high -y
cd dist
ls | xargs pip install

