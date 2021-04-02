# Convert notebook to .rst file
jupyter-nbconvert --to rst $1
# Convert ipython3 flag to python
sed -i '' "s/.. code:: ipython3/.. code:: python/" $1
