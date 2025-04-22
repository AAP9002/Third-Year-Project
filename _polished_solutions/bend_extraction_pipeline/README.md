Run to disable plot windows whilst running batch processing
```bash
export MPLBACKEND=Agg
```

Update the desired path to the input and output directories in `batch_harness.py`. The input directory should contain the raw data files, and the output directory will be where the processed data will be saved.

install pip requirements:
```bash
pip install opencv-python matplotlib pandas numpy scikit-learn pyproj kneed --quiet
```

Run the batch processing script:
```bash
time (python batch_harness.py)
```
| time optional