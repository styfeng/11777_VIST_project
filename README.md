# 11777_VIST_project
GitHub repo for our group (Steven, Chai, Alex) project for 11777, focusing on visual storytelling (VIST).


### Installation Packages
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install jupyterlab pandas numpy scipy matplotlib h5py scikit-learn -c conda-forge
pip install opencv-python nltk plotly bokeh seaborn imageio
```

### Data Re-Formatting
  * We are only taking the first 3 training splits (~25%) as our training data.
  * Checkout `data_reformatting/vist_demo.ipynb` for a detailed demo
  * The re-formatted stories are saved in `data_reformatting/stories_jsons`

### Linguistic Analysis
  * The jupyter notebook in the linguistic_analysis sub-folder contains all the code for linguistic analysis.
