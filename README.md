# 11777_VIST_project
GitHub repo for our group (Steven, Chai, Alex) project for 11777, focusing on visual storytelling (VIST).


### Installation Packages
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install jupyterlab pandas numpy scipy matplotlib h5py scikit-learn -c conda-forge
pip install opencv-python nltk plotly bokeh seaborn imageio
```

### Data Re-Formatting
  * We are only taking the first 5 training splits and retaining 10k-train/3k-val/3k-test stories, which also have captions for all their corresponding images. This split accounts for approximately 25%/60%/60% of entire training/validation/test data.
  * Checkout `data_reformatting/vist_demo.ipynb` for a detailed demo
  * The re-formatted 10k-train/3k-val/3k-test stories with captions are saved in `data_reformatting/stories_withcaptions_jsons`

### Embeddings Visualization
  * Checkout `embeddings_visualization/BERT_visualizations.ipynb` for t-SNE analysis of stories and captions.
  * The sub-folder `embeddings_visualization` also has high-quality visualizations of t-SNE components for words, contextualized words and sentence representations.

### Linguistic Analysis
  * The jupyter notebook in the linguistic_analysis sub-folder contains all the code for linguistic analysis.

### uniModalVis
  * vis_visualization.ipynb includes the visual analysis of the VIST data
  * qualitative, quantitative folder contains visual analysis results (figures)
