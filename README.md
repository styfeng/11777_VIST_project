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
  * The jupyter notebook in the linguistic_analysis sub-folder contains the code and associated data for linguistic analysis for both assignment 2 and the midterm.

### Evaluation
  * The evaluation sub-folder contains scripts for certain metrics including BERTScore (between two .txt files, one example per line), and perplexity using GPT-2 (of text in a single .txt file, one example per line).
  
### Baseline: AREL
  * The reimplementation of AREL baseline with our 25% subset of training split is provided in `baseline_reimplementations/AREL`. Experiments of stage 1 and 2 using XE and AREL optimizations are present in its directories: `data/save/777_XE` and `data/save/777_AREL` respectively.

### Baseline Generations and Results
  * The baseline_generations_results subfolder contains generations and metric results using the baseline models (AREL and GLAC Net) on our subset of the VIST test split. The images_midterm subsubfolder contains the images corresponding to the qualitative examples in our midterm report.
  
### uniModalVis
  * vis_visualization.ipynb includes the visual analysis of the VIST data
  * qualitative, quantitative folder contains visual analysis results (figures)
