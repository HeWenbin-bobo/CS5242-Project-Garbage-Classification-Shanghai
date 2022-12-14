# CS5242-Project-Garbage-Classification-Shanghai
For Neural Networks and Deep Learning (CS 5242) AY 2022-2023 Sem 1.
* Team member: Li Guoshen (A0237348X), An Yuhong (A0254379R), He Wenbin (A0237325H)
    * Github repository: https://github.com/HeWenbin-bobo/CS5242-Project-Garbage-Classification-Shanghai
    * Bokeh interative plot: [garbage_classification_plot_interactive](https://hewenbin-bobo.github.io/garbage_classification_plot_interactive.html)
    * Gradio online prediction: [CS5242-Project-Garbage-Classification-Shanghai (Gradio)](https://huggingface.co/spaces/wenbin1996/CS5242-Project-Garbage-Classification-Shanghai)
    * Google Drive: https://drive.google.com/drive/folders/10xNIXeeg0g6RxRFuXWtXgW94jSYIJuVb?usp=share_link
    * Please contact e0767608@u.nus.edu if any problem
## Abstract
* Increasing municipal garbage poses a great threat to city sustainability especially in developing countries. It has become one of the main sources of environmental pollution in Chinese cities. Accordingly, Urban Garbage Classification is of great significance to city sustainability. Essentially, it’s a social behavior and entails public participation. Based on Shanghai Garbage Classification System, MLP, CNN, and ANN models are utilized to solve the garbage classification problems. Our findings show that CNN and ANN models are more suitable for image handling. The best model, Vision Transformer, achieves an accuracy of 0.935 on the test set, suggesting its effectiveness on garbage classification. Also, some visualization tools are used to provide better analysis. Our study provides a new solution for urban garbage classification. It can be an useful tool for the government to protect environment and the sustainable development of cities.

## Project initialization
* Please run `model_code.ipynb` on colab. Program will automatically download all necessary files from github repository. (Or you can directly get all files from google drive link)

## MEMO
Since the well-trained VIT model is too large, we have to remove it in our Github repository (otherwise, we need to pay the Github LFS service). Instead, we upload the rar files of this model. Due to this update, some code may need to update in notebook (even if you download all files through Google Drive, still need to update some code to avoid re-download these files)