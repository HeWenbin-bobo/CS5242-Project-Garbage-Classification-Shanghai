from bokeh.models import Div

#header
header = Div(text="""<h1>CS5242 Project: Garbage Classification - Based on Rules of Shanghai Garbage Classification System - Group 27</h1><h2>Team member: Li Guoshen (A0237348X), An Yuhong (A0254379R), He Wenbin (A0237325H)</h2>""")

# title for the toolbox
toolbox_header = Div(text="""<h1>Toolbox:</h1>""")

# project description
description = Div(text="""<p1>An inevitable consequence of intense human activities is the rapid increase in the amount of waste that is produced, and it is the way these waste are handled, stored, collected, and disposed of, which can pose risks to the environment and to public health. At the heart of waste disposal is reasonable waste classification. <b>①Although different countries/cities may have proposed the details for garbage classification, it is not realistic for people to classify them without intelligent tools.</b> As an increasingly detailed understanding of the different degrees of environmental, social, and economic risks associated with managing various types of ‘waste’ has developed, a variety of systems have been established to differentiate between materials. <b>②However, most available systems do not accept pictures as input, which makes it difficult for citizens to inquire about waste types.</b>
Our algorithm model and solution therefore complements the gap between legislation and practice by developing an image-based waste sorting system, which assists users to understand the characteristics of waste so that it can be managed and monitored appropriately in a manner that protects human health and the environment. Based on the optimal model among MLP, CNN, and ANN, users can get quick and accurate judgments on the waste types, improving the garbage classification quality. Also, a simple front-end layer is included in our solution that allows users to interact easily.
</p1>""")

# steps description
description2 = Div(text="""<h3>Approach:</h3>
<ul>
  <li>Fetch the images of garbage by designing our scrawler and modify their labels like <b>Recyclable - Glasses</b> to adapt our multi-class garbage classification model.</li>
  <li>Based on deep learning technology, design an image classification model to solve the garbage classification problem. <b>MLP</b> is set as the baseline, while <b>CNN</b> and <b>ANN</b> are considered to improve it.</li>
  <li>Based on the classification model, design a simple front-end layer for user interaction.</li>
</ul> 
<p1>Total of <b>450 samples</b> analysed.</p1>""")

# citation
cite = Div(text="""<p1><h3>Citation:</h3><a href="https://github.com/HeWenbin-bobo/CS5242-Project-Garbage-Classification-Shanghai">CS5344 Project: Garbage Classification - Based on Rules of Shanghai Garbage Classification System</a></p1>
<br><br>
""")

description_search = Div(text="""<h3>Filter by Text:</h3><p1>Search keyword to filter out the plot. It will search prediction labels. Press enter when ready.
Clear and press enter to reset the plot.</p1>""")

description_slider = Div(text="""<h3>Filter by the actual labels:</h3><p1>The slider below can be used to filter the target label. 
Simply slide the slider to the desired cluster number to display the plots that belong to that label. 
Slide back to 5 to show all.</p1>""")

description_keyword = Div(text="""<h3>Keywords:</h3>""")

description_current = Div(text="""<h3>Selected:</h3>""")

notes = Div(text="""<h3>Contact:</h3><p1><b>Organization: </b>School of Computing, National University Singapore (NUS). <br>
                                <b>Project Author: </b>Li Guoshen (XXXX), An Yuhong (XXXX), He Wenbin (e0767608@u.nus.edu)<br>
                                <b>PI: </b>Dr. You Yang & Dr. Ai Xin<br>
                                <b>GitHub: </b><a href="https://github.com/HeWenbin-bobo/CS5242-Project-Garbage-Classification-Shanghai">CS5242-Project-Garbage-Classification-Shanghai</a><br>
                                <b>Online Prediction APP: </b><a href="https://huggingface.co/spaces/wenbin1996/CS5242-Project-Garbage-Classification-Shanghai">CS5242-Project-Garbage-Classification-Shanghai (Gradio)</a><br>
                                <b>Bokeh Visualization: </b><a href="https://hewenbin-bobo.github.io/garbage_classification_plot_interactive.html">CS5242-Project-Garbage-Classification-Shanghai (Iterative Plot)</a><br>
</p1>
<br>""")

dataset_description = Div(text="""<h3>Dataset Description:</h3><p1><i>Shanghai Waste Classification System is valuable to provide waste-type information for thousands of items. It categorizes items into four main types: recyclable waste, hazardous waste, wet waste, and dry waste. Since this platform communicates in Chinese, we also need to use Baidu Translation API to get the corresponding English labels. Based on item labels, the waste image URLs are crawled through Flickr API. It displays identifiers for users, photos, photo albums, and other uniquely identifiable objects. Then, a simple python script is written, which fetches all the images available on a web page by giving a web page URL.</i></p1>""")

# steps description
tasks_for_each_member = Div(text="""<h3>Tasks for each member:</h3>
<ul>
  <li>An Yuhong: Majorly working on attention network such as transformer. Helping dataset acquisition & preprocessing and model training work.</li>
  <li>He Wenbin: Majorly working on MLP & CNN models and evaluation & interpretation. Develop a suitable front-end layer for solution. Summarize content and integrate reports.</li>
  <li>Li Guoshen: Majorly working on using class activation map for neural network interpretability, incorporate attention block  in CNN, and help model training. Helping and advising on data pre-processing work.</li>
</ul>""")