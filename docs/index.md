***
# Supervised Learning : Convolutional Neural Network Frameworks for Multi-Class Image Classification

***
### John Pauline Pineda <br> <br> *December 30, 2023*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
        * [1.4.1 Image Description](#1.4.1)
        * [1.4.2 Image Normalization](#1.4.2)
        * [1.4.3 Image Reshaping](#1.4.3)
        * [1.4.4 Image Augmentation](#1.4.4)
        * [1.4.5 Preprocessed Data Description](#1.4.5)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
    * [1.6 Model Development](#1.6)
        * [1.6.1 Premodelling Data Description](#1.6.1)
        * [1.6.2 CNN With No Regularization](#1.6.2)
        * [1.6.3 CNN With Dropout Regularization](#1.6.3)
        * [1.6.4 CNN With Batch Normalization Regularization](#1.6.4)
        * [1.6.5 CNN With Dropout and Batch Normalization Regularization](#1.6.5)
    * [1.7 Consolidated Findings](#1.7)   
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project explores the various convolutional neural network (CNN) frameworks for processeing images through convolutional, activation, pooling, and fully connected layers, capturing hierarchical features and learning to map input images to their respective classes during training using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark>. Various CNN architectures applied in the analysis to learn features and patterns at different levels of abstraction in images included **CNN Without Regularization**, **CNN With Dropout Regularization**, **CNN With Batch Normalization Regularization** and **CNN With Dropout and Batch Normalization Regularization**. The different CNN algorithms were evaluated using the categorical cross entropy loss which measures the difference between the predicted probability distribution and the true distribution of the class labels. Model multi-classification performance was measured using **Accuracy**, **Precision**, **Recall** and **F1 Score**. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document.

A [convolutional neural network model](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) is a type of neural network architecture specifically designed for image classification and computer vision tasks by automatically learning hierarchical features directly from raw pixel data. The core building block of a CNN is the convolutional layer. Convolution operations apply learnable filters (kernels) to input images to detect patterns such as edges, textures, and more complex structures. The layers systematically learn hierarchical features from low-level (e.g., edges) to high-level (e.g., object parts) as the network deepens. Filters are shared across the entire input space, enabling the model to recognize patterns regardless of their spatial location. After convolutional operations, an activation function is applied element-wise to introduce non-linearity and allow the model to learn complex relationships between features. Pooling layers downsample the spatial dimensions of the feature maps, reducing the computational load and the number of parameters in the network - creating spatial hierarchy and translation invariance. Fully connected layers process the flattened features to make predictions and produce an output vector that corresponds to class probabilities using an activation function. The CNN is trained using backpropagation and optimization algorithms. A loss function is used to measure the difference between predicted and actual labels. The network adjusts its weights to minimize this loss. Gradients are calculated with respect to the loss, and the weights are updated accordingly through a backpropagation mechanism.


## 1.1. Data Background <a class="anchor" id="1.1"></a>

A subset of an open [COVID-19 Radiography Dataset](https://www.kaggle.com/datasets/preetviradiya/covid19-radiography-dataset) from [Kaggle](https://www.kaggle.com/) (with all credits attributed to [Preet Viradiya](https://www.kaggle.com/preetviradiya)) was used for the analysis as consolidated from the following primary sources: 
1. Covid19 X-Ray Images from [BIMCV Medical Imaging Databank of the Valencia Region](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711)
2. Covid19 X-Ray Images from [GitHub: ML Group](https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png)
3. Covid19 X-Ray Images from [Italian Society of Medical and Interventional Radiology](https://sirm.org/category/senza-categoria/covid-19/)
4. Covid19 X-Ray Images from [European Society of Radiology](https://eurorad.org/)
5. Covid19 X-Ray Images from [GitHub: Joseph Paul Cohen](https://github.com/ieee8023/covid-chestxray-dataset)
6. Covid19 X-Ray Images from [Publication: COVID-CXNet: Detecting COVID-19 in Frontal Chest X-ray Images using Deep Learning](https://github.com/armiro/COVID-CXNet)
7. Pneumonia and Normal X-Ray Images from [Kaggle: RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data)
8. Pneumonia and Normal X-Ray Images from [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

This study hypothesized that images contain a hierarchy of features which allows the differentiation and classification across various image categories. 

The target variable for the study is:
* <span style="color: #FF0000">CLASS</span> - Multi-categorical diagnostic classification for the x-ray images 

The hierarchical representation of image features enables the network to transform raw pixel data into a meaningful and compact representation, allowing it to make accurate predictions during image classification. The different features automatically learned during the training process are as follows:
* <span style="color: #FF0000">LOW-LEVEL FEATURES</span> - Edges and textures
* <span style="color: #FF0000">MID-LEVEL FEATURES</span> - Patterns and shapes
* <span style="color: #FF0000">HIGH-LEVEL FEATURES</span> - Object parts
* <span style="color: #FF0000">ABSTRACT FEATURES</span> - Object semantics
* <span style="color: #FF0000">SEMANTIC CONCEPTS</span> - Object categories
* <span style="color: #FF0000">HIERARCHICAL REPRESENTATION</span> - Spatial hierarchy
* <span style="color: #FF0000">ROTATION | SCALE INVARIANCE</span> - Invariant features
* <span style="color: #FF0000">LOCALIZATION INFORMATION</span> - Spatial localization


## 1.2. Data Description <a class="anchor" id="1.2"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details


```python
##################################
# Installing important packages
##################################
# !pip install mlxtend
# !pip install --upgrade tensorflow
# !pip install opencv-python
# !pip install keras==2.12.0
```


```python
##################################
# Loading Python Libraries 
# for Data Loading,
# Data Preprocessing and
# Exploratory Data Analysis
##################################
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
%matplotlib inline
import os
from PIL import Image
import cv2
from glob import glob
import random
import tensorflow
```

    WARNING:tensorflow:From C:\Users\John pauline magno\AppData\Roaming\Python\Python311\site-packages\keras\losses.py:2664: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
    
    


```python
##################################
# Loading Python Libraries 
# for Model Development
##################################
import keras
from keras.models import Sequential, Model,load_model
from keras.layers import Activation,Dense, Dropout, Flatten, Conv2D, MaxPooling2D,MaxPool2D,AveragePooling2D,GlobalMaxPooling2D, BatchNormalization
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, array_to_img
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
```


```python
##################################
# Loading Python Libraries 
# for Model Evaluation
##################################
from keras.metrics import PrecisionAtRecall,Recall 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
```


```python
##################################
# Setting random seed options
# for the analysis
##################################
import random, os
import numpy as np
import tensorflow as tf

def set_seed(seed=88888888):
    np.random.seed(seed) 
    tf.random.set_seed(seed) 
    keras.utils.set_random_seed(seed)
    random.seed(seed)
    tf.config.experimental.enable_op_determinism()
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISM'] = "1"
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()
```


```python
##################################
# Loading the dataset
##################################
path = 'C:/Users/John pauline magno/Python Notebooks/COVID-19_Radiography_Dataset/'

diagnosis_code_dictionary = {'COVID': 0,
                             'Normal': 1,
                             'Viral Pneumonia': 2}

diagnosis_description_dictionary = {'COVID': 'Covid-19',
                                    'Normal': 'Healthy',
                                    'Viral Pneumonia': 'Viral Pneumonia'}

imageid_path_dictionary = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path, '*','*.png'))}
```


```python
##################################
# Taking a snapshot of the dictionary
##################################
dict(list(imageid_path_dictionary.items())[0:5]) 
```




    {'COVID-1': 'C:/Users/John pauline magno/Python Notebooks/COVID-19_Radiography_Dataset\\COVID\\COVID-1.png',
     'COVID-10': 'C:/Users/John pauline magno/Python Notebooks/COVID-19_Radiography_Dataset\\COVID\\COVID-10.png',
     'COVID-100': 'C:/Users/John pauline magno/Python Notebooks/COVID-19_Radiography_Dataset\\COVID\\COVID-100.png',
     'COVID-1000': 'C:/Users/John pauline magno/Python Notebooks/COVID-19_Radiography_Dataset\\COVID\\COVID-1000.png',
     'COVID-1001': 'C:/Users/John pauline magno/Python Notebooks/COVID-19_Radiography_Dataset\\COVID\\COVID-1001.png'}




```python
##################################
# Consolidating the information
# from the dataset
# into a dataframe
##################################
xray_images = pd.DataFrame.from_dict(imageid_path_dictionary, orient = 'index').reset_index()
xray_images.columns = ['Image_ID','Path']
classes = xray_images.Image_ID.str.split('-').str[0]
xray_images['Diagnosis'] = classes
xray_images['Target'] = xray_images['Diagnosis'].map(diagnosis_code_dictionary.get) 
xray_images['Class'] = xray_images['Diagnosis'].map(diagnosis_description_dictionary.get) 
```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ')
display(xray_images.shape)
```

    Dataset Dimensions: 
    


    (3600, 5)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(xray_images.dtypes)
```

    Column Names and Data Types:
    


    Image_ID     object
    Path         object
    Diagnosis    object
    Target        int64
    Class        object
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
xray_images.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image_ID</th>
      <th>Path</th>
      <th>Diagnosis</th>
      <th>Target</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COVID-1</td>
      <td>C:/Users/John pauline magno/Python Notebooks/C...</td>
      <td>COVID</td>
      <td>0</td>
      <td>Covid-19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>COVID-10</td>
      <td>C:/Users/John pauline magno/Python Notebooks/C...</td>
      <td>COVID</td>
      <td>0</td>
      <td>Covid-19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>COVID-100</td>
      <td>C:/Users/John pauline magno/Python Notebooks/C...</td>
      <td>COVID</td>
      <td>0</td>
      <td>Covid-19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>COVID-1000</td>
      <td>C:/Users/John pauline magno/Python Notebooks/C...</td>
      <td>COVID</td>
      <td>0</td>
      <td>Covid-19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>COVID-1001</td>
      <td>C:/Users/John pauline magno/Python Notebooks/C...</td>
      <td>COVID</td>
      <td>0</td>
      <td>Covid-19</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Performing a general exploration of the numeric variables
##################################
print('Numeric Variable Summary:')
display(xray_images.describe(include='number').transpose())
```

    Numeric Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Target</th>
      <td>3600.0</td>
      <td>1.0</td>
      <td>0.81661</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the object variable
##################################
print('Object Variable Summary:')
display(xray_images.describe(include='object').transpose())
```

    Object Variable Summary:
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Image_ID</th>
      <td>3600</td>
      <td>3600</td>
      <td>COVID-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Path</th>
      <td>3600</td>
      <td>3600</td>
      <td>C:/Users/John pauline magno/Python Notebooks/C...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Diagnosis</th>
      <td>3600</td>
      <td>3</td>
      <td>COVID</td>
      <td>1200</td>
    </tr>
    <tr>
      <th>Class</th>
      <td>3600</td>
      <td>3</td>
      <td>Covid-19</td>
      <td>1200</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the target variable
##################################
xray_images.Diagnosis.value_counts()
```




    COVID              1200
    Normal             1200
    Viral Pneumonia    1200
    Name: Diagnosis, dtype: int64




```python
##################################
# Performing a general exploration of the target variable
##################################
xray_images.Diagnosis.value_counts(normalize=True)
```




    COVID              0.333333
    Normal             0.333333
    Viral Pneumonia    0.333333
    Name: Diagnosis, dtype: float64



## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

Data quality findings based on assessment are as follows:
1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details


```python
##################################
# Counting the number of duplicated images
##################################
xray_images.duplicated().sum()
```




    0




```python
##################################
# Gathering the number of null images
##################################
xray_images.isnull().sum()
```




    Image_ID     0
    Path         0
    Diagnosis    0
    Target       0
    Class        0
    dtype: int64



## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>


### 1.4.1 Image Description <a class="anchor" id="1.4.1"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details


```python
##################################
# Including the pixel information
# of the actual images
# in array format
# into a dataframe
##################################
xray_images['Image'] = xray_images['Path'].map(lambda x: np.asarray(Image.open(x).resize((75,75))))
```


```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(xray_images.dtypes)
```

    Column Names and Data Types:
    


    Image_ID     object
    Path         object
    Diagnosis    object
    Target        int64
    Class        object
    Image        object
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
xray_images.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Image_ID</th>
      <th>Path</th>
      <th>Diagnosis</th>
      <th>Target</th>
      <th>Class</th>
      <th>Image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COVID-1</td>
      <td>C:/Users/John pauline magno/Python Notebooks/C...</td>
      <td>COVID</td>
      <td>0</td>
      <td>Covid-19</td>
      <td>[[15, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>COVID-10</td>
      <td>C:/Users/John pauline magno/Python Notebooks/C...</td>
      <td>COVID</td>
      <td>0</td>
      <td>Covid-19</td>
      <td>[[129, 125, 123, 121, 119, 117, 114, 104, 104,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>COVID-100</td>
      <td>C:/Users/John pauline magno/Python Notebooks/C...</td>
      <td>COVID</td>
      <td>0</td>
      <td>Covid-19</td>
      <td>[[11, 0, 0, 3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>COVID-1000</td>
      <td>C:/Users/John pauline magno/Python Notebooks/C...</td>
      <td>COVID</td>
      <td>0</td>
      <td>Covid-19</td>
      <td>[[42, 39, 38, 42, 38, 35, 31, 26, 24, 24, 24, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>COVID-1001</td>
      <td>C:/Users/John pauline magno/Python Notebooks/C...</td>
      <td>COVID</td>
      <td>0</td>
      <td>Covid-19</td>
      <td>[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0,...</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Taking a snapshot of the dataset
##################################
n_samples = 5
fig, m_axs = plt.subplots(3, n_samples, figsize = (3*n_samples, 8))
for n_axs, (type_name, type_rows) in zip(m_axs, xray_images.sort_values(['Diagnosis']).groupby('Diagnosis')):
    n_axs[2].set_title(type_name, fontsize = 14, weight = 'bold')
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1).iterrows()):       
        picture = c_row['Path']
        image = cv2.imread(picture)
        c_ax.imshow(image)
        c_ax.axis('off')
```


    
![png](output_30_0.png)
    



```python
##################################
# Sampling a single image
##################################
samples, features = xray_images.shape
plt.figure()
pic_id = random.randrange(0, samples)
picture = xray_images['Path'][pic_id]
image = cv2.imread(picture) 
```


    <Figure size 640x480 with 0 Axes>



```python
##################################
# Plotting using subplots
##################################
plt.figure(figsize=(15, 5))

##################################
# Formulating the original image
##################################
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Formulating the blue channel
##################################
plt.subplot(1, 4, 2)
plt.imshow(image[ : , : , 0])
plt.title('Blue Channel', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Formulating the green channel
##################################
plt.subplot(1, 4, 3)
plt.imshow(image[ : , : , 1])
plt.title('Green Channel', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Formulating the red channel
##################################
plt.subplot(1, 4, 4)
plt.imshow(image[ : , : , 2])
plt.title('Blue Channel', fontsize = 14, weight = 'bold')
plt.axis('off')

##################################
# Consolidating all images
##################################
plt.show()
```


    
![png](output_32_0.png)
    



```python
##################################
# Determining the image shape
##################################
print('Image Shape:')
display(image.shape)
```

    Image Shape:
    


    (299, 299, 3)



```python
##################################
# Determining the image height
##################################
print('Image Height:')
display(image.shape[0])
```

    Image Height:
    


    299



```python
##################################
# Determining the image width
##################################
print('Image Width:')
display(image.shape[0])
```

    Image Width:
    


    299



```python
##################################
# Determining the image dimension
##################################
print('Image Dimension:')
display(image.ndim)
```

    Image Dimension:
    


    3



```python
##################################
# Determining the image size
##################################
print('Image Size:')
display(image.size)
```

    Image Size:
    


    268203



```python
##################################
# Determining the image data type
##################################
print('Image Data Type:')
display(image.dtype)
```

    Image Data Type:
    


    dtype('uint8')



```python
##################################
# Determining the maximum RGB value
##################################
print('Image Maximum RGB:')
display(image.max())
```

    Image Maximum RGB:
    


    205



```python
##################################
# Determining the minimum RGB value
##################################
print('Image Minimum RGB:')
display(image.min())
```

    Image Minimum RGB:
    


    10


### 1.4.2 Image Normalization <a class="anchor" id="1.4.2"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details


```python
##################################
# Update
##################################
```

### 1.4.3 Image Reshaping <a class="anchor" id="1.4.3"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details


```python
##################################
# Update
##################################
```

### 1.4.4 Image Augmentation <a class="anchor" id="1.4.4"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details


```python
##################################
# Update
##################################
```

### 1.4.5 Preprocessed Data Description <a class="anchor" id="1.4.5"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details


```python
##################################
# Update
##################################
```

## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details


```python
##################################
# Update
##################################
mean_val = []
std_dev_val = []
max_val = []
min_val = []

for i in range(0,samples):
    mean_val.append(xray_images['Image'][i].mean())
    std_dev_val.append(np.std(xray_images['Image'][i]))
    max_val.append(xray_images['Image'][i].max())
    min_val.append(xray_images['Image'][i].min())

imageEDA = xray_images.loc[:,['Image', 'Class','Path']]
imageEDA['Mean'] = mean_val
imageEDA['StDev'] = std_dev_val
imageEDA['Max'] = max_val
imageEDA['Min'] = min_val

subt_mean_samples = imageEDA['Mean'].mean() - imageEDA['Mean']
imageEDA['Subt_Mean'] = subt_mean_samples
```


```python
##################################
# Update
##################################
ax = sns.displot(data = imageEDA, x = 'Mean', kind="kde");
plt.title('Images Colour Mean Value Distribution', fontsize = 16,weight = 'bold');
ax = sns.displot(data = imageEDA, x = 'Mean', kind="kde", hue = 'Class');
plt.title('Images Colour Mean Value Distribution by Class', fontsize = 16,weight = 'bold');
ax = sns.displot(data = imageEDA, x = 'Max', kind="kde", hue = 'Class');
plt.title('Images Colour Max Value Distribution by Class', fontsize = 16,weight = 'bold');
ax = sns.displot(data = imageEDA, x = 'Min', kind="kde", hue = 'Class');
plt.title('Images Colour Min Value Distribution by Class', fontsize = 16,weight = 'bold');
```


    
![png](output_52_0.png)
    



    
![png](output_52_1.png)
    



    
![png](output_52_2.png)
    



    
![png](output_52_3.png)
    



```python
plt.figure(figsize=(20,8))
sns.set(style="ticks", font_scale = 1)
ax = sns.scatterplot(data=imageEDA, x="Mean", y=imageEDA['StDev'], hue = 'Class',alpha=0.8);
sns.despine(top=True, right=True, left=False, bottom=False)
plt.xticks(rotation=0,fontsize = 12)
ax.set_xlabel('Image Channel Colour Mean',fontsize = 14,weight = 'bold')
ax.set_ylabel('Image Channel Colour Standard Deviation',fontsize = 14,weight = 'bold')
plt.title('Mean and Standard Deviation of Image Samples', fontsize = 16,weight = 'bold');
```


    
![png](output_53_0.png)
    



```python
plt.figure(figsize=(20,8));
g = sns.FacetGrid(imageEDA, col="Class",height=5);
g.map_dataframe(sns.scatterplot, x='Mean', y='StDev');
g.set_titles(col_template="{col_name}", row_template="{row_name}", size = 16)
g.fig.subplots_adjust(top=.7)
g.fig.suptitle('Mean and Standard Deviation of Image Samples',fontsize=16, weight = 'bold')
axes = g.axes.flatten()
axes[0].set_ylabel('Standard Deviation');
for ax in axes:
    ax.set_xlabel('Mean')
g.fig.tight_layout()
```


    <Figure size 2000x800 with 0 Axes>



    
![png](output_54_1.png)
    



```python
def getImage(path):
    return OffsetImage(cv2.imread(path),zoom = 0.1)

DF_sample = imageEDA.sample(frac=1.0, replace=False, random_state=1)
paths = DF_sample['Path']

fig, ax = plt.subplots(figsize=(20,8))
ab = sns.scatterplot(data=DF_sample, x="Mean", y='StDev')
sns.despine(top=True, right=True, left=False, bottom=False)
ax.set_xlabel('Image Channel Colour Mean',fontsize = 14,weight = 'bold')
ax.set_ylabel('Image Channel Colour Standard Deviation',fontsize = 14,weight = 'bold')
plt.title('Mean and Standard Deviation of Image Samples', fontsize = 16,weight = 'bold');

for x0, y0, path in zip(DF_sample['Mean'], DF_sample['StDev'],paths):
    ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    ax.add_artist(ab)
```


    
![png](output_55_0.png)
    


## 1.6. Model Development <a class="anchor" id="1.6"></a>

### 1.6.1 Premodelling Data Description <a class="anchor" id="1.6.1"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details


```python
##################################
# Update
##################################
#add the path general where the classes subpath are allocated
path = 'C:/Users/John pauline magno/Python Notebooks/COVID-19_Radiography_Dataset'

classes=["COVID", "Normal", "Viral Pneumonia"]
num_classes = len(classes)
batch_size = 16

set_seed()

#Define the parameters to create the training and validation set Images and Data Augmentation parameters
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

set_seed()

#**No Augmentation on the Test set Images**
test_datagen = ImageDataGenerator(rescale=1./255, 
                                  validation_split=0.2)


#loading the images to training set
train_gen = train_datagen.flow_from_directory(directory=path, 
                                              target_size=(299, 299),
                                              class_mode='categorical',
                                              subset='training',
                                              shuffle=True, classes=classes,
                                              batch_size=batch_size, 
                                              color_mode="grayscale")
#loading the images to test set
test_gen = test_datagen.flow_from_directory(directory=path, 
                                              target_size=(299, 299),
                                              class_mode='categorical',
                                              subset='validation',
                                              shuffle=False, classes=classes,
                                              batch_size=batch_size, 
                                              color_mode="grayscale")
```

    Found 2880 images belonging to 3 classes.
    Found 720 images belonging to 3 classes.
    


```python
def plot_training_history(history, model_name):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name} Training Loss', fontsize = 16, weight = 'bold', pad=20)
    plt.ylim(0, 5)
    plt.xlabel('Epoch', fontsize = 14, weight = 'bold',)
    plt.ylabel('Loss', fontsize = 14, weight = 'bold',)
    plt.legend()
    plt.show()
```

### 1.6.2 CNN With No Regularization <a class="anchor" id="1.6.2"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details


```python
##################################
# Formulating the network architecture
# for CNN with no regularization
##################################
set_seed()
batch_size = 16
model_nr = Sequential()
model_nr.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding = 'Same', input_shape=(299, 299, 1)))
model_nr.add(MaxPooling2D(pool_size=(2, 2)))
model_nr.add(Conv2D(64, kernel_size=(3, 3), padding = 'Same', activation='relu'))
model_nr.add(MaxPooling2D(pool_size=(2, 2)))
model_nr.add(Flatten())
model_nr.add(Dense(128, activation='relu'))
model_nr.add(Dense(num_classes, activation='softmax'))

##################################
# Compiling the network layers
##################################
model_nr.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall()])
```

    WARNING:tensorflow:From C:\Users\John pauline magno\AppData\Roaming\Python\Python311\site-packages\keras\backend.py:873: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    WARNING:tensorflow:From C:\Users\John pauline magno\AppData\Roaming\Python\Python311\site-packages\keras\layers\pooling\max_pooling2d.py:160: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.
    
    WARNING:tensorflow:From C:\Users\John pauline magno\AppData\Roaming\Python\Python311\site-packages\keras\optimizers\__init__.py:300: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    


```python
##################################
# Fitting the model
# for CNN with no regularization
##################################
epochs = 100
set_seed()
model_nr_history = model_nr.fit(train_gen, 
                                steps_per_epoch=len(train_gen) // batch_size,   
                                validation_steps=len(test_gen) // batch_size, 
                                validation_data=test_gen, 
                                epochs=epochs,
                                verbose=0)
```

    WARNING:tensorflow:From C:\Users\John pauline magno\AppData\Roaming\Python\Python311\site-packages\keras\utils\tf_utils.py:490: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.
    
    


```python
##################################
# Evaluating the model
# for CNN with no regularization
# on the independent validation set
##################################
model_nr_y_pred = model_nr.predict(test_gen)
```

    45/45 [==============================] - 4s 76ms/step
    


```python
##################################
# Plotting the loss profile
# for CNN with no regularization
# on the training and validation sets
##################################
plot_training_history(model_nr_history, 'CNN With No Regularization : ')
```


    
![png](output_64_0.png)
    



```python
##################################
# Consolidating the predictions
# for CNN with no regularization
# on the validation set
##################################
model_nr_predictions = np.array(list(map(lambda x: np.argmax(x), model_nr_y_pred)))
model_nr_y_true=test_gen.classes

##################################
# Formulating the confusion matrix
# for CNN with no regularization
# on the validation set
##################################
CMatrix = pd.DataFrame(confusion_matrix(model_nr_y_true, model_nr_predictions), columns=classes, index =classes)

##################################
# Plotting the confusion matrix
# for CNN with no regularization
# on the validation set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(CMatrix, annot = True, fmt = 'g' ,vmin = 0, vmax = 250,cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('CNN With No Regularization : Validation Set Confusion Matrix',fontsize = 14, weight = 'bold',pad=20);

##################################
# Resetting all states generated by Keras
##################################
keras.backend.clear_session()
```


    
![png](output_65_0.png)
    



```python
##################################
# Calculating the model accuracy
# for CNN with no regularization
# for the entire validation set
##################################
model_nr_acc = accuracy_score(model_nr_y_true, model_nr_predictions)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for CNN with no regularization
# for the entire validation set
##################################
model_nr_results_all = precision_recall_fscore_support(model_nr_y_true, model_nr_predictions, average='macro',zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for CNN with no regularization
# for each category of the validation set
##################################
model_nr_results_class = precision_recall_fscore_support(model_nr_y_true, model_nr_predictions, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for CNN with no regularization
##################################
metric_columns = ['Precision','Recall', 'F-Score','Support']
model_nr_all_df = pd.concat([pd.DataFrame(list(model_nr_results_class)).T,pd.DataFrame(list(model_nr_results_all)).T])
model_nr_all_df.columns = metric_columns
model_nr_all_df.index = ['COVID', 'Normal', 'Viral Pneumonia','Total']
model_nr_all_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>COVID</th>
      <td>0.921739</td>
      <td>0.883333</td>
      <td>0.902128</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.905213</td>
      <td>0.795833</td>
      <td>0.847007</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>Viral Pneumonia</th>
      <td>0.795699</td>
      <td>0.925000</td>
      <td>0.855491</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.874217</td>
      <td>0.868056</td>
      <td>0.868209</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### 1.6.3 CNN With Dropout Regularization <a class="anchor" id="1.6.3"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details    


```python
##################################
# Formulating the network architecture
# for CNN with dropout regularization
##################################
set_seed()
batch_size = 16
model_dr = Sequential()
model_dr.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding = 'Same', input_shape=(299, 299, 1)))
model_dr.add(MaxPooling2D(pool_size=(2, 2)))
model_dr.add(Dropout(0.25))
model_dr.add(Conv2D(64, kernel_size=(3, 3), padding = 'Same', activation='relu'))
model_dr.add(MaxPooling2D(pool_size=(2, 2)))
model_dr.add(Dropout(0.25))
model_dr.add(Flatten())
model_dr.add(Dense(128, activation='relu'))
model_dr.add(Dropout(0.25))
model_dr.add(Dense(num_classes, activation='softmax'))

##################################
# Compiling the network layers
##################################
model_dr.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall()])
```


```python
##################################
# Fitting the model
# for CNN with dropout regularization
##################################
epochs = 100
set_seed()
model_dr_history = model_dr.fit(train_gen, 
                                steps_per_epoch=len(train_gen) // batch_size, 
                                validation_steps=len(test_gen) // batch_size, 
                                validation_data=test_gen, 
                                epochs=epochs,
                                verbose=0)
```


```python
##################################
# Evaluating the model
# for CNN with dropout regularization
# on the independent validation set
##################################
model_dr_y_pred = model_dr.predict(test_gen)
```

    45/45 [==============================] - 4s 96ms/step
    


```python
##################################
# Plotting the loss profile
# for CNN with dropout regularization
# on the training and validation sets
##################################
plot_training_history(model_dr_history, 'CNN With Dropout Regularization : ')
```


    
![png](output_71_0.png)
    



```python
##################################
# Consolidating the predictions
# for CNN with dropout regularization
# on the validation set
##################################
model_dr_predictions = np.array(list(map(lambda x: np.argmax(x), model_dr_y_pred)))
model_dr_y_true=test_gen.classes

##################################
# Formulating the confusion matrix
# for CNN with dropout regularization
# on the validation set
##################################
CMatrix = pd.DataFrame(confusion_matrix(model_dr_y_true, model_dr_predictions), columns=classes, index =classes)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for CNN with dropout regularization
# for each category of the validation set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(CMatrix, annot = True, fmt = 'g' ,vmin = 0, vmax = 250, cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('CNN With Dropout Regularization : Validation Set Confusion Matrix',fontsize = 14, weight = 'bold', pad=20);

##################################
# Resetting all states generated by Keras
##################################
keras.backend.clear_session()
```


    
![png](output_72_0.png)
    



```python
##################################
# Calculating the model accuracy
# for CNN with dropout regularization
# for the entire validation set
##################################
model_dr_acc = accuracy_score(model_dr_y_true, model_dr_predictions)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for CNN with dropout regularization
# for the entire validation set
##################################
model_dr_results_all = precision_recall_fscore_support(model_dr_y_true, model_dr_predictions, average='macro',zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for CNN with dropout regularization
# for each category of the validation set
##################################
model_dr_results_class = precision_recall_fscore_support(model_dr_y_true, model_dr_predictions, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for CNN with dropout regularization
##################################
metric_columns = ['Precision','Recall', 'F-Score','Support']
model_dr_all_df = pd.concat([pd.DataFrame(list(model_dr_results_class)).T,pd.DataFrame(list(model_dr_results_all)).T])
model_dr_all_df.columns = metric_columns
model_dr_all_df.index = ['COVID', 'Normal', 'Viral Pneumonia','Total']
model_dr_all_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>COVID</th>
      <td>0.898734</td>
      <td>0.887500</td>
      <td>0.893082</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.882629</td>
      <td>0.783333</td>
      <td>0.830022</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>Viral Pneumonia</th>
      <td>0.829630</td>
      <td>0.933333</td>
      <td>0.878431</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.870331</td>
      <td>0.868056</td>
      <td>0.867178</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### 1.6.4 CNN With Batch Normalization Regularization <a class="anchor" id="1.6.4"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details  


```python
##################################
# Formulating the network architecture
# for CNN with batch normalization regularization
##################################
set_seed()
batch_size = 16
model_bnr = Sequential()
model_bnr.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding = 'Same', input_shape=(299, 299, 1)))
model_bnr.add(MaxPooling2D(pool_size=(2, 2)))
model_bnr.add(Conv2D(64, kernel_size=(3, 3), padding = 'Same', activation='relu'))
model_bnr.add(BatchNormalization())
model_bnr.add(Activation('relu'))
model_bnr.add(MaxPooling2D(pool_size=(2, 2)))
model_bnr.add(Flatten())
model_bnr.add(Dense(128, activation='relu'))
model_bnr.add(Dense(num_classes, activation='softmax'))

##################################
# Compiling the network layers
##################################
model_bnr.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall()])
```


```python
##################################
# Fitting the model
# for CNN with batch normalization regularization
##################################
epochs = 100
set_seed()
model_bnr_history = model_bnr.fit(train_gen, 
                                  steps_per_epoch=len(train_gen) // batch_size,
                                  validation_steps=len(test_gen) // batch_size, 
                                  validation_data=test_gen, epochs=epochs,
                                  verbose=0)
```


```python
##################################
# Evaluating the model
# for CNN with batch normalization regularization
# on the independent validation set
##################################
model_bnr_y_pred = model_bnr.predict(test_gen)
```

    45/45 [==============================] - 4s 87ms/step
    


```python
##################################
# Plotting the loss profile
# for CNN with batch normalization regularization
# on the training and validation sets
##################################
plot_training_history(model_bnr_history, 'CNN With Batch Normalization Regularization : ')
```


    
![png](output_78_0.png)
    



```python
##################################
# Consolidating the predictions
# for CNN with batch normalization regularization
# on the validation set
##################################
model_bnr_predictions = np.array(list(map(lambda x: np.argmax(x), model_bnr_y_pred)))
model_bnr_y_true=test_gen.classes

##################################
# Formulating the confusion matrix
# for CNN with batch normalization regularization
# on the validation set
##################################
CMatrix = pd.DataFrame(confusion_matrix(model_bnr_y_true, model_bnr_predictions), columns=classes, index =classes)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for CNN with batch normalization regularization
# for each category of the validation set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(CMatrix, annot = True, fmt = 'g' ,vmin = 0, vmax = 250,cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('CNN With Batch Normalization Regularization : Validation Set Confusion Matrix',fontsize = 16,weight = 'bold',pad=20);

##################################
# Resetting all states generated by Keras
##################################
keras.backend.clear_session()
```


    
![png](output_79_0.png)
    



```python
##################################
# Calculating the model accuracy
# for CNN with batch normalization regularization
# for the entire validation set
##################################
model_bnr_acc = accuracy_score(model_bnr_y_true, model_bnr_predictions)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for CNN with batch normalization regularization
# for the entire validation set
##################################
model_bnr_results_all = precision_recall_fscore_support(model_bnr_y_true, model_bnr_predictions, average='macro',zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for CNN with batch normalization regularization
# for each category of the validation set
##################################
model_bnr_results_class = precision_recall_fscore_support(model_bnr_y_true, model_bnr_predictions, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for CNN with batch normalization regularization
##################################
metric_columns = ['Precision','Recall', 'F-Score','Support']
model_bnr_all_df = pd.concat([pd.DataFrame(list(model_bnr_results_class)).T,pd.DataFrame(list(model_bnr_results_all)).T])
model_bnr_all_df.columns = metric_columns
model_bnr_all_df.index = ['COVID', 'Normal', 'Viral Pneumonia','Total']
model_bnr_all_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>COVID</th>
      <td>0.924686</td>
      <td>0.920833</td>
      <td>0.922756</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.837302</td>
      <td>0.879167</td>
      <td>0.857724</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>Viral Pneumonia</th>
      <td>0.877729</td>
      <td>0.837500</td>
      <td>0.857143</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.879906</td>
      <td>0.879167</td>
      <td>0.879207</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### 1.6.5 CNN With Dropout and Batch Normalization Regularization <a class="anchor" id="1.6.5"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details      


```python
##################################
# Formulating the network architecture
# for CNN with dropout and batch normalization regularization
##################################
set_seed()
batch_size = 16
model_dr_bnr = Sequential()
model_dr_bnr.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding = 'Same', input_shape=(299, 299, 1)))
model_dr_bnr.add(MaxPooling2D(pool_size=(2, 2)))
model_dr_bnr.add(Conv2D(64, kernel_size=(3, 3), padding = 'Same', activation='relu'))
model_dr_bnr.add(BatchNormalization())
model_dr_bnr.add(Activation('relu'))
model_dr_bnr.add(Dropout(0.25))
model_dr_bnr.add(MaxPooling2D(pool_size=(2, 2)))
model_dr_bnr.add(Flatten())
model_dr_bnr.add(Dense(128, activation='relu'))
model_dr_bnr.add(Dense(num_classes, activation='softmax'))

##################################
# Compiling the network layers
##################################
model_dr_bnr .compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Recall()])
```


```python
##################################
# Fitting the model
# for CNN with dropout and
# batch normalization regularization
##################################
epochs = 100
set_seed()
model_dr_bnr_history = model_dr_bnr.fit(train_gen,
                                        steps_per_epoch=len(train_gen) // batch_size,
                                        validation_steps=len(test_gen) // batch_size, 
                                        validation_data=test_gen, 
                                        epochs=epochs,
                                        verbose=0)
```


```python
##################################
# Evaluating the model
# for CNN with dropout and
# batch normalization regularization
# on the independent validation set
##################################
model_dr_bnr_y_pred = model_dr_bnr.predict(test_gen)
```

    45/45 [==============================] - 4s 93ms/step
    


```python
##################################
# Plotting the loss profile
# for CNN with dropout and
# batch normalization regularization
# on the training and validation sets
##################################
plot_training_history(model_dr_bnr_history, 'CNN With Dropout and Batch Normalization Regularization : ')
```


    
![png](output_85_0.png)
    



```python
##################################
# Consolidating the predictions
# for CNN with dropout and
# batch normalization regularization
# on the validation set
##################################
model_dr_bnr_predictions = np.array(list(map(lambda x: np.argmax(x), model_dr_bnr_y_pred)))
model_dr_bnr_y_true=test_gen.classes

##################################
# Formulating the confusion matrix
# for CNN with dropout and
# batch normalization regularization
# on the validation set
##################################
CMatrix = pd.DataFrame(confusion_matrix(model_dr_bnr_y_true, model_dr_bnr_predictions), columns=classes, index =classes)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for CNN with dropout and
# batch normalization regularization
# for each category of the validation set
##################################
plt.figure(figsize=(10, 6))
ax = sns.heatmap(CMatrix, annot = True, fmt = 'g' ,vmin = 0, vmax = 250,cmap = 'icefire')
ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
ax.set_xticklabels(ax.get_xticklabels(),rotation =0)
ax.set_ylabel('Actual',fontsize = 14,weight = 'bold') 
ax.set_yticklabels(ax.get_yticklabels(),rotation =0)
ax.set_title('CNN With Dropout and Batch Normalization Regularization : Validation Set Confusion Matrix',fontsize = 16,weight = 'bold',pad=20);

##################################
# Resetting all states generated by Keras
##################################
keras.backend.clear_session()
```


    
![png](output_86_0.png)
    



```python
##################################
# Calculating the model accuracy
# for CNN with dropout and
# batch normalization regularization
# for the entire validation set
##################################
model_dr_bnr_acc = accuracy_score(model_dr_bnr_y_true, model_dr_bnr_predictions)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for CNN with dropout and
# batch normalization regularization
# for the entire validation set
##################################
model_dr_bnr_results_all = precision_recall_fscore_support(model_dr_bnr_y_true, model_dr_bnr_predictions, average='macro',zero_division = 1)

##################################
# Calculating the model 
# Precision, Recall, F-score and Support
# for CNN with dropout and
# batch normalization regularization
# for each category of the validation set
##################################
model_dr_bnr_results_class = precision_recall_fscore_support(model_dr_bnr_y_true, model_dr_bnr_predictions, average=None, zero_division = 1)

##################################
# Consolidating all model evaluation metrics 
# for CNN with dropout and
# batch normalization regularization
##################################
metric_columns = ['Precision','Recall', 'F-Score','Support']
model_dr_bnr_all_df = pd.concat([pd.DataFrame(list(model_dr_bnr_results_class)).T,pd.DataFrame(list(model_dr_bnr_results_all)).T])
model_dr_bnr_all_df.columns = metric_columns
model_dr_bnr_all_df.index = ['COVID', 'Normal', 'Viral Pneumonia','Total']
model_dr_bnr_all_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F-Score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>COVID</th>
      <td>0.908714</td>
      <td>0.912500</td>
      <td>0.910603</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>Normal</th>
      <td>0.875000</td>
      <td>0.845833</td>
      <td>0.860169</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>Viral Pneumonia</th>
      <td>0.842105</td>
      <td>0.866667</td>
      <td>0.854209</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>Total</th>
      <td>0.875273</td>
      <td>0.875000</td>
      <td>0.874994</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## 1.7. Consolidated Findings <a class="anchor" id="1.7"></a>

1. Details
    * 1.1 Details
        * 1.1.1 Details
            * 1.1.1.1 Details         


```python
##################################
# Update
##################################
```

# 2. Summary <a class="anchor" id="Summary"></a>

A detailed [report](https://github.com/JohnPaulinePineda/Unsupervised-Machine-Learning-Final-Assignment/blob/main/UnsupervisedMachineLearningCapstone_JohnPaulinePineda.pdf) was formulated documenting all the analysis steps and findings.



```python
##################################
# Introduction
##################################
```


```python
##################################
# Methodology
##################################
```


```python
##################################
# Data Gathering
##################################
```


```python
##################################
# Data Description
##################################
```


```python
##################################
# Data Quality Assessment
##################################
```


```python
##################################
# Data Preprocessing
##################################
```


```python
##################################
# Data Exploration
##################################
```


```python
##################################
# Model Development
##################################
```


```python
##################################
# Overall Findings and Implications
##################################
```


```python
##################################
# Conclusions
##################################
```

# 3. References <a class="anchor" id="References"></a>
* **[Book]** [Data Preparation for Machine Learning: Data Cleaning, Feature Selection, and Data Transforms in Python](https://machinelearningmastery.com/data-preparation-for-machine-learning/) by Jason Brownlee
* **[Book]** [Feature Engineering and Selection: A Practical Approach for Predictive Models](http://www.feat.engineering/) by Max Kuhn and Kjell Johnson
* **[Book]** [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) by Alice Zheng and Amanda Casari
* **[Book]** [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) by Max Kuhn and Kjell Johnson
* **[Book]** [Data Mining: Practical Machine Learning Tools and Techniques](https://www.sciencedirect.com/book/9780123748560/data-mining-practical-machine-learning-tools-and-techniques?via=ihub=) by Ian Witten, Eibe Frank, Mark Hall and Christopher Pal 
* **[Book]** [Data Cleaning](https://dl.acm.org/doi/book/10.1145/3310205) by Ihab Ilyas and Xu Chu
* **[Book]** [Data Wrangling with Python](https://www.oreilly.com/library/view/data-wrangling-with/9781491948804/) by Jacqueline Kazil and Katharine Jarmul
* **[Book]** [Finding Groups in Data: An Introduction to Cluster Analysis](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316801) by Leonard Kaufman and Peter Rousseeuw
* **[Book]** [The Elements of Statistical Learning](https://link.springer.com/book/10.1007/978-0-387-84858-7) by Trevor Hastie, Robert Tibshirani and Jerome Friedman
* **[Book]** [Training Systems using Python Statistical Modeling](https://www.packtpub.com/product/training-systems-using-python-statistical-modeling/9781838823733) by Curtis Miller
* **[Book]** [Python Data Science Handbook](https://www.oreilly.com/library/view/python-data-science/9781098121211/) by Jake VanderPlas
* **[Book]** [Theory of Agglomerative Hierarchical Clustering](https://link.springer.com/book/10.1007/978-981-19-0420-2) by Sadaaki Miyamoto
* **[Python Library API]** [NumPy](https://numpy.org/doc/) by NumPy Team
* **[Python Library API]** [pandas](https://pandas.pydata.org/docs/) by Pandas Team
* **[Python Library API]** [seaborn](https://seaborn.pydata.org/) by Seaborn Team
* **[Python Library API]** [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) by MatPlotLib Team
* **[Python Library API]** [itertools](https://docs.python.org/3/library/itertools.html) by Python Team
* **[Python Library API]** [operator](https://docs.python.org/3/library/operator.html) by Python Team
* **[Python Library API]** [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) by Scikit-Learn Team
* **[Python Library API]** [sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.cluster](https://scikit-learn.org/stable/modules/clustering.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.mixture](https://scikit-learn.org/stable/modules/mixture.html) by Scikit-Learn Team
* **[Python Library API]** [SciPy](https://docs.scipy.org/doc/scipy/) by SciPy Team
* **[Python Library API]** [GeoPandas](https://geopandas.org/en/stable/docs.html) by GeroPandas Team
* **[Article]** [Step-by-Step Exploratory Data Analysis (EDA) using Python](https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/#:~:text=Exploratory%20Data%20Analysis%20(EDA)%20with,distributions%20using%20Python%20programming%20language.) by Malamahadevan Mahadevan (Analytics Vidhya)
* **[Article]** [Exploratory Data Analysis in Python — A Step-by-Step Process](https://towardsdatascience.com/exploratory-data-analysis-in-python-a-step-by-step-process-d0dfa6bf94ee) by Andrea D'Agostino (Towards Data Science)
* **[Article]** [Exploratory Data Analysis with Python](https://medium.com/@douglas.rochedo/exploratory-data-analysis-with-python-78b6c1d479cc) by Douglas Rocha (Medium)
* **[Article]** [4 Ways to Automate Exploratory Data Analysis (EDA) in Python](https://builtin.com/data-science/EDA-python) by Abdishakur Hassan (BuiltIn)
* **[Article]** [10 Things To Do When Conducting Your Exploratory Data Analysis (EDA)](https://www.analyticsvidhya.com) by Alifia Harmadi (Medium)
* **[Article]** [How to Handle Missing Data with Python](https://machinelearningmastery.com/handle-missing-data-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Statistical Imputation for Missing Values in Machine Learning](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Imputing Missing Data with Simple and Advanced Techniques](https://towardsdatascience.com/imputing-missing-data-with-simple-and-advanced-techniques-f5c7b157fb87) by Idil Ismiguzel (Towards Data Science)
* **[Article]** [Missing Data Imputation Approaches | How to handle missing values in Python](https://www.machinelearningplus.com/machine-learning/missing-data-imputation-how-to-handle-missing-values-in-python/) by Selva Prabhakaran (Machine Learning +)
* **[Article]** [Master The Skills Of Missing Data Imputation Techniques In Python(2022) And Be Successful](https://medium.com/analytics-vidhya/a-quick-guide-on-missing-data-imputation-techniques-in-python-2020-5410f3df1c1e) by Mrinal Walia (Analytics Vidhya)
* **[Article]** [How to Preprocess Data in Python](https://builtin.com/machine-learning/how-to-preprocess-data-python) by Afroz Chakure (BuiltIn)
* **[Article]** [Easy Guide To Data Preprocessing In Python](https://www.kdnuggets.com/2020/07/easy-guide-data-preprocessing-python.html) by Ahmad Anis (KDNuggets)
* **[Article]** [Data Preprocessing in Python](https://towardsdatascience.com/data-preprocessing-in-python-b52b652e37d5) by Tarun Gupta (Towards Data Science)
* **[Article]** [Data Preprocessing using Python](https://medium.com/@suneet.bhopal/data-preprocessing-using-python-1bfee9268fb3) by Suneet Jain (Medium)
* **[Article]** [Data Preprocessing in Python](https://medium.com/@abonia/data-preprocessing-in-python-1f90d95d44f4) by Abonia Sojasingarayar (Medium)
* **[Article]** [Data Preprocessing in Python](https://medium.datadriveninvestor.com/data-preprocessing-3cd01eefd438) by Afroz Chakure (Medium)
* **[Article]** [Detecting and Treating Outliers | Treating the Odd One Out!](https://www.analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/) by Harika Bonthu (Analytics Vidhya)
* **[Article]** [Outlier Treatment with Python](https://medium.com/analytics-vidhya/outlier-treatment-9bbe87384d02) by Sangita Yemulwar (Analytics Vidhya)
* **[Article]** [A Guide to Outlier Detection in Python](https://builtin.com/data-science/outlier-detection-python) by Sadrach Pierre (BuiltIn)
* **[Article]** [How To Find Outliers in Data Using Python (and How To Handle Them)](https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/) by Eric Kleppen (Career Foundry)
* **[Article]** [Statistics in Python — Collinearity and Multicollinearity](https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f) by Wei-Meng Lee (Towards Data Science)
* **[Article]** [Understanding Multicollinearity and How to Detect it in Python](https://towardsdatascience.com/everything-you-need-to-know-about-multicollinearity-2f21f082d6dc) by Terence Shin (Towards Data Science)
* **[Article]** [A Python Library to Remove Collinearity](https://www.yourdatateacher.com/2021/06/28/a-python-library-to-remove-collinearity/) by Gianluca Malato (Your Data Teacher)
* **[Article]** [8 Best Data Transformation in Pandas](https://ai.plainenglish.io/data-transformation-in-pandas-29b2b3c61b34) by Tirendaz AI (Medium)
* **[Article]** [Data Transformation Techniques with Python: Elevate Your Data Game!](https://medium.com/@siddharthverma.er.cse/data-transformation-techniques-with-python-elevate-your-data-game-21fcc7442cc2) by Siddharth Verma (Medium)
* **[Article]** [Data Scaling with Python](https://www.kdnuggets.com/2023/07/data-scaling-python.html) by Benjamin Obi Tayo (KDNuggets)
* **[Article]** [How to Use StandardScaler and MinMaxScaler Transforms in Python](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Feature Engineering: Scaling, Normalization, and Standardization](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/) by Aniruddha Bhandari  (Analytics Vidhya)
* **[Article]** [How to Normalize Data Using scikit-learn in Python](https://www.digitalocean.com/community/tutorials/normalize-data-in-python) by Jayant Verma (Digital Ocean)
* **[Article]** [What are Categorical Data Encoding Methods | Binary Encoding](https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/) by Shipra Saxena  (Analytics Vidhya)
* **[Article]** [Guide to Encoding Categorical Values in Python](https://pbpython.com/categorical-encoding.html) by Chris Moffitt (Practical Business Python)
* **[Article]** [Categorical Data Encoding Techniques in Python: A Complete Guide](https://soumenatta.medium.com/categorical-data-encoding-techniques-in-python-a-complete-guide-a913aae19a22) by Soumen Atta (Medium)
* **[Article]** [Categorical Feature Encoding Techniques](https://towardsdatascience.com/categorical-encoding-techniques-93ebd18e1f24) by Tara Boyle (Medium)
* **[Article]** [Ordinal and One-Hot Encodings for Categorical Data](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Hypothesis Testing with Python: Step by Step Hands-On Tutorial with Practical Examples](https://towardsdatascience.com/hypothesis-testing-with-python-step-by-step-hands-on-tutorial-with-practical-examples-e805975ea96e) by Ece Işık Polat (Towards Data Science)
* **[Article]** [17 Statistical Hypothesis Tests in Python (Cheat Sheet)](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [A Step-by-Step Guide to Hypothesis Testing in Python using Scipy](https://medium.com/@gabriel_renno/a-step-by-step-guide-to-hypothesis-testing-in-python-using-scipy-8eb5b696ab07) by Gabriel Rennó (Medium)
* **[Article]** [10 Clustering Algorithms With Python](https://machinelearningmastery.com/clustering-algorithms-with-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Elbow Method for Optimal Value of K in KMeans](https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/) by Geeks For Geeks Team (Geeks For Geeks)
* **[Article]** [How to Use the Elbow Method in Python to Find Optimal Clusters](https://www.statology.org/elbow-method-in-python/) by Statology Team (Statology)
* **[Article]** [Tutorial: How to Determine the Optimal Number of Clusters for K-Means Clustering](https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f) by Tola Alade (Cambridge Spark)
* **[Article]** [Optimizing Cluster Hyperparameters: Elbow and Silhouette Method](https://eightify.app/summary/data-science-and-analytics/optimizing-cluster-hyperparameters-elbow-silhouette-method) by Lucas Parisi (Eightify)
* **[Article]** [Clustering Metrics Better Than the Elbow Method](https://www.kdnuggets.com/2019/10/clustering-metrics-better-elbow-method.html) by Tirthajyoti Sarkar (KD Nuggets)
* **[Article]** [Practical Implementation Of K-means, Hierarchical, and DBSCAN Clustering On Dataset With Hyperparameter Optimization](https://medium.com/analytics-vidhya/practical-implementation-of-k-means-hierarchical-and-dbscan-clustering-on-dataset-with-bd7f3d13ef7f) by Janibasha Shaik (Towards Data Science)
* **[Article]** [KMeans Hyper-parameters Explained with Examples](https://towardsdatascience.com/kmeans-hyper-parameters-explained-with-examples-c93505820cd3) by Sujeewa Kumaratunga (Towards Data Science)
* **[Article]** [KMeans Silhouette Score Python Examples](https://vitalflux.com/kmeans-silhouette-score-explained-with-python-example/#google_vignette) by Ajitesh Kumar (Analytics Yogi)
* **[Publication]** [A Comparison of Document Clustering Techniques](https://www.semanticscholar.org/paper/A-Comparison-of-Document-Clustering-Techniques-Steinbach-Karypis/9378a3797d5f815babe7b392a199ea9d8d4f1dcf) by Michael Steinbach, George Karypis and Vipin Kumar (Computer Science)



```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>


***
