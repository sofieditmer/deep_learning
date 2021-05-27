# Assignment 4: Text Classification Using Deep Learning

### Description of Task: Classifying Game of Thrones Seasons Based on Dialogue <br>
This assignment was assigned by the course instructor as “Assignment 6 – Text Classification Using Deep Learning”. The purpose of this assignment was to see how successfully we were able to use a CNN model to classify scripts from the TV series Game of Thrones, a dataset available on [Kaggle](https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons). Hence, the task was to see how accurately we were able to model the relationship between each season and its manuscript. This task involved first building a baseline model using classical machine learning with CountVectorization and LogisticRegression. This baseline model should then be used to evaluate the performance of the more complex deep learning CNN model. <br>
In addition to the compulsory requirements, I also chose to implement grid search in the building of logistic regression model to estimate the most optimal hyperparameters. 

### Content and Repository Structure <br>
If the user wishes to engage with the code and reproduce the obtained results, this section includes the necessary instructions to do so. It is important to remark that all the code that has been produced has only been tested in Linux and MacOS. Hence, for the sake of convenience, I recommend using a similar environment to avoid potential problems. <br>
The repository follows the overall structure presented below. The two scripts, ```lr-got.py``` and ```cnn-got.py```, are located in the ```src``` folder. The full dataset is provided in the ```data``` folder, and the outputs produced when running the scripts can be found within the ```output``` folder. Utility functions are located in the ```utils``` folder. The README file contains a detailed run-through of how to engage with the code and reproduce the contents.

| Folder | Description|
|--------|:-----------|
| ```data``` | A folder containing a the full dataset on which the classification was performed.
| ```src``` | A folder containing the python scripts for the particular assignment.
| ```output``` | A folder containing the outputs produced when running the python scripts within the src folder.
| ```utils``` | A folder containing utility functions used within the main scripts.
| ```requirements.txt```| A file containing the dependencies necessary to run the python script.
| ```create_GoT_venv.sh```| A bash-file that creates a virtual environment in which the necessary dependencies listed in the ```requirements.txt``` are installed. This script should be run from the command line.
| ```LICENSE``` | A file declaring the license type of the repository.

### Usage and Technicalities <br>
To reproduce the results of this assignment, the user will have to create their own version of the repository by cloning it from GitHub. This is done by executing the following from the command line: 

```
$ git clone https://github.com/sofieditmer/deep_learning.git  
```

Once the user has cloned the repository, a virtual environment must be set up in which the relevant dependencies can be installed. To set up the virtual environment and install the relevant dependencies, a bash-script is provided, which creates a virtual environment and installs the dependencies listed in the ```requirements.txt``` file when executed. To run the bash-script that sets up the virtual environment and installs the relevant dependencies, the user must first navigate to the topic modeling repository:

```
$ cd deep_learning
$ bash create_GoT_venv.sh 
```

Once the virtual environment has been set up and the relevant dependencies listed in ```requirements.txt``` have been installed within it, the user is now able to run the scripts provided in the ```src``` folder directly from the command line. In order to run the script, the user must first activate the virtual environment in which the script can be run. Activating the virtual environment is done as follows:

```
$ source GoT_venv/bin/activate
```

Once the virtual environment has been activated, the user is now able to run the two scripts within it:

```
(GoT_venv) $ cd src

(GoT_venv) $ python lr-got.py

(GoT_venv) $ python cnn-got.py
```

For the ```lr-got.py``` script the user is able to modify the following parameters, however, this is not compulsory:

```
-i, --input_data: str <name-of-input-data>, default = "game_of_thrones_script.csv"
-o, --main_output_filename: str <name-of-output-file>, default = "lr_classification_report.txt"
-c, --chunk_size: int <size-of-chunks>, default = 10
-t, --test_size: float <size-of-test-split>, default = 0.25
````

For the ```cnn-got.py``` script the user is able to modify the following parameters, however, once again this is not compulsory:

```
-i, --input_data: str <name-of-input-data>, default = "game_of_thrones_script.csv"
-c, --chunk_size: int <size-of-chunks>, default = 2
-e, --n_epochs: int <number-of-epochs>, default = 10
-ts, --test_size: float <size-of-test-split>, default = 0.25
-em, --embedding_dim: int <embedding-dimensions>, default = 100
-r, --regularization_value: float <regularization-value>, default = 0.001
-d, --dropout_value: float <likelihood-of-dropping-nodes>, default = 0.01
-o, --optimizer: str <choice-of-optimizer-algorithm>, default = “adam”
-n, --n_words: int <number-of-words-to-initialize-tokenizer>, default = 5000
```

The abovementioned parameters allow the user to adjust the analysis of the input data, if necessary, but default parameters have been set making the script run without explicitly specifying these arguments. The user is able to modify the chunk size, i.e., how many sentences to chunk together, the number of training epochs, the size of the test-split, the number of word embedding dimensions to use, the regularization strength, the likelihood of dropping nodes in the dropout layers of the model, the optimizer algorithm, and lastly the number of words to use when initializing the tokenizer. 

### Output <br>
When running the ```lr-got.py``` script, the following files will be saved in the ```output``` folder: 
1. ```lr_classification_report.txt``` Classification report of the logistic regression classifier.
2. ```lr_heatmap.png``` Normalized heatmap displaying an overview of the performance of the logistic regression classifier.
3. ```lr_cross_validation_results.png``` Cross-validation results obtained by the logistic regression classifier.

When running the ```cnn-got.py``` script, the following files will be saved in the ```output``` folder: 
1. ```cnn_model_summary.txt``` Summary of the CNN model architecture.
2. ```cnn_classification_metrics.txt``` Classificaiton report of the model.
3. ```cnn_model_history.png``` Loss/accuracy history of model.

### Discussion of Results <br>
The initial Logistic Regression baseline classifier obtained a weighted accuracy of 42% (see figure 1). The accuracy of the model fluctuates substantially depending on the season. For instance, for season 7 the model obtains an accuracy of 70% while on season 8 it obtains an accuracy of 17%. This is most likely due to the amount of data available for season 1 compared to season 8. Hence, when the model is fed more data, it is able to make more accurate results. One could have balanced the data to have an equal number of cases for each season, however, as the amount of data available was very limited to begin with I chose to not balance the data. This is of course with the risk of the model overfitting the data and not being able to generalize well. I tried to approach this problem by tokenizing the lines and combining them into batches of sentences, however, it still seems that particular seasons have more lines than others. Hence, I was aware of the trade-off between having a balanced dataset and a model that seems to overfit.

<img src="https://github.com/sofieditmer/deep_learning/blob/main/output/lr_classification_report.txt" width="300">
Figure 1: Logistic regression classification report. <br> <br>

The normalized heatmap displays the predicted classes made by the model and the actual classes (see figure 2). For season 7, the model is able to correctly predict almost 70% of the cases, while for season 8 the model only correctly predicts 10%.  What is also interesting is that the model tends to confuse season 4 with season 3, which is illustrated by the fact that it predicts 33% of the cases of season 4 to be season 3. This suggests that season 4 and season 3 might be similar in terms of dialogue. 
 
<img src="https://github.com/sofieditmer/deep_learning/blob/main/output/lr_heatmap.png" width="500">
Figure 2: Normalized heatmap. <br> <br>

When assessing the cross-validation results it becomes clear that there is a problem with overfitting (see figure 3). While the cross-validation score increases, the training score remains stable around 1 which is a clear sign of overfitting. Ideally, we would want the gap between the cross-validation curve and the training score curve to be smaller, and the training score to be below 100%. When assessing the scalability of the model, we can see that when adding more data, the accuracy increases. The plot of the scalability of the model also tells us that the more data the model the model is trained on, the longer it takes to fit the model, which gives us an insight into the training time of the model. <br>
Overall, the model is performing reasonably well, however, there is a problem with overfitting the data. Increasing the amount of data would potentially alleviate this issue.  

<img src="https://github.com/sofieditmer/deep_learning/blob/main/output/lr_cross_validation_results.png" width="1000">
Figure 3: Cross-validation results  <br> <br>

When comparing the baseline logistic regression classifier to the more complex deep learning CNN model, the deep learning CNN model obtained a weighted accuracy of 25% (see figure 4). Thus, it seems that the logistic regression classifier performs better at predicting the seasons based on the dialogue, which might be due to the fact that the logistic regression model uses a count vectorizer, which provides it with an advantage over the CNN model. Nevertheless, I would have expected the deep learning model to perform better than the logistic regression model, given that it is able to capture word context. 
 
<img src="https://github.com/sofieditmer/deep_learning/blob/main/output/cnn_classification_metrics.txt" width="300">
Figure 4: CNN model classification report. <br> <br>

It would be interesting to see whether these results can be reproduced for other series as well. Perhaps predicting season based on dialogue is a complex task no matter the series in question, or perhaps there is something specific about Game of Thrones that makes predicting the season based on the dialogue particularly difficult. Perhaps the dialogue across seasons in Game of Thrones is very alike which would explain why the model has a hard time when trying to distinguish them. <br>
When assessing the loss and accuracy curves of the CNN model for the training and validation data respectively, there are once again clear signs of overfitting (see figure 5) While the training loss decreases, the validation loss increases, which suggests that the model is overfitting the training data and as a result it is not generalizing well to the validation data. Similarly, the training accuracy is increasing, while the validation accuracy remains stable which also suggests overfitting.

<img src="https://github.com/sofieditmer/deep_learning/blob/main/output/cnn_model_history.png" width="500">
Figure 5: CNN model history. <br> <br>


### License <br>
This project is licensed under the MIT License - see the [LICENSE](https://github.com/sofieditmer/deep_learning/blob/main/LICENSE) file for details.

### Contact Details <br>
If you have any questions feel free to contact me on [201805308@post.au.dk](201805308@post.au.dk)
