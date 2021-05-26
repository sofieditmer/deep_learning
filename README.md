## Assignment 6 - Text classification using Deep Learning

__Task__ <br>
See how accurately you can model the relationship between each Game of Throne season and the lines spoken. Can you predict which season a line comes from? Or to phrase that another way, is dialogue a good predictor of season? Start by making a baseline using a 'classical' ML solution such as CountVectorization + LogisticRegression and use this as a means of evaluating how well your model performs. Then you should try to come up with a solution which uses a DL model, such as the CNNs we went over in class.

__Data__ <br>
You can find the data here: https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons. The data is also provided in the data folder.

__Running the script__ <br>
1. Clone the repository
```
git clone https://github.com/sofieditmer/cds-language.git cds-language-sd
```

2. Navigate to the newly created directory
```
cd cds-language-sd/assignments/assignment6_CNNs
```

3. Create and activate virtual environment, "ass6", by running the bash script create_ass6_venv.sh. This will install the required dependencies listed in requirements.txt 

```
bash create_ass6_venv.sh
source ass6/bin/activate
```

4. Now you have activated the virtual environment in which you can run scripts *logistic_regression.py* and *cnn_model*. First you need to navigate to the src folder in which the script is located.

```
cd src
```

Example: <br>
```
$ python logistic_regression.py
$ python cnn_model.py
```