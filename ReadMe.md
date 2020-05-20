# ReadMe

The data that was used for this project is [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). It was provided by Udacity.

## Actions Performed

The image classifier Jupyter Notebook code is split into this format:
1. Loading the Data
2. Mapping the labels for the flowers to their actual name.
3. Building and training the classifier (I used densenet121 but you could use other classifiers as well).
4. Testing the trained model.
5. Saving a checkpoint and loading it again.
6. Processing an image and outputting its' image and predicted probabilities for the top 5 guesses.

After the Jupyter Notebook was made, I created train.py and predict.py

These files contain essentially the same code that is in the Jupyter notebook, but the code is formatted into functions and contains a parser which allows users to set their own hyperparameters for different variables.

## Results

The model had a 87.4% accuracy rating.
