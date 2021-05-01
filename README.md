# Vowel Classifier

## Abstract 
The goal of this project is to investigate the use of building a deep learning model that is capable of determining what vowel a person says. A focus will be put on transforming audio files into a way that allows a model to learn from audio data most efficently and effectively.

## Building the model

The vowelClassifier jupyter notebook will take you through my journey of building a model capable of classifying a vowel. 

## Results 

The trained model works with almost 100% accuracy on my voice, you can test the model yourself by using the inference jupyter notebook. I suggest recording your own data and retraining the model, about 20 samples per vowel should be more than enough.

### Directions

#### Record Data

1. Go to recordVowels.ipynb
2. Run import statements, if missing any libriries install them before continuing
3. Go to the code block that is one line and says record('a')
4. Run the code block when it says recording say the letter that is in the parameters of record()
5. To change the vowel being recorded change the parameters to the appriopriate letter

#### Train Nerual Network
1. Go to vowel_classifier.ipynb
2. Run all the code blocks
3. Once it is done training you are ready to either test at the bottom or inferance.ipynb
