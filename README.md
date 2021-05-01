# Vowel Classifier
Portfolio For Computational Physics

By: Spencer Peters and Anthony Demattos

### Overview 
This project uses a convolutional nerual network that is capable of determining what vowel a person says. It will transform waveform audio files into mel spectograms which esentially turns it into a computer vision problem rather than audio. The model will be trained on one second audio clips collected from various people. 

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
