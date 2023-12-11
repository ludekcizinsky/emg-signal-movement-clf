In this mini project, we are supposed to do the following tasks:

- [X] Visualize and preprocess the data. 

- [X] Split the data into training, validation, and testing sets for each subject. Why do we need the different datasets?

- [X] Perform sliding windows (choose a reasonable window width and sliding step) and explain your choice.

- [X] Extract the same set of features across 10 different subjects.

- [X] Look at the typical values of those features across the same set of movements for different subjects. What do you see? Are there any regularities between the different subjects? What are some possible reasons for similarity/dissimilarity?

- [X] Perform classification (use a method of your choice) on different subjects separately.

- [X] Perform analysis to determine the importance of the features to the classification. Compare this ranking across the different subjects. Are the features stable?

- [ ] Train a classification model on a set of subjects and test it on a subject that does not
belong to that set. Evaluate the performance. How does it compare to training on that
subject and testing on the same subject directly?

- [ ] Repeat the training by varying the number of subjects in the training set. Discuss how the
number of subjects used in the training set could affect the classification
performance.

---

(Optional - was not part of the assigment):

- [ ] Do people with higher BMI have higher EMG?

> Furthermore, regression analysis on the classification results reveals that classification accuracy is negatively correlated with a subject’s Body Mass Index (BMI).

- [ ] In the original paper, they use the boxplots to measure variability of features accross subjects --> maybe use the boxplots instead (already have those), In addition, they also looked at the boxplot of the 12 stimulus accross repetitions to asses the variability. They also did sEMG amplitude as a function of repetition and observed that the amplitude decreases with repetition. --> these three plots we could have instead of the one big plot we have now and kinda showcase the limitations/challenges of the data.

- [ ] while the movement label is defined as the movement type at the moment of the most recent sample within a window --> start of a window

- [ ] Subsampling the training set by a factor of 10 
