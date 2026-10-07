lab 1

The goal was to set up the environment and test two models on the Breast Cancer dataset.

I used a 70/30 train-test split with random_state=42. I tested Logistic Regression and Random Forest and measured accuracy, training time, inference time, model size and memory usage.

Logistic Regression had an accuracy of 0.9415. Training took 0.4140 s, inference took 0.0638 ms, and the model size was 1.03 KB.

Random Forest had an accuracy of 0.9357. Training took 0.2068 s, inference took 2.6315 ms, and the model size was 284.07 KB.

Both models fit the Edge requirements.

Logistic Regression had slightly better accuracy. Random Forest was faster to train but was much larger. Both models can be used on Edge.
