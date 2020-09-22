# Machine-Learning  <img src="/Resources/AI.gif" width="40" height="35"/>
 
In this repository you will find all the work which I have done in Machine Learning.
I have structured the repository around the datasets, instead of machine learning topics. 

Please go through some of the concepts/techniques I have used

<details><summary><h1>Initial Exploration</h1></summary>
    
- Basic statistical analysis for different features
- Distribution for all the features
- Visualizing the data using `matplotlib`
<br/>
</details>

<details><summary><h1>Data Cleansing</h1></summary>

- Impute the numerical values using `sklearn.impute.SimpleImputer`
- Add new features using my custom class, `addAdditionalAttributes`
- Scale the numerical values using `sklearn.preprocessing.StandardScalar`
- Encode the categorical values using `sklearn.preprocessing.OneHotEncoder`
</details>

<details><summary><h1>Data Preparation</h1></summary>

- Split the dataset, using `sklearn.model_selection.train_test_split`
- Split the dataset into Training and Test based on a column, using `sklearn.model_selection.StratifiedShuffleSplit`
</details>

<details><summary><h1>Data Transformation</h1></summary>

- Created numerical and categorical pipeline, using `skelarn.pipeline.Pipeline`
- Combined both the pipelines into one, using `skelarn.compose.ColumnTransformer`
</details>

<details><summary><h1>Machine Learning Algorithms</h1></summary>
</br>
<details><summary><h4>Simple Regression</h4></summary>

- Linear Regression, using `sklearn.linear_model_LinearRegression`
- Decision Tree Regression, using `sklearn.tree.DecisionTreeRegressor`
- Random Forest Regression, using `sklearn.ensemble.RandomForestRegressor`
</details>

<details><summary><h4>Simple Classification</h4></summary>
    
- Binary, Multi Label and Multi Output Classifiers
- Random Forest Classification, using `sklearn.ensemble.RandomForestClassifier`
- SVC Classification, using `sklearn.svm.SVC`
- SGD Classification, using `sklearn.linear_model.SGDClassifier`
- One versus One Classification, using `sklear.multicall.OneVsOneClassifier`
- KNeighbors Classification, using `sklearn.neighbors.KNeighborsClassifier`
</details>

</details>

<details><summary><h1>Evaluation Methods</h1></summary>

- Root Mean Square Error _RMSE_, using `sklearn.metrics.mean_squared_error`
- Cross-Validation, using `sklearn.model_selection.cross_val_score`
- Confusion Matrix, using `sklearn.metrics.confusion_matrix`
- Precision Score, using `sklearn.metrics.precision_score`
- Recall Score, using `sklearn.metrics.recall_score`
- F1 Score Score, using `sklearn.metrics.f1_score`
- Precision Recall Curve, using `sklearn.metrics.precision_recall_curve`
- ROC Curve, using `sklearn.metrics.roc_curve`
- ROC AUC Score, using `sklearn.metrics.roc_auc_score`
</details>

<details><summary><h1>Fine Tuning the Model</h1></summary>

- Grid search, using `sklearn.model_selection.GridSearchCV`
- Randomized Grid search, using `sklearn.model_selection.RandomizedSearchCV`
</details>

<details><summary><h1>Additional Topics</h1></summary>

- Combined Data Cleansing, Tranformation, Machine Learning steps into a single pipeline
- Used Grid Search to fine tune the data cleansing steps
</details>

<details><summary><h1>Datasets used</h1></summary>

- California House Prices
- MNIST Dataset
- Titanic Dataset
</details>

<details><summary><h1>Libraries used</h1></summary>

- sklearn
- numpy
- scipy
- pandas
</details>

<details><summary><h1>Futue Updates</h1></summary>

- Will dive deeper into specific machine learning algorithms and learn out various hyperparameters
</details>

<details><summary><h1>Note</h1></summary>

I am following [Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/). In my opinion, it is one of the best books I have come across for understand and learning Machine Learning, given you have some base knowledge and understanding about Python.
</details>

<!-- ### [About me](https://preetparmar.github.io/) -->
<details><summary><h1>[About me](https://preetparmar.github.io/)</h1></summary>

I am a beginner in Machine Learning and always learning new things in python. Feel free to reach out with any suggestions, questions or just to say hi!
Also, look at my other repositories to see some of the projects I have worked on.
You can find my portfolio [here](https://preetparmar.github.io/)
</details>