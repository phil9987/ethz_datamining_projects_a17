# Data Mining Autumn 2017

## Team
Etienne De Stoutz
Philip Junker
Valentin Venzin

## Task1 description
In this project, we are interested in detecting near-duplicate html pages. Suppose you want to write your own search engine. For reliability reasons, many web pages store duplicates or near duplicate versions of the same information. Your task is to develop an efficient method to to detect whether a pages are near-duplicates of each other so that your search engine does not present the user redundant search results.

### Input and Output Specification
We will use Jaccard similarity based on the page features as the similarity metric. You are given two text files:

    handout_shingles.txt: Each line contains the features for one page file and is formatted as follows: PAGE_XXXXXXXXX followed by a list of space delimited integers in range [0, 8192]. You can consider them equivalent to shingles in the context of near-duplicate document retrieval.
    handout_duplicates.txt: Each line contains a pair of near duplicates (pages that are at least 85% similar according to the Jaccard similarity). Each line is a tab-delimited pair of integers where the first integer is always smaller. This file is used to measure the error of your output as described below.

Your goal is to develop a Locality Sensitive Hashing program in the MapReduce setting. To facilitate development, we provide you with a (small) MapReduce implementation: The runner.py script allows you to run a MapReduce program and measure the performance of the produced solutions.

To create a valid MapReduce program for this task, you need to create a Python source file that contains both a mapper and a reducer function. The mapper(key, value) function takes as input a (key, value) tuple where key is None and value is a string. It should yield (key, value) pairs. The reducer(key, value) function takes as input a key and a list of values. It should yield (key, value) pairs. A skeleton of such a function is provided in the example.py.

You are free to implement the mapper and reducer in any way you see fit, as long as the following holds:

    The maximum number of hash functions per mapper used is 1024.
    The cummulative runtime of both mappers and reducers is limited to 3 minutes.
    Each mapper receives a key and value pair where key is always None (for consistency). Each value is one line of input as described above.
    Reducer should output a key, value pair of two integers representing the ID's of duplicated pages, the smaller ID being the key and the larger the value.
    You may use the Python 2.7 standard library and NumPy. You are not allowed to use multithreading, multiprocessing, networking, files and sockets.

### Evaluation and Grading
For each line of the output of your algorithm we check whether the reported pages are in fact at least 85% similar. If they are, this output will count as one true positive. If they are not, it will count as one false positive. In addition, each pair of approximate neighbors that was not reported by your algorithm will count as one false negative. Given the number of true positives, false positives and false negatives, TP, FP and FN respectively, we can calculate:

    Precision, also referred to as Positive predictive value (PPV), as P = TP/(TP+FP).
    Recall, also referred to as the True Positive Rate or Sensitivity, as R = TP/(TP+FN).

Given precision and recall we will calculate the F1 score defined as F1 = 2PR/(P + R). We will compare the F1 score of your submission to two baseline solutions: A weak one (called baseline easy) and a strong one (baseline hard). These will have the F1 score of FBE and FBH respectively, calculated as described above. Both baselines will appear in the rankings together with the F1 score of your solutions.

Your grade on this task depends on the solution and the description that you hand in. As a rough (non-binding) guidance, if you hand in a properly-written description and your handed-in submission performs better than the easy baseline, you will obtain a grade exceeding a 4. If in addition, your submission beats the hard baseline, you obtain a 6.

## Task2 description
In this project your task is to classify images in one of two classes according to their visual content. We will provide a labeled dataset containing two sets of images: Portraits and Landscapes.

### Dataset
A set of 400 features has been extracted from each picture. We provide 16k training images (handout_train.txt) and 4k testing images (handout_test.txt), sampled rougly at the same frequency from both categories. We only provide the features for each image, from which the actual image cannot be reconstructed. Each line in the files corresponds to one image and is formatted as follows:

    Elements are space separated.
    The first element in the line is the class y {+1,-1} which correspond to Portrait and Landscape class, respectively.
    The next 400 elements are real numbers which represent the feature values x0... x399.

### Task
The goal is to solve this classification problem using Parallel Stochastic Gradient Descent. To facilitate development, we provide you with a (small) MapReduce implementation for UNIX: The runner.py script allows you to run a MapReduce program and measure the performance of the produced solutions.

To create a valid MapReduce program for this task, you need to create a Python source file that contains both a mapper and a reducer function. The mapper(key, value) function takes as input a (key, value) tuple where key is None and value is a list of strings. It should yield (key, value) pairs. The reducer(key, value) function takes as input a key and a list of values. It should yield a 1D NumPy array. A skeleton of such a function is provided in the example.py.

You are free to implement the mapper and reducer in any way you see fit, as long as the following holds:

    The cumulative runtime of both mappers, reducers and evaluation is limited to 5 minutes.
    The cumulative memory limit is 1 GB.
    Each mapper receives a key and value pair where key is always None (for consistency). Each value is a 2D NumPy array representing the subset of images passed to the mapper.
    There will be one reducer process. All mappers should output the same key.
    The reducer should output the weight vector that we will use to perform predictions as described below.
    You may use the Python 2.7 standard library and NumPy. You are not allowed to use multithreading, multiprocessing, networking, files and sockets. In particular, you are not allowed to use the scikit-learn library.

### Evaluation and Grading
The prediction of your model on a test instance x will be calculated as y' = sgn(<w, x>). If you decide to apply any transformation t to the given features your predictions will be given by y = sgn(<w, t(x)>). If you apply transformations to the original features you have to implement a transform function in the given template. It is important that the transform function is called from the mapper, otherwise we will not be able to transform the evaluation data using the same function and evaluate your submission. The transform function must work with both vectors and 2D Numpy arrays.

Based on the predictions we will calculate the predictive accuracy as (TP + TN)/(TP + TN + FP + FN) where TP, TN, FP, FN are the number of true positives, true negatives, false positives and false negatives, respectively.

We will compare the score of your submission to two baseline solutions: A weak one (called baseline easy) and a strong one (baseline hard). These will have the accuracy of FBE and FBH respectively, calculated as described above. Both baselines will appear in the rankings together with the score of your solutions.

Your grade on this task depends on the solution and the description that you hand in. As a rough (non-binding) guidance, if you hand in a properly-written description and your handed-in submission performs better than the easy baseline, you will obtain a grade exceeding a 4. If in addition your submission performs better than the hard baseline, you obtain a 6.

For this task, each submission will be evaluated on two datasets: a public data set for which you will see the score in the leaderboard and a private data set for which the score is kept private until hand in. Both the public and the private score of the handed in submission are used in the final grading. 

## Task3 description
The goal of this project is to extract representative elements from a large image data set. The quality of the selected set of points is measured by the sum of squared distances from each point of the dataset to the closest point in the selected set. For details check the handout below.

### Dataset
In the original representation, each image was an integer vector of dimension 3072 (32 * 32 * 3, intensity is computed for each pixel). We have performed mean normalization, feature scaling, dimensionality reduction with PCA, as well as whitening. We then extracted a subset from that dataset which contains 27K images, each being a 250 dimensional feature vector. In addition, we provide you with a subset of those 27K images for testing. The dataset has been serialized to the npy format which enables efficient loading. The conversion is done for you and you may assume that the value in the mapper will be a 2D NumPy array. Further feature transformations are not allowed.
Task

To create a valid MapReduce program for this task, you need to create a Python source file that contains both a mapper and a reducer function. The mapper(key, value) function takes as input a (key, value) tuple where key is None and value is a NumPy array. It should yield (key, value) pairs. The reducer(key, value) function takes as input a key and a NumPy array that contains the concatenation of the values emitted by the mappers. It should yield (key, value) pairs. A skeleton of such a function is provided in the example.py.

You are free to implement the mapper and reducer in any way you see fit, as long as the following holds:

    The cummulative runtime of mappers, reducers and evaluation is limited to 5 minutes.
    The cummulative memory limit is 4 GB.
    Each mapper receives a key and value pair where key is always None (for consistency). Each value is a 2D NumPy array representing the subset of images passed to the mapper.
    There will be one reducer process. All mappers should output the same key.
    Reducer should output a 2D NumPy array containing 200 vectors representing the selected centers (each being 250 floats).
    You may use the Python 2.7 standard library and both NumPy and SciPy libraries. You are not allowed to use multithreading, multiprocessing, networking, files and sockets. In particular, you are not allowed to use the scikit-learn library.

### Evaluation and Grading
To evaluate the quality of the returned set we will use the normalized quantization error: average squared distance from each point of the dataset to the closest point in the returned set.

We will compare the score of your submission to two baseline solutions: A weak one (called baseline easy) and a strong one (baseline hard). These will have the quantization error of FBE and FBH respectively, calculated as described above. Both baselines will appear in the rankings together with the score of your solutions.

Your grade on this task depends on the solution and the description that you hand in. As a rough (non-binding) guidance, if you hand in a properly-written description and your handed-in submission performs better than the easy baseline, you will obtain a grade exceeding a 4. If in addition your submission performs better than the hard baseline, you obtain a 6.

For this task, each submission will be evaluated on two datasets: a public data set for which you will see the score in the leaderboard and a private data set for which the score is kept private until hand in. Both the public and the private score of the handed in submission are used in the final grading. 

## Taks4 description
The goal of this task is to learn a policy that explores and exploits among available choices in order to learn user preferences and recommend news articles to users. For this task we will use real-world log data shared by Yahoo!. The data was collected over a 10 day period and consists of 45 million log lines that capture the interaction between user visits and 271 news articles, one of which was randomly displayed for every user visit. In each round, you are given a user context and a list of available articles for recommendation.

Your task is to then select one article from this pool such to maximize the click-through rate (clicks/impressions). If the article you selected matches the one displayed to the user (in the log file), then your policy is evaluated for that log line. Otherwise, the line is discarded. Since the articles were displayed uniformly at random during the data collection phase, there is approximately 1 in 20 chance that any given line will be evaluated.
Dataset Description

In the handout for this project, you will find the 100000 lines of log data webscope-logs.txt, where each line is formated as follows:

    timestamp: integer
    user features: 6 dimensional vector of doubles
    available articles: list of article IDs that are available for selection

In addition, you are given access to webscope-articles.txt, where each line is formated as follows:

    ArticleID feature1 feature2 feature3 feature4 feature5 feature6

To evaluate your policy you can use (policy.py and runner.py). Your task is to complete the functions recommend, update and set_articles in the policy file. We will first call the set_articles method and pass in all the article features. Then for every line in in webscope-logs.txt the provided runner will call your recommend function. If your chosen article matches the displayed article in the log, the result of choosing this article (click or no click) is fed back to your policy and you have a chance to update the current model accordingly. This is achieved via the update method.

You are free to implement the policy in any way you see fit, as long as the following holds:

    Your policy always returns an article from the provided list. Failing to do so will result in an exception and the execution will then be aborted.
    The cummulative runtime is 30 minutes.
    The cummulative memory limit is 4 GB.

### Evaluation and Grading

An iteration is evaluated only if you have chosen the same article as the one chosen by the random policy used to generate the data set. Only those selections of your policy that match the log line are counted as an impression. Based on the number of impressions and clicks we calculate the click-through rate.

We will compare the score of your submission to two baseline solutions: A weak one (called baseline easy) and a strong one (baseline hard). These will have the quantization error of FBE and FBH respectively, calculated as described above. Both baselines will appear in the rankings together with the score of your solutions.

Your grade on this task depends on the solution and the description that you hand in. As a rough (non-binding) guidance, if you hand in a properly-written description and your handed-in submission performs better than the easy baseline, you will obtain a grade exceeding a 4. If in addition your submission performs better than the hard baseline, you obtain a 6.