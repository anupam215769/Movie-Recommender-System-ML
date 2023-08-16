# Movie Recommender System Using Cosine Similarity

A Movie Recommender System based on content-based filtering using K-Nearest Neighbors – Cosine Similarity

## Screenshot

### Movie Recommender System (Content-Based) [Code](https://github.com/anupam215769/Movie-Recommender-System-ML/blob/main/movie-recommender-system.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Movie-Recommender-System-ML/blob/main/movie-recommender-system.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

![rec](https://i.imgur.com/qGAxNDi.png)

## How To Run (Graphical Interface/In Web Browser)

> Note - Install streamlit library before running the code

```
pip install streamlit
```

1. Download all the files and put them in a same folder

2. Open app.py using any python compiler

3. Run the app.py

4. Then type `streamlit run app.py` in the terminal

5. This project will open in your web browser (as shown in the screenshot above)


## How To Run (In Jupyter Notebook)

> Note - Install Jupyter Notebook

```
pip install jupyter-lab
```

1. Open Jupyter Notebook using CMD by typing `jupyter-lab`

2. Now, locate the folder of this project

3. Open `movie-recommender-system.ipynb` and run all the cells

4. At the last line of the code, put movie name in the brackets `recommend('movie_name')`

5. You will get the recommended movies for the given movie






## Theory Behind The System

Recommender System is a system that seeks to predict or filter preferences according to the user’s choices. Recommender systems are utilized in a variety of areas including movies, music, news, books, research articles, search queries, social tags, and products in general. 
Recommender systems produce a list of recommendations in any of the two ways –

- **Collaborative filtering:** Collaborative filtering approaches build a model from the user’s past behavior (i.e. items purchased or searched by the user) as well as similar decisions made by other users. This model is then used to predict items (or ratings for items) that users may have an interest in.
- **Content-based filtering:** Content-based filtering approaches uses a series of discrete characteristics of an item in order to recommend additional items with similar properties. Content-based filtering methods are totally based on a description of the item and a profile of the user’s preferences. It recommends items based on the user’s past preferences.

![comparison](https://i.imgur.com/09y3k9S.png)



## Content-Based Filtering

Content-based filtering uses item features to recommend other items similar to what the user likes, based on their previous actions or explicit feedback.

To demonstrate content-based filtering, let’s hand-engineer some features for the Google Play store. The following figure shows a feature matrix where each row represents an app and each column represents a feature. Features could include categories (such as Education, Casual, Health), the publisher of the app, and many others. To simplify, assume this feature matrix is binary: a non-zero value means the app has that feature.

You also represent the user in the same feature space. Some of the user-related features could be explicitly provided by the user. For example, a user selects "Entertainment apps" in their profile. Other features can be implicit, based on the apps they have previously installed. For example, the user installed another app published by Science R Us.

The model should recommend items relevant to this user. To do so, you must first pick a similarity metric (for example, dot product). Then, you must set up the system to score each candidate item according to this similarity metric. Note that the recommendations are specific to this user, as the model did not use any information about other users.

![matrix](https://i.imgur.com/SEl4fQE.jpg)


<h3 class="hide-from-toc" id="using-dot-product-as-a-similarity-measure" data-text="Using Dot Product as a Similarity Measure">Using Dot Product as a Similarity Measure</h3>

Consider the case where the user embedding `x` and the app embedding `y` are both binary vectors. Since <img src="https://i.imgur.com/O3m5gUq.png" width="120" height="20" /> , a feature appearing in both `x` and `y` contributes a 1 to the sum. In other words, `<x, y>` is the number of features that are active in both vectors simultaneously. A high dot product then indicates more common features, thus a higher similarity.


## Cosine Similarity

Recommendation Systems work based on the similarity between either the content or the users who access the content.
There are several ways to measure the similarity between two items. The recommendation systems use this similarity matrix to recommend the next most similar product to the user.
This Machine Learning model is based on Cosine Similarity.

**Cosine similarity** is a measure of similarity between two non-zero vectors of an inner product space. It is defined to equal the cosine of the angle between them, which is also the same as the inner product of the same vectors normalized to both have length 1.

### Example

Suppose I want to check if Bernard and Clarissa have similar movie preferences, and I only have two movie reviews. The reviews are scores from 1 to 5, where 5 is the best score and 1 the worst, and 0 means that a person has not watched the movie.

![eg](https://miro.medium.com/max/535/1*xBpc0BFUMW_8vIoplpM94w.png)

We can represent each person’s reviews in a separate vector.

![eg](https://miro.medium.com/max/284/1*0MZ-kl2jsa1SH1r9J7yfCg.png)

The cosine similarity will measure the similarity between these two vectors which is a measurement of how similar are the preferences between these two people.

![eg](https://miro.medium.com/max/690/1*kDWCM-8qopE2ekudxlOudw.png)

In the image, below each vector represents a person’s preferences and they have an angle θ between them. Similar vectors will have a lower angle θ, and dissimilar vectors (different film preferences) will have bigger θ.

![eg](https://miro.medium.com/max/875/1*JdXBbKlOKS9UNFpLNchfdA.png)

In the example above the similarity 0.989 is close to the maximum value of 1, this means that given only two movie reviews the two users have similar preferences.

Theoretically, the cosine similarity can be any number between -1 and +1 because of the [image](https://en.wikipedia.org/wiki/Image_(mathematics)) of the cosine function, but in this case, there will not be any negative movie rating so the angle θ will be between 0º and 90º bounding the cosine similarity between 0 and 1. If the angle θ = 0º =>cosine similarity = 1, if θ = 90º => cosine similarity =0.


The cosine similarity can be calculated for more than 2 movies. In the example below I will add the ratings for a movie that Bernard liked and Clarissa disliked this should decrease cosine similarity value.

![eg](https://miro.medium.com/max/714/1*d-qq_Ee_rUpQZfN3N28xYA.png)

The new vectors are:

![eg](https://miro.medium.com/max/479/1*lv9PosfMD9fUZC-3IgI-EQ.png)

The plot has 3 dimensions now:

![eg](https://miro.medium.com/max/645/1*JDlFeJM-Z3VQFprHGBTd9A.png)

Calculating the similarity:

![eg](https://miro.medium.com/max/463/1*VJE2Yi7S-o1x_-xi7GnWRQ.png)

The similarity has reduced from 0.989 to 0.792 due to the difference in ratings of the District 9 movie. The cosine can also be calculated in Python using the Sklearn library.




    
    
