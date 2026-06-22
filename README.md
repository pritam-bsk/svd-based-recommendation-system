# Movie Recommender System

This project implements a collaborative filtering-based movie recommender system using Singular Value Decomposition (SVD). The model is built and evaluated in the `main.ipynb` Jupyter Notebook.

## Project Overview

The recommender system learns latent features for users and movies from the MovieLens dataset. It can provide personalized movie recommendations for both existing users in the dataset and new users based on a small number of initial ratings (a "cold start" scenario).

### Key Features
- **Data Processing**: Loads and processes the MovieLens dataset (`ml-32m`) into a sparse user-item rating matrix.
- **SVD Model**: Uses Truncated SVD to decompose the ratings matrix and generate user and movie embeddings.
- **Recommendation**: 
    - Recommends movies for existing users.
    - Handles the "cold start" problem by recommending movies to new users based on their initial ratings.
- **Evaluation**: The model's performance is evaluated by plotting the singular values, cumulative explained variance, and calculating the Root Mean Squared Error (RMSE) for different numbers of latent factors (k).
- **Model Export**: The trained movie embeddings are saved to `recommender_model.npz` for later use.

## Model Evaluation

The model's performance is evaluated both qualitatively and quantitatively.

### Singular Value Decomposition (SVD)

The core of the recommender system is the factorization of the user-item rating matrix $X$ using Truncated SVD. The centered rating matrix $X_{centered}$ is decomposed into:

$X_{centered} \approx U_k \Sigma_k V_k^T$

-   $U_k$: User-feature matrix (user embeddings).
-   $\Sigma_k$: Diagonal matrix of the top $k$ singular values.
-   $V_k^T$: Feature-movie matrix (movie embeddings).

The predicted rating for a user $u$ and movie $i$ is calculated by taking the dot product of the user's and movie's latent feature vectors and adding back the user's mean rating $\mu_u$:

$\hat{r}_{ui} = (U_k \cdot \sqrt{\Sigma_k})_u \cdot (V_k^T \cdot \sqrt{\Sigma_k})_i^T + \mu_u$

### Explained Variance

To understand the contribution of each latent factor, the explained variance is plotted. This helps in choosing an appropriate value for $k$ (the number of components).

### Root Mean Squared Error (RMSE)

The model is evaluated by computing the RMSE on a validation set. The RMSE measures the difference between the predicted ratings and the actual ratings:

$RMSE = \sqrt{\frac{1}{N} \sum_{(u,i) \in Val} (\hat{r}_{ui} - r_{ui})^2}$

The notebook evaluates and plots the RMSE for different values of $k$ to find the optimal number of latent factors.

### Cold Start Recommendations

For new users who are not in the original dataset, a "cold start" recommendation is provided. Given a set of initial ratings from a new user, a new user preference vector $p_{new}$ is computed:

$p_{new} = \sum_{i \in I_{rated}} (r_i - \mu_{new}) \cdot V_k(i)$

where:
- $I_{rated}$ is the set of movies rated by the new user.
- $r_i$ is the rating for movie $i$.
- $\mu_{new}$ is the mean of the new user's ratings.
- $V_k(i)$ is the embedding for movie $i$.

The predicted scores for all movies are then:

$Scores = p_{new} \cdot V_k^T + \mu_{new}$

Movies with the highest scores (that the user hasn't already rated) are recommended.

## Dataset

This project uses the **MovieLens 32M Dataset**. You need to download it and place the `ml-32m` directory in the root of the project. The notebook uses `ratings.csv` and `movies.csv` from this dataset.
