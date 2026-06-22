## Project Overview

The recommender system learns latent features for users and movies from the MovieLens dataset. It can provide personalized movie recommendations for both existing users in the dataset and new users based on a small number of initial ratings (a "cold start" scenario).


## Model Evaluation

The model's performance is evaluated both qualitatively and quantitatively.

### Singular Value Decomposition (SVD)

The core of the recommender system is the factorization of the user-item rating matrix $X$ using Truncated SVD. The centered rating matrix $X_{centered}$ is decomposed into:

$X_{centered} \approx U_k \Sigma_k V_k^T$

-   $U_k$: User-feature matrix (user embeddings).
-   $\Sigma_k$: Diagonal matrix of the top $k$ singular values.
-   $V_k^T$: Feature-movie matrix (movie embeddings).

The predicted rating for a user $u$ and movie $i$ is calculated by taking the dot product of the user's and movie's latent feature vectors and adding back the user's mean rating $\mu_u$:

$\hat{r}_{ui} = (U_k \cdot \sqrt{\Sigma_k})_u \cdot (V_k^T \cdot \sqrt{\Sigma_k})_i^T + \mu_u$<br>
or <br>
$\hat{r}_{u} = (U_k {\Sigma_k} V_k^T) + \mu$

### Root Mean Squared Error (RMSE)

The model is evaluated by computing the RMSE on a validation set. The RMSE measures the difference between the predicted ratings and the actual ratings:

$RMSE = \sqrt{\frac{1}{N} \sum_{(u,i) \in Val} (\hat{r}_{ui} - r_{ui})^2}$

The notebook evaluates and plots the RMSE for different values of $k$ to find the optimal number of latent factors.

### Cold Start Recommendations

Given a set of initial ratings from a new user, a new user preference vector $p_{new}$ is computed:

$p_{new}  = (r - \mu_{new})V_{k}$ 
- r is the vector made by new user's rating for each movie


it transforms the vector made my user's rating to movie embeded latent space
The predicted scores for all movies are then:

$Scores = p_{new} \cdot V_k^T + \mu_{new}$<br>
it transforms back movie space to user space


Movies with the highest scores (that the user hasn't already rated) are recommended.

## Dataset

This project uses the **MovieLens 32M Dataset**. You need to download it and place the `ml-32m` directory in the root of the project. The notebook uses `ratings.csv` and `movies.csv` from this dataset.
