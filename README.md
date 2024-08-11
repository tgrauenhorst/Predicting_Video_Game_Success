![UI_Preview](https://github.com/tgrauenhorst/Predicting_Video_Game_Success/blob/main/images/PredictingVideoGameSuccess_UI_Preview.gif)

# Predicting Video Game Success

### Project Overview
Objective: First, predicting the success of video games with machine learning models using information on genre, game engine, platform, number of supported languages, price, multiplayer support, publication date, length of the single-player campaign and description text. Second, building a dashboard that allows for easy access of the models by Indie developers to compare the potential effects of different game parameters and to optimize aspects such as game descriptions.

Context: We predict multiple conceptually different target variables related to video game success and have to optimize machine learning models for all targets. The main challenge of this project is the application of natural language processing to integrate the description of games. Additionally, we merge different data sets and use web scraping to extend the available information.

Significance: This project contributes to the prediction of market success of video games andd is especially helpful for Indie developers in various stages of the game development process. Developers can use the dashboard and compare how different game parameters, including the description text, affect predicted owner numbers, user ratings and sales.

Goal: This project aims to predict the success of video games as indicated by owner numbers, user ratings and sales based on very limited information. In order to optimize the predictive models and the use of available information, we compare various alternative models and recodings of independent variables, especially with regard to the description text.

## Team Members

- Thomas Grauenhorst: [GitHub](https://github.com/tgrauenhorst) - ML models, dashboard
- Fabian Joos: [GitHub](https://github.com/fabian-joos) - EDA, data merging
- Gülden Erkol: [GitHub](https://github.com/erkolg) - EDA, web scraping

## Jupyter Notebooks

This project consists of 14 Jupyter Notebooks that serve different purposes:

1. **Scrape_Steamdb_htlb_single.ipynb**

2. **Scrape_Steamdb_price.ipynb**

3. **Scrape_Steamdb_stsp_owners_mdntime.ipynb**

4. **Scrape_Steamdb_Voiceover.ipynb**

	In these four notebooks, we scrape additional data from various websites.

5. **merge_steamdb_and_game_data.ipynb**

	In this notebook, we merge game information from two sources.	

6. **merge_sales_data.ipynb**

	In this notebook, we merge game information with data on sales.

7. **eda_game_data_all.ipynb**

8. **eda_steamdb.ipynb**

9. **eda_vgchartz_sales_data.ipynb**

10. **eda_steam_merged.ipynb**

	In these four notebooks, we explore the single datasets and the merged data.

11. **analyses_nlp_comparison.ipynb**

	In this notebook, we use various forms of natural language processing [NLP] for the description text and various machine learning models to compare the model fit and predictive power of these alternatives. We specifically use different text vectorizations with varying numbers of included features for the description text and extract words that show significant influences or high feature imprtance to be used in the vocabulary of the final NLP vectorization. We test Linear Regression, Lasso, Ridge, ElasticNet and Random Forests.

12. **analyses_nlp_merged_1.ipynb**

	In this notebook, we analyze the merged dataset with extended case numbers and attributes. We again use various forms of natural language processing [NLP] for the description text and various machine learning models to compare the model fit and predictive power of these alternatives. We specifically use different text vectorizations with varying numbers of included features for the description text and extract words that show significant influences or high feature imprtance to be used in the vocabulary of the final NLP vectorization. We test Linear Regression, Lasso, Ridge, ElasticNet and Random Forests. We build the vectorizer to be used for the final predictive models with the vocabulary based on the previously extracted word lists. We then conduct grid searches to optimize our models in terms of hyperparameters.

13. **analyses_final_pipelines.ipynb**

	In this notebook, we build our final pipelines to predict owner numbers, user ratings and sales based on the best models and vectorization methods identified in the two previous analysis notebooks.

14. **predictions_dash_only.ipynb**
	
    In this final ntoebook, we build a dashboard for making predictions of owner numbers, user ratings and sales for new video games. The dashboard can be opened by running this notebook and accepts game information provided ny users in the UI to make all the predictions.



## Installation and Setup

To set up the project locally, follow these steps:

1. Clone the repository:
```
git clone https://github.com/tgrauenhorst/Predicting_Video_Game_Success.git
```

2. Navigate to the project directory:
```
cd your-repository
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. This is enough to use the dashboard in the notebook `predictions_dash_only.ipynb`, but if you want to explore the model comparisons, download the datasets and place them in the project directory. The original datasets can be acquired from the links [Steam Video Game Database](https://github.com/leinstay/steamdb), [Steam Releases](https://www.kaggle.com/datasets/whigmalwhim/steam-releases) and [Video Game Sales 2024](https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024).

**Note:** If any of the above files are missing, the corresponding functionality may not work as expected.

Once the setup is complete, you can use the notebook `predictions_dash_only.ipynb` to open the dashboard and use game information to predict the success of games. 

If you installed the datasets in step 4, you can also use the other notebooks to follow all analyses step and make model comparisons, but in that case, you have to merge the data first


## Datasets

This project uses three datasets avaiable at the following links: [Steam Video Game Database](https://github.com/leinstay/steamdb) (53981 rows, 46 columns), [Steam Releases](https://www.kaggle.com/datasets/whigmalwhim/steam-releases) (67571 rows, 20 columns) and [Video Game Sales 2024](https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024) (64016 rows, 14 columns).

The first two datasets contain information on games available at the Steam store pages and additional websites. The third dataset includes information on sales.

## Attribute Information

We use the following attributes from these datasets in the final models:

1. [Steam Video Game Database](https://github.com/leinstay/steamdb)
	
	- 'store_url' (Steam store URL, used for merging)
	- 'store_uscore' (User Score, target)
 	- 'published_store' (Publication data, used for feature engineering)
    - 'name' (Name of the game, used for merging)
    - 'description' (Description text, used for feature engineering)
    - 'full_price' (Price of the game, feature)
    - 'platforms' (Platforms the game is available on, feature)
    - 'languages' (Supported languages, used for feature engineering)
 	- 'voiceovers' (Voiceover languages, not used in final models)
    - 'categories' (Genres of the game, used for feature engineering)
 	- 'genres' (Genres of the game, used for feature engineering)
    - 'stsp_owners' (Owners of the game on Steam, target)
    - 'hltb_single' (Length of single player campaign, feature)

2. [Steam Releases](https://www.kaggle.com/datasets/whigmalwhim/steam-releases)

	- 'link' (Steam store URL, used for merging)
	- 'rating' (User Rating, target)
    - 'detected_technologies' (Game engines and other technologies, used for feature engineering)

3. [Video Game Sales 2024](https://www.kaggle.com/datasets/asaniczka/video-game-sales-2024)

	- 'title' (Name of the game, used for merging)
    - 'total_sales' (Sales of the game, target)
    

## Scraping/Merging/EDA/Cleaning

In the notebooks beginning with "Scrape_" that are included in the "notebooks/web_scraping" folder, we use web scraping to extend the datasets, specifically with regard to missing values. This did not lead to additional data in most cases. We then merge the datasets in the notebooks beginning with "merge_" that are included in the "notebooks/merge_data" folders. Consecutively, we explore the data and identify potential problems for the predictive models in the notebooks beginning with "eda_" that are included in the "notebooks/eda" folder.

## Model Choices

We predict ratio-scaled features and compared Linear Regression, Lasso, Ridge, ElasticNet and Random Forest models since these types of models performed best in preliminary tests. We compare different sets of features, especially with regard to the vectorization of the description text. We extract a vocabulary for the vectorization of the description text in the final predictive models based on the significance and feature importance of vectorized words in the model comparisons. Random Forests are used for the prediction of owners and user ratings. We use Grid Search to identify the best hyperparameters for these models. Sales data is predicted with a linear regression based on owner information in a separate dataset with very limited case numbers. The reason for this separate step is the very high number of missing cases for the merged data that includes sales data.

## Results

We use Adjusted R-squared values to compare the fit of models since the metric provides easily interpretable values of fit for ratio-scaled variables. The final models explain between 10% (number of owners) and 39% (user rating) of the variance in the target variables of the test data.

## Prediction Function

When users provide game information in the dashboard, the notebook runs the saved models based on the final pipelines to predict user ratings, user score, number of Steam owners and sales. For user ratings, user score and the number of owners, the predictions are based on Random Forest models. These models use the vocabularies extracted based on previous analyses to vectorize the game description and all additional information provided by users of the dashboard. For sales predictions, a simple linear regression based on the number of Steam owners is used.

## Final Remarks

The predictions are based on limited information on games and therefore have to be regarded as roughly approximative and have to be interpreted cautiously. Effects of features have to be considered to be subject to omitted-variable bias since the most important aspects of games related to gameplay experience and marketing are not included in the models. For instance, increasing the price of a game can have positive effects since games with higher prices are usually games with a bigger investments and efforts in terms of development. This obviously does not mean that increasing the price of an Indie game always has a positive effect on owners or sales. Similarly, increasing the length of a game does not necessarily mean that this positively affects ratings or sales of the game. Real-world effects of a change in this regard plausible depend on the user experience that this change implies. Users of the dashboard always have to kepe in mind that aspects of gameplay, game content, user experience and marketing are not included in the models.
