#!/usr/bin/env python
# coding: utf-8

## Classes and functions used in multiple notebooks

## General imports
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

## Imports for NLP
import nltk, re, spacy, string
from spacy.lang.en.examples import sentences
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

## Imports for analyses
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR



## Function for encoding

def oh_encoder(df, column):
    """Simple one-hot encoder that takes a dataframe and a column name and returns the dataframe with encoded column.
    """

    df = df.copy()
    all_categories = list(set(g for categories in df[column] for g in categories))
    one_hot_df = pd.DataFrame(0, index=df.index, columns=all_categories)
    for i, categories in enumerate(df[column]):
            one_hot_df.loc[i, categories] = 1
    df = df.drop(columns=[column]).join(one_hot_df)
        
    return df



## Function for data preparation

def data_preparation(df):
    """Take a dataframe, prepare it for use in NLP and analyses and return prepared dataframe.
    
    Args:
        df (Dataframe): Original dataframe to be prepared
        
    Returns:
        df (Dataframe): Prepared dataframe
        
    """

    df = df.copy()
    
    ###############################    
    ## Data Cleaning & Recoding
    ###############################
    
    ## drop columns not used in analyses
    df.drop(['sid', 'store_url', 'store_promo_url', 'published_meta', 'published_stsp', 'published_hltb',
           'published_igdb', 'image', 'current_price', 'discount', 
           'gfq_url', 'gfq_difficulty_comment', 'gfq_rating_comment', 'gfq_length_comment',
           'hltb_url', 'meta_url', 'igdb_url'], axis=1, inplace=True)
    
    ## publish date as timedelta
    df["published_store"] = pd.to_datetime(df["published_store"]) - pd.Timestamp(1997, 1, 1)
    df["published_store"] = df["published_store"].apply(lambda value: value.days)
     
    ## missing data 1: If language or voiceover is missing, set to "One_unknown"
    df.loc[df["languages"].isna(), "languages"] = "One_unknown"
    df.loc[df["voiceovers"].isna(), "voiceovers"] = "One_unknown"

    ## delete games without English as language:
    count_no_en = 0
    for x in df.index:
        if "english" not in df.loc[x,"languages"].lower():
            count_no_en += 1
            df = df.drop(labels=x, axis=0)
    print(f"Games without English language: {count_no_en}")
             
    ## use only number of languages and voiceovers
    df["languages"] = df["languages"].apply(lambda value: len(value.split(",")))
    df["voiceovers"] = df["voiceovers"].apply(lambda value: len(value.split(",")))
      
    ## missing data 2: drop columns with more than 75% missing data:
    for col in df.columns:
        if df[col].isna().sum() > df.shape[0]*0.75:
            df.drop(col, axis=1, inplace=True) 
    
    ## One-Hot-Encoding
    
    ## Genres
    ## split strings in genre and platform columns
    df['genres'] = df['genres'].apply(lambda x: x.split(','))
    df['platforms'] = df['platforms'].apply(lambda x: x.split(','))
    ## replace genres
    df['genres'] = df['genres'].apply(lambda genres: list(set(['Indie' if genre == 'Инди' else genre for genre in genres])))
    df['genres'] = df['genres'].apply(lambda genres: list(set(['Adventure' if genre == 'Приключенческие игры' else genre for genre in genres])))
    
    ## One-Hot Encoding
    df = oh_encoder(df, "genres")
    df = oh_encoder(df, "platforms")
    
    ## Rename columns including spaces
    df.rename(columns={'Game Development':'Game_Development',
                      'Free to Play':'Free_to_Play',
                      'Massively Multiplayer':'Massively_Multiplayer',
                      'Early Access':'Early_Access',
                      'Sexual Content':'Sexual_Content'}, inplace=True)

    return df


## Function for text cleaning

def text_cleaner(sentence):
    """Take a string, clean it for use in vectorization and return cleaned string.
    
    Args:
        sentence (string): Original string to be cleaned
        
    Returns:
        doc_str (string): Cleaned String
        
    """
    
    ## counter
    global call_count 
    call_count += 1
    if call_count%1000 == 0:
        print(call_count)
    if sentence is None:
        doc_str = ""
    else:
        ## tokenize and delete pronouns, stopwords and punctuation
        doc = nlp(sentence)
        clean_doc = [token.lemma_.lower() for token in doc if (token.pos_ !="PRON") and (token.lemma_ not in stopWords) and (token.lemma_ not in punctuations)]
        ## rejoin texts
        doc_str = " ".join(clean_doc)
        ## deleting points, tabs, spaces and line breaks
        doc_str = re.sub("[\s]+", " ", doc_str)
        ## deleting numbers
        doc_str = re.sub(r'\d+', '', doc_str) 
    return doc_str


## Class for analyses

class NLPAnalyzer():
    """A class used for the analyses including NLP.

        Attributes:

            df (Dataframe):
                The DataFrame to be used for analyses.
            target_var (string):
                The name of the target column.
            max_feature_list (list):
                A list of interger values to be used as maximum number of features for vectorization.
            test_size (float):
                The fraction of the Dataframe to be used as the test sample.
            tfidf (bool):
                A Boolean that is True if TF-IDF vectorization should be used instead of BOW/Count vectorization.
            var_dict (dictionary):
                A dictionary of potential targets with column names (keys) and column descriptions (values).
                These columns are excluded from analyses and only the target used in target_var is included as target.
            train_data (Dataframe):
                Data to be used as train features. Generated when runnning :meth: "analyze".
            test_data (Dataframe):
                Data to be used as test features. Generated when runnning :meth: "analyze".
            train_target (Dataframe):
                Data to be used as train target. Generated when runnning :meth: "analyze".
            test_target (Dataframe):
                Data to be used as test target. Generated when runnning :meth: "analyze".
            vectorizer (instance of vectorizer class from sklearn):
                Vectorizer used. Can be CountVectorizer or TfidfVectorizer. Generated when runnning :meth: analyze.
            model_sm (instance of statsmodels.api.OLS):
                OLS regression used for extraction of t-values. Generated when runnning :meth: analyze.
        
        Methods:
    
            analyze():
                Conduct analyses, print figures for Adjusted R-squared and T-values of non-NLP features. Return table of results.
  
            extract_words(max_features=50):
                Conduct OLS regression for max_features and return list of words showing significant effects and corresponding t-values.
                
            plot_words_multi(word_list_1, word_list_2, target_1, target_2, top_number=25):
                Plot words of two analyses extracted with :meth: extract_words.
                       
        """
    
    def __init__(self, df, target_var, target_name, max_feature_list=[0, 2000, 2500, 3000, 3500, 4000], test_size=0.25, tfidf=False, naming_suffix=""):
        """Construct all the necessary attributes for the NLPAnalyzer object.

        Args:
            df (Dataframe):
                The DataFrame to be used for analyses.
            target_var (string):
                The target column.
            target_name (string):
                The name of the target column (used for plotting).
            max_feature_list (list):
                A list of interger values to be used as maximum number of features for vectorization.
            test_size (float):
                The fraction of the Dataframe to be used as the test sample.
            tfidf (bool):
                A Boolean that is True if TF-IDF vectorization should be used instead of BOW/Count vectorization.
            naming_suffix (string):
                A string that is added to plots for simple comparison of different models for a specific target.
                
        """
        
        self.df = df.copy()
        self.target_var = target_var
        self.target_name = target_name
        self.test_size = test_size
        self.max_feature_list = max_feature_list
        self.tfidf = tfidf
        self.naming_suffix = naming_suffix
        
        print("*"*50, "\n", "Initialization - Target:", self.target_var, "\n", "*"*50)

        ## delete missings
        self.df = self.df.dropna(axis=0, how="any")

        ## Reset index
        self.df.reset_index()
        
        print("*"*50, "\n", "Data preparation done", "\n", "*"*50)

    
## Class for analyses

class NLPAnalyzer():
    """A class used for the analyses including NLP.

        Attributes:

            df (Dataframe):
                The DataFrame to be used for analyses.
            target_var (string):
                The name of the target column.
            max_feature_list (list):
                A list of interger values to be used as maximum number of features for vectorization.
            test_size (float):
                The fraction of the Dataframe to be used as the test sample.
            tfidf (bool):
                A Boolean that is True if TF-IDF vectorization should be used instead of BOW/Count vectorization.
            var_dict (dictionary):
                A dictionary of potential targets with column names (keys) and column descriptions (values).
                These columns are excluded from analyses and only the target used in target_var is included as target.
            train_data (Dataframe):
                Data to be used as train features. Generated when runnning :meth: "analyze".
            test_data (Dataframe):
                Data to be used as test features. Generated when runnning :meth: "analyze".
            train_target (Dataframe):
                Data to be used as train target. Generated when runnning :meth: "analyze".
            test_target (Dataframe):
                Data to be used as test target. Generated when runnning :meth: "analyze".
            vectorizer (instance of vectorizer class from sklearn):
                Vectorizer used. Can be CountVectorizer or TfidfVectorizer. Generated when runnning :meth: analyze.
            model_sm (instance of statsmodels.api.OLS):
                OLS regression used for extraction of t-values. Generated when runnning :meth: analyze.
        
        Methods:
    
            analyze():
                Conduct analyses, print figures for Adjusted R-squared and T-values of non-NLP features. Return table of results.
  
            extract_words(max_features=50):
                Conduct OLS regression for max_features and return list of words showing significant effects and corresponding t-values.
                
            plot_words_multi(word_list_1, word_list_2, target_1, target_2, top_number=25):
                Plot words of two analyses extracted with :meth: extract_words.
                       
        """
    
    def __init__(self, df, target_var, target_name, max_feature_list=[0, 2000, 2500, 3000, 3500, 4000], test_size=0.25, tfidf=False, naming_suffix=""):
        """Construct all the necessary attributes for the NLPAnalyzer object.

        Args:
            df (Dataframe):
                The DataFrame to be used for analyses.
            target_var (string):
                The target column.
            target_name (string):
                The name of the target column (used for plotting).
            max_feature_list (list):
                A list of interger values to be used as maximum number of features for vectorization.
            test_size (float):
                The fraction of the Dataframe to be used as the test sample.
            tfidf (bool):
                A Boolean that is True if TF-IDF vectorization should be used instead of BOW/Count vectorization.
            naming_suffix (string):
                A string that is added to plots for simple comparison of different models for a specific target.
                
        """
        
        self.df = df.copy()
        self.target_var = target_var
        self.target_name = target_name
        self.test_size = test_size
        self.max_feature_list = max_feature_list
        self.tfidf = tfidf
        self.naming_suffix = naming_suffix
        
        print("*"*50, "\n", "Initialization - Target:", self.target_var, "\n", "*"*50)

        ## delete missings
        self.df = self.df.dropna(axis=0, how="any")

        ## Reset index
        self.df.reset_index()
        
        print("*"*50, "\n", "Data preparation done", "\n", "*"*50)

    
    def analyze(self):
        """Conduct analyses, print figures for Adjusted R-squared and T-values of non-NLP features. Return list of results.     
            
            Returns:
                results_df (Dataframe):
                    Dataframe of models used, maximum number of features and corresponding values of adjusted R-squared.
                    
        """
        
        # Train-test split
        self.train_data, self.test_data, self.train_target, self.test_target = train_test_split(
            self.df.drop([self.target_var], axis=1), self.df[self.target_var], test_size=self.test_size, random_state=42)
        
        # List for results
        results = []

        # Define the columns_to_scale list using a list comprehension
        columns_to_scale = [
        col for col in self.train_data.columns
        if self.train_data[col].nunique() > 2 and pd.api.types.is_numeric_dtype(self.train_data[col])
        ]

        # Print the columns to be scaled for debugging purposes
        print(f"Scaling the following columns for analyses: {columns_to_scale}")

        # Check if there are columns to scale
        if columns_to_scale:
            scaler = StandardScaler()
            self.train_data[columns_to_scale] = scaler.fit_transform(self.train_data[columns_to_scale])
            self.test_data[columns_to_scale] = scaler.transform(self.test_data[columns_to_scale])
        else:
            print("No columns to scale.")

        
        # Loop for different values of max_features
        for max_feat in self.max_feature_list:
            try:
                if max_feat > 0:

                    if self.tfidf:
                        self.vectorizer = TfidfVectorizer(
                            stop_words='english',
                            max_df=0.9,
                            min_df=10,
                            max_features=max_feat
                        )

                        print("*"*50, "\n", "Using TF-IDF Vectorizer", "\n", "*"*50)
        
                    else:
                        self.vectorizer = CountVectorizer(
                            stop_words='english',
                            max_df=0.9,
                            min_df=10,
                            max_features=max_feat
                        )
                        
                        print("*"*50, "\n", "Using BOW Vectorizer", "\n", "*"*50)
                    
                    # Fit and transformation
                    X_train_text = self.vectorizer.fit_transform(self.train_data['description_clean_nonum'])
                    X_test_text = self.vectorizer.transform(self.test_data['description_clean_nonum'])
                    
                    # Convert text features into dataframe
                    X_train_text_df = pd.DataFrame(X_train_text.toarray(), columns=self.vectorizer.get_feature_names_out(), index=self.train_data.index)
                    X_test_text_df = pd.DataFrame(X_test_text.toarray(), columns=self.vectorizer.get_feature_names_out(), index=self.test_data.index)
                else:
                    # When max_feat is 0, use only non-text features
                    X_train_text_df = pd.DataFrame()
                    X_test_text_df = pd.DataFrame()
               
                # Keep non-text features
                X_train_non_text = self.train_data.drop('description_clean_nonum', axis=1)
                X_test_non_text = self.test_data.drop('description_clean_nonum', axis=1)
                
                # Make sure all non-text features are numeric
                X_train_non_text = X_train_non_text.apply(pd.to_numeric, errors='coerce')
                X_test_non_text = X_test_non_text.apply(pd.to_numeric, errors='coerce')
                
                # Concatenate dataframes
                X_train = pd.concat([X_train_non_text, X_train_text_df], axis=1)
                X_test = pd.concat([X_test_non_text, X_test_text_df], axis=1)
                
                # Make sure all features are numeric
                X_train = X_train.apply(pd.to_numeric, errors='coerce')
                X_test = X_test.apply(pd.to_numeric, errors='coerce')
       
                print("*"*50, "\n", "NLP:", max_feat, "words - Vectorization done", "\n", "*"*50)
                
                
                # OLS with statsmodels
                self.model_sm = sm.OLS(self.train_target, sm.add_constant(X_train)).fit()
                adj_r2_sm = self.model_sm.rsquared_adj
                if max_feat == 0:
                    display(self.model_sm.summary())
                print("*"*50, "\n", "R-squared for Statsmodels OLS (train data):", self.model_sm.rsquared)
                print("*"*50, "\n", "Adjusted R-squared for Statsmodels OLS (train data):", adj_r2_sm)
                
                # Add constant to X_test for prediction
                X_test_const = sm.add_constant(X_test, has_constant='add')
                r2_sm_test = r2_score(self.test_target, self.model_sm.predict(X_test_const))
                adj_r2_sm_test = 1 - ( ( (1-r2_sm_test) * (len(self.test_target) - 1) ) / ( (len(self.test_target) - X_test_const.shape[1] - 1) ) )
                print("*"*50, "\n", "R-squared for Statsmodels OLS (test data):", r2_sm_test)    
                print("*"*50, "\n", "Adjusted R-squared for Statsmodels OLS (test data):", adj_r2_sm_test)                          
                results.append(('OLS', max_feat, adj_r2_sm_test))
              
                print("*"*50, "\n", "NLP:", max_feat, "words - OLS SM done", "\n", "*"*50)
        
                
                # OLS with sklearn
                model_lr = LinearRegression(fit_intercept=True).fit(X_train, self.train_target)
                r2_lr = r2_score(self.test_target, model_lr.predict(X_test))
                adj_r2_lr = 1 - (1-r2_lr)*(len(self.test_target)-1)/(len(self.test_target)-X_test.shape[1]-1)
                results.append(('LinearRegression', max_feat, adj_r2_lr))
                print(r2_lr, adj_r2_lr)
                print("*"*50, "\n", "NLP:", max_feat, "words - OLS SK done", "\n", "*"*50)
        
                
                # Lasso
                model_lasso = Lasso(fit_intercept=True, alpha=1e4).fit(X_train, self.train_target)
                r2_lasso = r2_score(self.test_target, model_lasso.predict(X_test))
                adj_r2_lasso = 1 - (1-r2_lasso)*(len(self.test_target)-1)/(len(self.test_target)-X_test.shape[1]-1)
                results.append(('Lasso', max_feat, adj_r2_lasso))
                print(r2_lasso, adj_r2_lasso)
                print("*"*50, "\n", "NLP:", max_feat, "words - Lasso done", "\n", "*"*50)
        
                
                # Ridge
                model_ridge = Ridge(fit_intercept=True, alpha=1e4).fit(X_train, self.train_target)
                r2_ridge = r2_score(self.test_target, model_ridge.predict(X_test))
                adj_r2_ridge = 1 - (1-r2_ridge)*(len(self.test_target)-1)/(len(self.test_target)-X_test.shape[1]-1)
                results.append(('Ridge', max_feat, adj_r2_ridge))
                print(r2_ridge, adj_r2_ridge)
                print("*"*50, "\n", "NLP:", max_feat, "words - Ridge done", "\n", "*"*50)

        
                # ElasticNet
                model_ela = ElasticNet(fit_intercept=True, alpha=1).fit(X_train, self.train_target)
                r2_ela = r2_score(self.test_target, model_ela.predict(X_test))
                adj_r2_ela = 1 - (1-r2_ela)*(len(self.test_target)-1)/(len(self.test_target)-X_test.shape[1]-1)
                results.append(('ElasticNet', max_feat, adj_r2_ela))
                print(r2_ela, adj_r2_ela)
                print("*"*50, "\n", "NLP:", max_feat, "words - ElasticNet done", "\n", "*"*50)

                
                # RandomForestRegressor
                model_rf = RandomForestRegressor(n_estimators=200, n_jobs=-1).fit(X_train, self.train_target)
                r2_rf = r2_score(self.test_target, model_rf.predict(X_test))
                adj_r2_rf = 1 - (1-r2_rf)*(len(self.test_target)-1)/(len(self.test_target)-X_test.shape[1]-1)
                results.append(('RandomForest', max_feat, adj_r2_rf))
                print(r2_rf, adj_r2_rf)
                print("*"*50, "\n", "NLP:", max_feat, "words - RF done", "\n", "*"*50)
                
            
            except ValueError as e:
                print(f"Error with max_features={max_feat}: {e}")
                continue

        # Convert results into dataframe
        self.results_df = pd.DataFrame(results, columns=['Model', 'Max_Features', 'Adjusted_R2'])
        self.results_df.loc[self.results_df["Adjusted_R2"]<0,"Adjusted_R2"] = 0.0
        
        # Plot adjusted R-squared for different max features
        plt.figure(figsize=(14, 7))
        for model in self.results_df['Model'].unique()[1:]:
            subset = self.results_df[self.results_df['Model'] == model]
            plt.plot(subset['Max_Features'], subset['Adjusted_R2'], label=model, alpha=0.5)
    
        plt.xlabel('Max Features')
        plt.ylabel('Adjusted R-squared')
        plt.title(f'Adjusted R-squared for Different Models (predicting {self.target_name})')
        plt.xticks(self.max_feature_list)
        plt.legend()
        plt.savefig(f'../../plots/fig_{self.target_var}_R2{self.naming_suffix}.png')
    
        # Plot t-values for non-text variables as horizontal bar plot
        # Sorting
        non_text_t_values = self.model_sm.tvalues[1:len(X_train_non_text.columns) + 1]
        non_text_feature_names = X_train_non_text.columns
        df_t_values = pd.DataFrame({
            'feature': non_text_feature_names,
            't_value': non_text_t_values
        })
        df_t_values_sorted = df_t_values.sort_values(by='t_value', ascending=False)
        non_text_t_values = df_t_values_sorted['t_value']
        non_text_feature_names = df_t_values_sorted['feature']

        plt.figure(figsize=(14, 7))
        plt.barh(non_text_feature_names, non_text_t_values, color="cornflowerblue")
        plt.xlabel('T-values')
        plt.ylabel('Non-text Features')
        plt.title(f'T-values for Non-text Variables (predicting {self.target_name})')

        max_y=len(non_text_feature_names)
        min_x= min(non_text_t_values)-0.5
        max_x= max(non_text_t_values)+0.5
        l1=plt.axvline(1.96, color="green", alpha=0.75)
        l2=plt.axvline(-1.96, color="green", alpha=0.75)
        plt.axvline(0, color="cornflowerblue", alpha=0.5)
        f1=plt.gca().fill_between(x=[-1.96, 1.96], y1=-1, y2=max_y, color="red", alpha=0.075)
        f2=plt.gca().fill_between(x=[min_x,-1.96], y1=-1, y2=max_y, color="green", label="Significant with p<0.05", alpha=0.075)
        f3=plt.gca().fill_between(x=[1.96, max_x], y1=-1, y2=max_y, color="green", alpha=0.075)
        
        legend = plt.gca().get_legend()
        if legend:
            legend.remove()        
        ax2=plt.gca().twinx()
        ax2.legend(handles=[f2], loc=1)

        plt.tight_layout()
        plt.savefig(f'../../plots/fig_{self.target_var}_Tvalues{self.naming_suffix}.png')
    
        return self.results_df

    
    def extract_words(self, max_features=50):
        """Conduct OLS regression for max_features and return list of significant words and corresponding t-values.
                
            Args:      
                max_features (int):
                    maximum number of features to be used for extraction of words showing significant effects.
                        
            Returns:
                significant_words (list):
                    List of words showing significant effects and corresponding t-values.

        """
        

        max_feat = max_features
        
        # OLS for t-value extraction with specific number of words
        try:
            if max_features > 0:

                if self.tfidf:
                    self.vectorizer = TfidfVectorizer(
                        stop_words='english',
                        max_df=0.9,
                        min_df=5,
                        max_features=max_feat
                    )

                    print("*"*50, "\n", "Using TF-IDF Vectorizer", "\n", "*"*50)
        
                else:
                    self.vectorizer = CountVectorizer(
                        stop_words='english',
                        max_df=0.9,
                        min_df=5,
                        max_features=max_feat
                    )
                        
                    print("*"*50, "\n", "Using BOW Vectorizer", "\n", "*"*50)
                    
                # Fit and transformation
                X_train_text = self.vectorizer.fit_transform(self.train_data['description_clean_nonum'])
                X_test_text = self.vectorizer.transform(self.test_data['description_clean_nonum'])
                    
                # Convert text features into dataframe
                X_train_text_df = pd.DataFrame(X_train_text.toarray(), columns=self.vectorizer.get_feature_names_out(), index=self.train_data.index)
                X_test_text_df = pd.DataFrame(X_test_text.toarray(), columns=self.vectorizer.get_feature_names_out(), index=self.test_data.index)
            
            else:
                # When max_feat is 0, word extraction is not conducted
                print("There are no influential words if maximum number of text features is zero.")
                return self
               
            # Keep non-text features
            X_train_non_text = self.train_data.drop('description_clean_nonum', axis=1)
            X_test_non_text = self.test_data.drop('description_clean_nonum', axis=1)
                
            # Make sure all non-text features are numeric
            X_train_non_text = X_train_non_text.apply(pd.to_numeric, errors='coerce')
            X_test_non_text = X_test_non_text.apply(pd.to_numeric, errors='coerce')
                
            # Concatenate dataframes
            X_train = pd.concat([X_train_non_text, X_train_text_df], axis=1)
            X_test = pd.concat([X_test_non_text, X_test_text_df], axis=1)
                
            # Make sure all features are numeric
            X_train = X_train.apply(pd.to_numeric, errors='coerce')
            X_test = X_test.apply(pd.to_numeric, errors='coerce')
      
            print("*"*50, "\n", "NLP:", max_feat, "words - Vectorization done", "\n", "*"*50)
              
            # OLS with statsmodels
            self.model_sm = sm.OLS(self.train_target, sm.add_constant(X_train)).fit()
                         
        except ValueError as e:
            print(f"Error with max_features={max_feat}: {e}")
      
        # Extract significant words
        significant_words = []
        text_feature_names = self.vectorizer.get_feature_names_out()
        
        for i, feature_name in enumerate(text_feature_names):
            t_value = self.model_sm.tvalues[len(X_train_non_text.columns) + i + 1]
            
            if abs(t_value) > 1.96:
                significant_words.append((feature_name, t_value))
        
        return significant_words


    def plot_words(self, number_words=50, max_words=25):
        """Plot words with significant influence on target variable in OLS regression.

        Args:
            number_words (int):
                Number of words to be used in OLS regression for extraction of words.
            max_words (int):
                Maximum number of words to be plotted.

        Returns:
            top (list):
                List of top words with highest t-values. Length is based on max_words.
                
        """

        # Sort word lists based on t-values in descending order
        sorted_words = sorted(self.extract_words(number_words), key=lambda x: x[1], reverse=True)
        
        # Extract top words
        top = sorted_words[:max_words]

        # Extract words and t-values
        words = [entry[0] for entry in top]
        t_values = [entry[1] for entry in top]
         
        # Plot
        plt.figure(figsize=(12, 8))
               
        # Make barh plot
        w1=plt.barh(words, t_values, color="cornflowerblue", label=self.target_name, alpha=0.5)
        plt.xlabel('T-values')
        plt.ylabel('Words')
        plt.title(f'T-values of most influential words in models for {self.target_name}')
        
        max_y=len(words)
        min_x= min(t_values)-0.5
        max_x= max(t_values)+0.5
        l1=plt.axvline(1.96, color="green", alpha=0.75)
        l2=plt.axvline(-1.96, color="green", alpha=0.75)
        plt.axvline(0, color="cornflowerblue", alpha=0.5)
        f1=plt.gca().fill_between(x=[-1.96, 1.96], y1=-1, y2=max_y, color="red", alpha=0.075)
        f2=plt.gca().fill_between(x=[min_x,-1.96], y1=-1, y2=max_y, color="green", label="Significant with p<0.05", alpha=0.075)
        f3=plt.gca().fill_between(x=[1.96, max_x], y1=-1, y2=max_y, color="green", alpha=0.075)
        
        legend = plt.gca().get_legend()
        if legend:
            legend.remove()
        ax2=plt.gca().twinx()
        ax2.legend(handles=[f2], loc=1)
        
        plt.tight_layout()
        plt.savefig(f'../../plots/fig_word_t-values_{self.target_var}{self.naming_suffix}.png')

        return top


    def plot_words_multi(self, word_list_1, word_list_2, target_1, target_2, max_words=25):
        """Plot words of two analyses extracted with :meth: extract_words.

        Args:
            words_list_1 (list):
                List of words and corresponding t-values extracted with :meth: extract_words. 
            words_list_1 (list):
                List of words and corresponding t-values extracted with :meth: extract_words. 
            target_1 (string):
                Name of dependent variable predicted with word_list_1. 
            target_2 (string):
                Name of dependent variable predicted with word_list_2. 
            max_words (int):
                Maximum number of words to be extracted and plotted.

            Returns:
                top_1 (list):
                    List of top words with highest t-values from word_list_1. Length is based on :attr: max_words.
                top_2 (list):
                    List of top words with highest t-values from word_list_1. Length is based on :attr: max_words.
                
        """

        # Sort word lists based on t-values in descending order
        sorted_words_1 = sorted(word_list_1, key=lambda x: x[1], reverse=True)
        sorted_words_2 = sorted(word_list_2, key=lambda x: x[1], reverse=True)
        
        # Extract top words
        top_1 = sorted_words_1[:max_words]
        top_2 = sorted_words_2[:max_words]
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Extract words and t-values
        words_1 = [entry[0] for entry in top_1]
        t_values_1 = [entry[1] for entry in top_1]
        words_2 = [entry[0] for entry in top_2]
        t_values_2 = [entry[1] for entry in top_2]
        
        # Make barh plot
        w1=plt.barh(words_1, t_values_1, color='orange', label=target_1, alpha=0.5)
        w2=plt.barh(words_2, t_values_2, color="cornflowerblue", label=target_2, alpha=0.5)
        plt.xlabel('T-values')
        plt.ylabel('Words')
        plt.title(f'T-values of most influential words in models for {target_1} and {target_2}')

        unique_values = list(set(words_1) | set(words_2))
        max_y=len(unique_values)
        min_x= min(min(t_values_1), max(t_values_1))-0.5
        max_x= max(max(t_values_1), max(t_values_1))+0.5
        l1=plt.axvline(1.96, color="green", alpha=0.75)
        l2=plt.axvline(-1.96, color="green", alpha=0.75)
        plt.axvline(0, color="cornflowerblue", alpha=0.5)
        f1=plt.gca().fill_between(x=[-1.96, 1.96], y1=-1, y2=max_y, color="red", alpha=0.075)
        f2=plt.gca().fill_between(x=[min_x,-1.96], y1=-1, y2=max_y, color="green", label="Significant with p<0.05", alpha=0.075)
        f3=plt.gca().fill_between(x=[1.96, max_x], y1=-1, y2=max_y, color="green", alpha=0.075)
        plt.gca().legend(handles=[w1, w2], loc=2, title="Models")
        ax2=plt.gca().twinx()
        ax2.legend(handles=[f2], loc=1)
        
        plt.tight_layout()
        plt.savefig(f'../../plots/fig_word_t-values_{target_1.replace(" ", "")}_{target_2.replace(" ", "")}{self.naming_suffix}.png')

        return top_1, top_2



