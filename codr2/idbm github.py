import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

class BollywoodDataAnalyzer:
    def __init__(self, csv_path):
        self.data = None
        self.csv_path = csv_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def import_and_clean(self):
        self.data = pd.read_csv(self.csv_path, encoding='latin1')
        print(self.data['Year'].head(20))  # Print the first 20 entries of the 'Year' column
        self.data.dropna(inplace=True)
        self._transform_columns()

    def _transform_columns(self):
        self.data['Duration'] = self.data['Duration'].str.extract('(\d+)').astype(float)
        self.data['Votes'] = self.data['Votes'].str.replace(',', '').astype(int)
        self.data['Year'] = self.data['Year'].str.extract('(\d{4})').astype(int)
        self.data['Actor_Combined'] = self.data[['Actor 1', 'Actor 2', 'Actor 3']].fillna('').agg(', '.join, axis=1)
        
        le = LabelEncoder()
        for col in ['Director', 'Genre', 'Actor_Combined']:
            self.data[f'{col}_Code'] = le.fit_transform(self.data[col])

    def visualize_genre_distribution(self):
        genre_data = self.data['Genre'].str.split(', ', expand=True).stack().value_counts()
        
        plt.figure(figsize=(12, 6))
        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(genre_data)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title('Genre Distribution')
        plt.show()

    def plot_top_contributors(self, column, title, n=20):
        top_n = self.data[column].value_counts().nlargest(n)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_n.index, y=top_n.values)
        plt.title(f'Top {n} {title}')
        plt.xticks(rotation=90)
        plt.show()

    def plot_rating_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['Rating'], bins=20, kde=True)
        plt.title('Distribution of Movie Ratings')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()

    def plot_year_vs_rating(self):
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='Year', y='Rating', data=self.data, alpha=0.6)
        plt.title('Movie Ratings Over the Years')
        plt.xlabel('Year')
        plt.ylabel('Rating')
        plt.show()

    def prepare_for_modeling(self):
        features = ['Year', 'Duration', 'Votes', 'Director_Code', 'Genre_Code', 'Actor_Combined_Code']
        X = self.data[features]
        y = self.data['Rating']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        imputer = SimpleImputer(strategy='median')
        self.X_train = imputer.fit_transform(self.X_train)
        self.X_test = imputer.transform(self.X_test)

    def evaluate_models(self):
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
            "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
            "CatBoost": CatBoostRegressor(n_estimators=100, random_state=42, verbose=False),
            "KNN": KNeighborsRegressor(n_neighbors=5)
        }

        results = []
        for name, model in models.items():
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, predictions, squared=False)
            r2 = r2_score(self.y_test, predictions)
            results.append({"Model": name, "RMSE": mse, "R2": r2})

        return pd.DataFrame(results).sort_values("R2", ascending=False)

def main():
    analyzer = BollywoodDataAnalyzer(r'C:\Users\sutar\Downloads\IMDb Movies India.csv')
    analyzer.import_and_clean()
    
    analyzer.visualize_genre_distribution()
    analyzer.plot_top_contributors('Director', 'Directors')
    analyzer.plot_top_contributors('Actor 1', 'Actors')
    analyzer.plot_rating_distribution()
    analyzer.plot_year_vs_rating()
    
    analyzer.prepare_for_modeling()
    model_performance = analyzer.evaluate_models()
    
    print("\nModel Performance Summary:")
    print(model_performance)

if __name__ == "__main__":
    main()
