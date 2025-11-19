import pandas as pd
import numpy as np
import shap
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

class YouTubeRecommendationSystem:
    def __init__(self, country_code):
        """
        Initialize the YouTube Recommendation System for a specific country.

        Args:
            country_code (str): Country code (e.g., 'US', 'KR').
        """
        self.country_code = country_code
        # Check required files
        required_files = [
            f'{self.country_code}_preprocessed_youtube_trending_data.csv',
            f'{self.country_code}_category_id_to_name.json',
            f'{self.country_code}_output/{self.country_code}_category_stats.csv',
            f'{self.country_code}_output/{self.country_code}_most_popular_tags_by_category.csv',
            f'{self.country_code}_output/{self.country_code}_top_title_keywords.csv',
            f'{self.country_code}_output/{self.country_code}_top_title_keywords_by_category.csv'
        ]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file {file} not found")
        
        sns.set(style='darkgrid', palette='deep', font='DejaVu Sans', font_scale=1.2)
        self.setup_features()
        self.load_data()
        self.X_test = None
        self.shap_values = None
        self.model = None
        self.scaler = MinMaxScaler()

    def load_data(self):
        """
        Load preprocessed data and related statistics, and filter category names based on available data.
        
        Attributes:
            df (pd.DataFrame): Preprocessed YouTube trending data.
            category_names (dict): Filtered mapping of category IDs to names.
            category_stats (pd.DataFrame): Category statistics.
            trending_stats (pd.DataFrame): Trending statistics by category.
            tag_stats (pd.DataFrame): Tag statistics by category.
            hours_to_trend (pd.DataFrame): Average hours to trend by category.
            peak_hours (pd.DataFrame): Peak publishing hours by category.
            top_title_keywords (pd.DataFrame): Top title keywords (global).
            top_title_keywords_by_category (pd.DataFrame): Top title keywords by category.
            tag_topics (pd.DataFrame): Tag topics (if available).
            title_topics (pd.DataFrame): Title topics (if available).
        
        Raises:
            ValueError: If any required data file cannot be loaded.
        """
        try:
            self.df = pd.read_csv(f'{self.country_code}_preprocessed_youtube_trending_data.csv')
            with open(f'{self.country_code}_category_id_to_name.json', 'r') as f:
                self.category_names = json.load(f)
            self.category_stats = pd.read_csv(f'{self.country_code}_output/{self.country_code}_category_stats.csv')
            self.trending_stats = pd.read_csv(f'{self.country_code}_output/{self.country_code}_trending_stats_by_category.csv')
            self.tag_stats = pd.read_csv(f'{self.country_code}_output/{self.country_code}_most_popular_tags_by_category.csv')
            self.hours_to_trend = pd.read_csv(f'{self.country_code}_output/{self.country_code}_avg_hours_to_trend_by_category.csv')
            self.peak_hours = pd.read_csv(f'{self.country_code}_output/{self.country_code}_peak_hours_by_category.csv')
            self.top_title_keywords = pd.read_csv(f'{self.country_code}_output/{self.country_code}_top_title_keywords.csv')
            self.top_title_keywords_by_category = pd.read_csv(f'{self.country_code}_output/{self.country_code}_top_title_keywords_by_category.csv')
            
            # Handle optional files
            try:
                self.tag_topics = pd.read_csv(f'{self.country_code}_output/{self.country_code}_tag_topics.csv')
                self.title_topics = pd.read_csv(f'{self.country_code}_output/{self.country_code}_title_topics.csv')
            except FileNotFoundError:
                self.tag_topics = pd.DataFrame()
                self.title_topics = pd.DataFrame()

            # Get valid category IDs from df
            valid_category_ids = set(self.df['categoryId'].astype(str).unique())
            category_stats_names = set(self.category_stats['Category'].str.strip().values)
            
            # Filter category_names to only include categories present in both category_stats and df
            valid_categories = set()
            unmatched_categories = []
            name_to_id = {}
            
            for category_id, category_name in self.category_names.items():
                category_name = category_name.strip()
                if category_name in category_stats_names and category_id in valid_category_ids:
                    if category_name not in name_to_id:
                        name_to_id[category_name] = category_id
                        valid_categories.add(category_id)
                    else:
                        print(f"⚠️ Warning: Duplicate category name '{category_name}' with ID {category_id} ignored (using ID {name_to_id[category_name]}).")
                else:
                    unmatched_categories.append(f"{category_id}: {category_name}")
            
            self.category_names = {k: v for k, v in self.category_names.items() if k in valid_categories}
            
            
            if not self.category_names:
                raise ValueError(f"No valid categories found for {self.country_code}. Check data consistency.")

        except Exception as e:
            raise ValueError(f"Error loading data: {str(e)}")

    def setup_features(self):
        """
        Define the features used for modeling.
        """
        self.features = [
            'view_count', 'likes', 'dislikes', 'comment_count',
            'engagement_rate', 'like_dislike_ratio', 'comment_view_ratio',
            'dislikes_per_comment', 'days_since_publication', 'likes_per_day',
            'comments_per_day', 'title_length', 'title_sentiment', 'view_growth_rate'
        ]
        self.log_transformed_features = [
            'view_count', 'likes', 'dislikes', 'comment_count',
            'like_dislike_ratio', 'dislikes_per_comment',
            'likes_per_day', 'comments_per_day', 'view_growth_rate'
        ]

    def load_model_and_shap(self):
        """
        Load the best model and corresponding SHAP values.
        """
        try:
            with open(f'{self.country_code}_output/{self.country_code}_best_model.json', 'r') as f:
                model_info = json.load(f)
            best_model_name = model_info['best_model']
            self.X_test = pd.read_csv(f'{self.country_code}_output/{self.country_code}_X_test.csv', index_col=0)
            self.X_test = self.X_test[self.features]
            if best_model_name == "XGBoost":
                self.model = joblib.load(f'{self.country_code}_output/{self.country_code}_model_xgb.joblib')
                self.shap_values = joblib.load(f'{self.country_code}_output/{self.country_code}_shap_values_xgb.joblib')
            else:
                self.model = joblib.load(f'{self.country_code}_output/{self.country_code}_model_lgb.joblib')
                self.shap_values = joblib.load(f'{self.country_code}_output/{self.country_code}_shap_values_lgb.joblib')
        except Exception as e:
            raise ValueError(f"Error loading model or SHAP: {str(e)}")

    def analyze_optimal_tags(self, category_id):
        """
        Analyze the optimal number of tags based on the most frequent count among top-performing videos.

        Args:
            category_id (str): Category ID.

        Returns:
            int: Optimal number of tags (minimum 3, maximum 10).
        """
        category_data = self.df[self.df['categoryId'].astype(str) == str(category_id)].copy()
        
        if 'tags' not in category_data.columns or category_data.empty:
            return 5
        
        category_data['num_tags'] = category_data['tags'].apply(
            lambda x: len(x.split('|')) if pd.notna(x) and x != '[none]' else 0
        )
        
        top_videos = category_data[category_data['view_velocity'] >= category_data['view_velocity'].quantile(0.75)]
        
        if top_videos.empty:
            return 5
        
        tag_counts = top_videos['num_tags'].value_counts()
        if tag_counts.empty:
            return 5
        
        optimal_num_tags = tag_counts.idxmax()
        return max(3, min(optimal_num_tags, 10))

    def get_category_insights(self, category_id):
        """
        Gather insights for a specific category.

        Args:
            category_id (str): Category ID.

        Returns:
            dict: Insights including general stats, trending patterns, tags, and time to trend.
        """
        category_id_str = str(category_id)
        category_id_int = int(category_id)
        category_name = self.category_names.get(category_id_str, f"Category {category_id_str}")
        insights = {
            'name': category_name,
            'general_stats': {},
            'trending_patterns': {},
            'tags': [],
            'time_to_trend': {},
            'optimal_hour': None,
            'optimal_num_tags': self.analyze_optimal_tags(category_id)
        }
        try:
            stats = self.category_stats[self.category_stats['Category'] == category_name]
            if not stats.empty:
                insights['general_stats'] = {
                    'percentage': float(stats['Percentage'].values[0]),
                    'avg_views': float(stats['Avg Views'].values[0]),
                    'avg_likes': float(stats['Avg Likes'].values[0])
                }
            patterns = self.trending_stats[self.trending_stats['Danh mục'] == category_name]
            if not patterns.empty:
                insights['trending_patterns'] = {
                    'peak_day': patterns.iloc[0].get('Ngày đỉnh', 'N/A'),
                    'peak_hour': patterns.iloc[0].get('Giờ đăng tối ưu', 'N/A'),
                    'avg_time': float(patterns.iloc[0].get('Thời gian lên trending (giờ)', 0))
                }
            optimal_hour = self.peak_hours[self.peak_hours['category_name'] == category_name]
            if not optimal_hour.empty:
                insights['optimal_hour'] = float(optimal_hour['publish_hour'].values[0])
            tags_rows = self.tag_stats[self.tag_stats['Category'] == category_name]
            if not tags_rows.empty:
                insights['tags'] = [tag for tag in tags_rows['Tag'].head(5).tolist() if tag != '[none]']
            time_stats = self.hours_to_trend[self.hours_to_trend['categoryId'] == category_id_int]
            if not time_stats.empty:
                median_days = max(1, time_stats['hours_to_trend'].median() / 24)
                insights['time_to_trend'] = {
                    'avg': float(time_stats['hours_to_trend'].mean() / 24),
                    'median': float(median_days),
                    'std': float(time_stats['hours_to_trend'].std() / 24) if len(time_stats['hours_to_trend']) > 1 else None,
                    'sample_count': len(time_stats['hours_to_trend'])
                }
        except:
            pass
        return insights

    def generate_shap_recommendations(self, category_data, category_id):
        """
        Generate recommendations based on SHAP analysis for a specific category.

        Args:
            category_data (pd.DataFrame): Data for the specific category.
            category_id (str): Category ID.

        Returns:
            tuple: List of recommendations and top features DataFrame.
        """
        recommendations = []
        try:
            if category_data.empty:
                return ["Insufficient data for SHAP analysis."], pd.DataFrame()

            # Select top 25% of videos by view_velocity
            top_videos = category_data[category_data['view_velocity'] >= category_data['view_velocity'].quantile(0.75)]
            if top_videos.empty:
                top_videos = category_data

            # Calculate median values and reverse log transformation
            median_values = top_videos[self.features].median()
            for feature in self.log_transformed_features:
                if feature in median_values and not pd.isna(median_values[feature]):
                    median_values[feature] = np.expm1(median_values[feature])

            # SHAP analysis
            test_indices = self.X_test.index.intersection(category_data.index)
            if len(test_indices) == 0:
                return ["No sufficient data for SHAP analysis in this category."], pd.DataFrame()

            test_positions = [list(self.X_test.index).index(idx) for idx in test_indices]
            category_shap = self.shap_values.values[test_positions]
            shap_df = pd.DataFrame(category_shap, columns=self.features)

            # Calculate mean SHAP values
            mean_shap = category_shap.mean(axis=0)
            if len(mean_shap) != len(self.features) or len(median_values) != len(self.features):
                return ["Error: Mismatch in feature dimensions for SHAP analysis."], pd.DataFrame()

            feature_importance = pd.DataFrame({
                'feature': self.features,
                'median_value': median_values.values,
                'mean_shap': mean_shap,
                'abs_shap': np.abs(mean_shap)
            }).sort_values('abs_shap', ascending=False)
            total_shap = feature_importance['abs_shap'].sum()
            if total_shap == 0:
                return ["Error: No significant SHAP values for this category."], pd.DataFrame()

            feature_importance['percent_contribution'] = feature_importance['abs_shap'] / total_shap * 100
            feature_importance['priority'] = feature_importance['percent_contribution'].apply(
                lambda x: 'High' if x > 30 else 'Medium' if x > 10 else 'Low'
            )

            top_features = feature_importance.head(5)

            # Create SHAP heatmap
            plt.figure(figsize=(12, 8))
            heatmap_data = feature_importance[['feature', 'percent_contribution', 'mean_shap']].set_index('feature')
            heatmap_data.columns = ['% Contribution', 'Mean SHAP Value']
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='viridis', cbar_kws={'label': 'Impact Intensity'})
            plt.title(f'Feature Impact on View Velocity\nCategory: {self.category_names.get(str(category_id), "Category " + str(category_id))} ({self.country_code})')
            plt.tight_layout()
            plot_path = f'{self.country_code}_output/{self.country_code}_shap_impact_category_{category_id}.png'
            plt.savefig(plot_path, dpi=300)
            plt.close()

            recommendations.append(
                f"📊 SHAP Analysis Summary\n"
                f"- Top 5 most important factors for trending videos:\n" +
                '\n'.join([f"  • {row['feature']}: {row['percent_contribution']:.1f}% (Priority: {row['priority']})" for i, row in top_features.iterrows()])
            )

            recommendations.append(
                f"\n📈 SHAP Impact Heatmap\n- View heatmap at: {plot_path}"
            )

            recommendations.append("\n🔍 Detailed Recommendations")
            for i, row in top_features.iterrows():
                feature = row['feature']
                median_val = row['median_value']
                percent_contribution = row['percent_contribution']
                priority = row['priority']
                
                if feature == 'view_count':
                    rec = f"• Views ({percent_contribution:.1f}%, {priority}): Aim for {median_val:,.0f} views within the first 24 hours."
                elif feature == 'likes':
                    rec = f"• Likes ({percent_contribution:.1f}%, {priority}): Target {median_val:,.0f} likes within the first 24 hours."
                elif feature == 'dislikes':
                    rec = f"• Dislikes ({percent_contribution:.1f}%, {priority}): Expect around {median_val:,.0f} dislikes."
                elif feature == 'comment_count':
                    rec = f"• Comments ({percent_contribution:.1f}%, {priority}): Aim for {median_val:,.0f} comments within the first 24 hours."
                elif feature == 'engagement_rate':
                    rec = f"• Engagement Rate ({percent_contribution:.1f}%, {priority}): Target an engagement rate of {median_val:.2%}."
                elif feature == 'like_dislike_ratio':
                    rec = f"• Like/Dislike Ratio ({percent_contribution:.1f}%, {priority}): Aim for a ratio of {median_val:.1f}."
                elif feature == 'comment_view_ratio':
                    rec = f"• Comment/View Ratio ({percent_contribution:.1f}%, {priority}): Target a ratio of {median_val:.4f}."
                elif feature == 'dislikes_per_comment':
                    rec = f"• Dislike/Comment Ratio ({percent_contribution:.1f}%, {priority}): Keep the ratio around {median_val:.2f}."
                elif feature == 'days_since_publication':
                    rec = f"• Time to Trend ({percent_contribution:.1f}%, {priority}): Aim to trend within {median_val:.1f} days."
                elif feature == 'likes_per_day':
                    rec = f"• Likes/Day ({percent_contribution:.1f}%, {priority}): Target {median_val:,.0f} likes per day."
                elif feature == 'comments_per_day':
                    rec = f"• Comments/Day ({percent_contribution:.1f}%, {priority}): Target {median_val:,.0f} comments per day."
                elif feature == 'title_length':
                    rec = f"• Title Length ({percent_contribution:.1f}%, {priority}): Use titles with around {median_val:.0f} characters."
                elif feature == 'title_sentiment':
                    rec = f"• Title Sentiment ({percent_contribution:.illin}%, {priority}): Aim for a sentiment score of {median_val:.2f} (neutral to positive)."
                elif feature == 'view_growth_rate':
                    rec = f"• View Growth Rate ({percent_contribution:.1f}%, {priority}): Target {median_val:,.0f} views per hour."
                recommendations.append(rec)

            return recommendations, top_features

        except Exception as e:
            return [f"Could not perform SHAP analysis: {str(e)}"], pd.DataFrame()

    def generate_eda_recommendations(self, insights, top_features, category_data):
        """
        Generate content optimization recommendations based on EDA.

        Args:
            insights (dict): Insights for the category.
            top_features (pd.DataFrame): Top features from SHAP analysis.
            category_data (pd.DataFrame): Data for the specific category.

        Returns:
            list: List of EDA-based recommendations.
        """
        recommendations = []
        category_name = insights['name']

        # Posting Time
        if insights['trending_patterns'] and insights['optimal_hour'] is not None:
            patterns = insights['trending_patterns']
            recommendations.append(
                f"• Post videos on {patterns['peak_day']} around {insights['optimal_hour']:.0f}:00 to trend faster (typically within {patterns['avg_time']:.1f} hours)."
            )

        # Time to Trend
        if insights['time_to_trend']:
            time_stats = insights['time_to_trend']
            if time_stats['std'] is not None:
                recommendations.append(
                    f"• Aim to trend within {time_stats['median']:.1f} days (typical range: {time_stats['median'] - time_stats['std']:.1f}–{time_stats['median'] + time_stats['std']:.1f} days)."
                )
            else:
                recommendations.append(
                    f"• Aim to trend within {time_stats['median']:.1f} days."
                )

        # Tags
        if insights['tags']:
            recommendations.append(f"• Use these popular tags: {', '.join(insights['tags'])}.")
        else:
            recommendations.append("• No popular tags available for this category.")

        # Title Keywords
        top_keywords = self.top_title_keywords_by_category[
            self.top_title_keywords_by_category['Category'] == category_name
        ]['Keyword'].head(5).tolist()
        if top_keywords:
            recommendations.append(f"• Include these category-specific title keywords: {', '.join(top_keywords)}.")
        else:
            top_keywords = self.top_title_keywords['Keyword'].head(5).tolist()
            recommendations.append(f"• No category-specific keywords available. Use these popular keywords: {', '.join(top_keywords)}.")

        # Optimal Number of Tags
        recommendations.append(f"• Use approximately {insights['optimal_num_tags']} tags per video.")

        return recommendations if recommendations else ["No optimization tips from data analysis."]

    def get_recommendations(self, category_id):
        """
        Generate comprehensive recommendations for a specific category.

        Args:
            category_id (str): Category ID.

        Returns:
            str: Formatted recommendations.
        """
        try:
            category_id = str(category_id)
            if category_id not in self.category_names:
                return f"Category {category_id} does not exist or has no data."
            category_data = self.df[self.df['categoryId'].astype(str) == category_id]
            if category_data.empty:
                return f"No data available for category {category_id} ({self.category_names[category_id]})."
            insights = self.get_category_insights(category_id)
            shap_recs, top_features = self.generate_shap_recommendations(category_data, category_id)
            eda_recs = self.generate_eda_recommendations(insights, top_features, category_data)
            
            priority_features = top_features['feature'].head(2).tolist() if not top_features.empty else []
            priority_actions = []
            
            for feature in priority_features:
                median_val = top_features[top_features['feature'] == feature]['median_value'].iloc[0]
                
                if feature == 'days_since_publication':
                    priority_actions.append(f"• Aim to trend within {median_val:.1f} days after posting.")
                elif feature == 'comment_count':
                    priority_actions.append(f"• Target {median_val:,.0f} comments within the first 24 hours.")
                elif feature == 'view_count':
                    priority_actions.append(f"• Aim for {median_val:,.0f} views within the first 24 hours.")
                elif feature == 'likes_per_day':
                    priority_actions.append(f"• Target {median_val:,.0f} likes per day.")
                elif feature == 'title_length':
                    priority_actions.append(f"• Use titles with around {median_val:.0f} characters.")
                elif feature == 'engagement_rate':
                    priority_actions.append(f"• Aim for an engagement rate of {median_val:.2%}.")
                elif feature == 'view_growth_rate':
                    priority_actions.append(f"• Target {median_val:,.0f} views per hour.")

            output = []
            output.append(f"📊 Recommendations for Category {insights['name']} ({self.country_code})")
            output.append("="*50)
            output.append("\n🚀 Priority Actions")
            if priority_actions:
                output.extend(priority_actions)
            else:
                output.append("• No priority actions available due to insufficient data.")
            output.append("\n🔍 Key Factor Analysis (SHAP)")
            output.extend(shap_recs)
            output.append("\n💡 Content Optimization Tips")
            output.extend(eda_recs)
            
            return "\n".join(output)

        except Exception as e:
            return f"Error generating recommendations: {str(e)}"

if __name__ == "__main__":
    countries = ['GB', 'US', 'BR', 'CA', 'MX']
    print("🎯 YouTube Content Recommendation System")
    print("Available countries: GB, US, BR, CA, MX")
    while True:
        country_input = input("\nEnter your country (or 'exit' to quit): ").strip().upper()
        if country_input.lower() == 'exit':
            break
        if country_input not in countries:
            print("⚠️ Invalid country")
            continue
        try:
            recommender = YouTubeRecommendationSystem(country_input)
        except Exception as e:
            print(f"Could not initialize recommender: {str(e)}")
            continue
        try:
            recommender.load_model_and_shap()
        except Exception as e:
            print(f"Could not load model or SHAP: {str(e)}")
            continue
        print(f"\nAvailable categories ({country_input}):")
        # Only display categories that have data in category_stats and df
        valid_category_names = {k: v for k, v in recommender.category_names.items() 
                              if v.strip() in recommender.category_stats['Category'].str.strip().values}
        if not valid_category_names:
            print("⚠️ No valid categories with data available.")
            continue
        for cat_id, name in sorted(valid_category_names.items(), key=lambda x: int(x[0])):
            print(f"- {cat_id}: {name}")
        while True:
            user_input = input("\nEnter category ID or name (or 'back' to select another country, 'exit' to quit): ").strip()
            if user_input.lower() == 'exit':
                exit()
            elif user_input.lower() == 'back':
                break
            category_id = None
            if user_input.isdigit():
                if user_input in valid_category_names:
                    category_id = user_input
            else:
                for cid, name in valid_category_names.items():
                    if user_input.lower() in name.lower():
                        category_id = cid
                        break
            if not category_id:
                print("⚠️ Category not found or has no data")
                continue
            recommendations = recommender.get_recommendations(category_id)
            print("\n" + recommendations)