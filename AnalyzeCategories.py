import pandas as pd
import json
from tabulate import tabulate
import os

def analyze_categories(country_code):
    """
    Analyze category statistics including video count, engagement stats, and percentages.

    Args:
        country_code (str): Country code (e.g., 'US', 'KR').

    Returns:
        pd.DataFrame: DataFrame with category statistics.
    """
    # Ensure output directory exists
    output_dir = f'{country_code}_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    df = pd.read_csv(f'{country_code}_youtube_trending_data.csv')

    # Load category mapping
    with open(f'{country_code}_category_id_to_name.json', 'r') as f:
        category_names = json.load(f)
    # Convert categoryId to str for mapping
    category_names = {str(k): v for k, v in category_names.items()}

    # Group by categoryId and calculate stats
    stats = df.groupby('categoryId').agg({
        'view_count': 'mean',
        'likes': 'mean',
        'dislikes': 'mean',
        'comment_count': 'mean',
        'video_id': 'count'
    }).rename(columns={
        'view_count': 'Avg Views',
        'likes': 'Avg Likes',
        'dislikes': 'Avg Dislikes',
        'comment_count': 'Avg Comments',
        'video_id': 'Video Count'
    }).reset_index()

    # Add category name
    stats['Category'] = stats['categoryId'].astype(str).map(category_names).fillna('Unknown')

    # Filter out 'Unknown' categories
    stats = stats[stats['Category'] != 'Unknown']

    # Add percentage
    stats['Percentage'] = (stats['Video Count'] / len(df) * 100).round(2)

    # Format columns
    stats = stats[['Category', 'Video Count', 'Percentage', 'Avg Views', 'Avg Likes', 'Avg Dislikes', 'Avg Comments']]
    stats = stats.round(2)

    # Display table
    print(f"=== All Categories with Engagement Stats ({country_code}) ===")
    print(tabulate(stats, headers='keys', showindex=True, tablefmt='pretty'))

    # Save results
    stats.to_csv(f'{output_dir}/{country_code}_category_stats.csv', index=False)
    print(f"\n✅ Saved results to '{output_dir}/{country_code}_category_stats.csv'")

    return stats

if __name__ == "__main__":
    countries = ['GB', 'US', 'BR', 'CA', 'MX']
    for country in countries:
        analyze_categories(country)