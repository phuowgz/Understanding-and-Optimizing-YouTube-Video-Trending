import pandas as pd
import json
import matplotlib.pyplot as plt
import os

def analyze_trending_hours(country_code):
    """
    Analyze average hours to trend by category and visualize the results.

    Args:
        country_code (str): Country code (e.g., 'US', 'KR').

    Returns:
        pd.DataFrame: DataFrame with average hours to trend by category.
    """
    # Ensure output directory exists
    output_dir = f'{country_code}_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    df = pd.read_csv(f'{country_code}_youtube_trending_data.csv')
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['trending_date'] = pd.to_datetime(df['trending_date'])
    df['hours_to_trend'] = (df['trending_date'] - df['publishedAt']).dt.total_seconds() / 3600

    # Load category mapping
    with open(f'{country_code}_category_id_to_name.json', 'r') as f:
        category_names = json.load(f)

    # Create a mapping from categoryId to category name
    category_id_to_name = {int(key): value for key, value in category_names.items()}
    df['category_name'] = df['categoryId'].map(category_id_to_name)

    # Filter out 'Unknown' categories
    df = df[df['category_name'].notna()]

    # Calculate average hours_to_trend by category
    category_avg = df.groupby('category_name')['hours_to_trend'].mean().sort_values()

    # Create DataFrame for results
    category_avg_df = category_avg.reset_index(name='hours_to_trend')

    # Map back to categoryId with error handling
    name_to_category_id = {v: k for k, v in category_id_to_name.items()}
    category_avg_df['categoryId'] = category_avg_df['category_name'].map(
        lambda x: name_to_category_id.get(x, -1)  # Use -1 for unknown categories
    )

    # Save results
    category_avg_df.to_csv(f'{output_dir}/{country_code}_avg_hours_to_trend_by_category.csv', index=False)

    # Visualize
    plt.figure(figsize=(10, 6))
    category_avg.plot(kind='barh', color='skyblue')
    plt.title(f'Average Hours to Become Trending by Category ({country_code})', fontsize=14)
    plt.xlabel('Average Hours', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{country_code}_avg_hours_to_trend_by_category.png')
    plt.close()

    print(f"=== Average Hours to Trend by Category ({country_code}) ===")
    print(category_avg.to_string(float_format="%.2f"))
    print(f"\n✅ Saved CSV to '{output_dir}/{country_code}_avg_hours_to_trend_by_category.csv'")
    print(f"✅ Saved visualization to '{output_dir}/{country_code}_avg_hours_to_trend_by_category.png'")

    return category_avg_df

if __name__ == "__main__":
    countries = ['GB', 'US', 'BR', 'CA', 'MX']
    for country in countries:
        analyze_trending_hours(country)