import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

##### USED AGENT TO MOVE CONTENT OF EDA NOTEBOOK TO A SCRIPT########

def plot_eda(churn_train: pd.DataFrame) -> None:
    """Create exploratory data analysis plots for customer churn dataset."""
    
    # Make a copy to avoid modifying original data
    df = churn_train.copy()
    
    # Define consistent color palette
    # 0 = No Churn (blue), 1 = Churn (red)
    churn_colors_dict = {0: '#1f77b4', 1: '#d62728'}  # Dictionary for countplot
    churn_palette_list = ['#1f77b4', '#d62728']  # List for histplot matches hue_order [0, 1]
    churn_order = [0, 1]  # Ensure consistent ordering
    
    # Missing values
    
    msno.bar(df)
    plt.tight_layout()
    plt.show()
    
    msno.matrix(df)
    plt.tight_layout()
    plt.show()
    
    # Age Distribution
    plt.figure(figsize=(12, 6))
    df_plot = df.copy()
    df_plot['Churn'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
    churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
    sns.countplot(data=df_plot, x='Age', hue='Churn', hue_order=['No Churn', 'Churn'], order=sorted(df['Age'].unique()), palette=churn_colors_labeled)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Age Distribution by Churn Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Gender Distribution
    plt.figure(figsize=(8, 6))
    df_plot = df.copy()
    df_plot['Churn'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
    churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
    sns.countplot(data=df_plot, x='Gender', hue='Churn', hue_order=['No Churn', 'Churn'], palette=churn_colors_labeled)
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.title('Gender Distribution by Churn Status')
    plt.tight_layout()
    plt.show()
    
    # Usage Frequency Distribution
    plt.figure(figsize=(12, 6))
    df_plot = df.copy()
    df_plot['Churn'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
    churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
    sns.countplot(data=df_plot, x='Usage Frequency', hue='Churn', hue_order=['No Churn', 'Churn'], 
                  order=sorted(df['Usage Frequency'].unique()), palette=churn_colors_labeled)
    plt.xlabel('Usage Frequency')
    plt.ylabel('Count')
    plt.title('Usage Frequency Distribution by Churn Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Subscription Type Distribution
    plt.figure(figsize=(10, 6))
    df_plot = df.copy()
    df_plot['Churn'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
    churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
    sns.countplot(data=df_plot, x='Subscription Type', hue='Churn', hue_order=['No Churn', 'Churn'], palette=churn_colors_labeled)
    plt.xlabel('Subscription Type')
    plt.ylabel('Count')
    plt.title('Subscription Type Distribution by Churn Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Contract Length Distribution
    plt.figure(figsize=(10, 6))
    df_plot = df.copy()
    df_plot['Churn'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
    churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
    sns.countplot(data=df_plot, x='Contract Length', hue='Churn', hue_order=['No Churn', 'Churn'], 
                  order=sorted(df['Contract Length'].unique()), palette=churn_colors_labeled)
    plt.xlabel('Contract Length')
    plt.ylabel('Count')
    plt.title('Contract Length Distribution by Churn Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Total Spend Distribution
    plt.figure(figsize=(12, 6))
    df_plot = df.copy()
    df_plot['Churn'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
    churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
    ax = sns.histplot(data=df_plot, x='Total Spend', hue='Churn', hue_order=['No Churn', 'Churn'], multiple='dodge', bins=30, palette=churn_colors_labeled)
    plt.xlabel('Total Spend')
    plt.ylabel('Count')
    plt.title('Total Spend Distribution by Churn Status')
    plt.tight_layout()
    plt.show()
    
    # Customer Status Distribution
    if 'Customer Status' in df.columns:
        plt.figure(figsize=(10, 6))
        df_plot = df.copy()
        df_plot['Churn'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
        churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
        sns.countplot(data=df_plot, x='Customer Status', hue='Churn', hue_order=['No Churn', 'Churn'], palette=churn_colors_labeled)
        plt.xlabel('Customer Status')
        plt.ylabel('Count')
        plt.title('Customer Status Distribution by Churn Status')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("\nCustomer Status proportions:")
        print(df[['Customer Status', 'Churn']].value_counts(normalize=True))
    
    # Last Payment Date (if exists)
    if 'Last Payment Date' in df.columns:
        df_temp = df.copy()
        df_temp['Last Payment Date'] = pd.to_datetime('2024-' + df_temp['Last Payment Date'].astype(str), 
                                                        format='%Y-%m-%d', 
                                                        errors='coerce')
        reference_date = df_temp['Last Payment Date'].max()
        df_temp['Days_Since_Payment'] = (reference_date - df_temp['Last Payment Date']).dt.days
        
        plt.figure(figsize=(12, 6))
        plot_data = df_temp.dropna(subset=['Days_Since_Payment', 'Churn']).copy()
        plot_data['Churn'] = plot_data['Churn'].map({0: 'No Churn', 1: 'Churn'})
        churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
        sns.histplot(data=plot_data, x='Days_Since_Payment', hue='Churn', hue_order=['No Churn', 'Churn'], 
                     multiple='dodge', bins=30, palette=churn_colors_labeled)
        plt.xlabel('Days Since Last Payment')
        plt.ylabel('Count')
        plt.title('Days Since Last Payment Distribution by Churn Status')
        plt.tight_layout()
        plt.show()
    
    # Last Due Date (if exists)
    if 'Last Due Date' in df.columns:
        df_temp = df.copy()
        df_temp['Last Due Date'] = pd.to_datetime('2024-' + df_temp['Last Due Date'].astype(str), 
                                                    format='%Y-%m-%d', 
                                                    errors='coerce')
        reference_date = df_temp['Last Due Date'].max()
        df_temp['Days_Since_Due'] = (reference_date - df_temp['Last Due Date']).dt.days
        
        plt.figure(figsize=(12, 6))
        plot_data = df_temp.dropna(subset=['Days_Since_Due', 'Churn']).copy()
        plot_data['Churn'] = plot_data['Churn'].map({0: 'No Churn', 1: 'Churn'})
        churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
        sns.histplot(data=plot_data, x='Days_Since_Due', hue='Churn', hue_order=['No Churn', 'Churn'], 
                     multiple='dodge', bins=30, palette=churn_colors_labeled)
        plt.xlabel('Days Since Last Due Date')
        plt.ylabel('Count')
        plt.title('Days Since Last Due Date Distribution by Churn Status')
        plt.tight_layout()
        plt.show()
    
    # Tenure Distribution
    plt.figure(figsize=(12, 6))
    df_plot = df.copy()
    df_plot['Churn'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
    churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
    sns.histplot(data=df_plot, x='Tenure', hue='Churn', hue_order=['No Churn', 'Churn'], multiple='dodge', bins=30, palette=churn_colors_labeled)
    plt.xlabel('Tenure (months)')
    plt.ylabel('Count')
    plt.title('Tenure Distribution by Churn Status')
    plt.tight_layout()
    plt.show()
    print(f"Tenure Missing values: {df['Tenure'].isna().sum()} ({df['Tenure'].isna().sum() / len(df) * 100:.2f}%)")
    
    # Support Calls Distribution
    plt.figure(figsize=(12, 6))
    df_plot = df.copy()
    df_plot['Churn'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
    churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
    sns.countplot(data=df_plot, x='Support Calls', hue='Churn', hue_order=['No Churn', 'Churn'], 
                  order=sorted(df['Support Calls'].dropna().unique()), palette=churn_colors_labeled)
    plt.xlabel('Support Calls')
    plt.ylabel('Count')
    plt.title('Support Calls Distribution by Churn Status')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print(f"Support Calls Missing values: {df['Support Calls'].isna().sum()} ({df['Support Calls'].isna().sum() / len(df) * 100:.2f}%)")
    
    # Last Interaction Distribution
    plt.figure(figsize=(12, 6))
    df_plot = df.copy()
    df_plot['Churn'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
    churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
    sns.histplot(data=df_plot, x='Last Interaction', hue='Churn', hue_order=['No Churn', 'Churn'], multiple='dodge', bins=30, palette=churn_colors_labeled)
    plt.xlabel('Last Interaction (days)')
    plt.ylabel('Count')
    plt.title('Last Interaction Distribution by Churn Status')
    plt.tight_layout()
    plt.show()
    print(f"Last Interaction Missing values: {df['Last Interaction'].isna().sum()} ({df['Last Interaction'].isna().sum() / len(df) * 100:.2f}%)")
    
    # Payment Delay Distribution
    plt.figure(figsize=(12, 6))
    df_plot = df.copy()
    df_plot['Churn'] = df_plot['Churn'].map({0: 'No Churn', 1: 'Churn'})
    churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
    sns.histplot(data=df_plot, x='Payment Delay', hue='Churn', hue_order=['No Churn', 'Churn'], multiple='dodge', bins=30, palette=churn_colors_labeled)
    plt.xlabel('Payment Delay (days)')
    plt.ylabel('Count')
    plt.title('Payment Delay Distribution by Churn Status')
    plt.tight_layout()
    plt.show()
    print(f"Payment Delay Missing values: {df['Payment Delay'].isna().sum()} ({df['Payment Delay'].isna().sum() / len(df) * 100:.2f}%)")
    
    # Missingness Patterns
    print("\n" + "="*50)
    print("MISSINGNESS PATTERNS BY CHURN STATUS")
    print("="*50)
    
    missing_cols = ['Tenure', 'Support Calls', 'Last Interaction', 'Payment Delay']
    
    for col in missing_cols:
        df[f'{col}_Missing'] = df[col].isna().astype(int)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(missing_cols):
        ax = axes[idx]
        missing_col = f'{col}_Missing'
        
        data = df[[missing_col, 'Churn']].value_counts().reset_index()
        data.columns = [missing_col, 'Churn', 'count']
        
        data['Churn'] = data['Churn'].map({0: 'No Churn', 1: 'Churn'})
        churn_colors_labeled = {'No Churn': '#1f77b4', 'Churn': '#d62728'}
        sns.barplot(data=data, x=missing_col, y='count', hue='Churn', hue_order=['No Churn', 'Churn'], ax=ax, palette=churn_colors_labeled)
        ax.set_xlabel(f'{col} Missing (0=Present, 1=Missing)')
        ax.set_ylabel('Count')
        ax.set_title(f'Missing {col} by Churn Status')
    
    plt.tight_layout()
    plt.show()
    
    print("\nMissing value proportions by Churn status:")
    for col in missing_cols:
        print(f"\n{col}:")
        print(df[[f'{col}_Missing', 'Churn']].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    import pandas as pd
    
    # Load training data
    churn_train = pd.read_csv("../data/raw/train.csv")
    
    # Run EDA
    plot_eda(churn_train)
