import pandas as pd
import numpy as np
import re
import os
from sklearn.metrics import mean_absolute_error
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.show_dimensions', False)


def clean_raw_answers(file_path):
    """
    Clean raw answers from CSV file, focusing only on extracting digits.
    Parameters:
    file_path (str): Path to the CSV file
    Returns:
    pandas.DataFrame: DataFrame with raw and cleaned answers
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    deleted_rows = []
    total_tasks = len(df)
    
    def extract_digits_exp2(raw_text, model_name=None):
        """Extraction logic specific to EXP2."""
        if pd.isna(raw_text):
            return None
            
        # Clean the text: remove newlines and backslashes
        raw_text = str(raw_text).strip().replace('\n', ' ').replace('\\', '')
        
        # Extract lists inside square brackets
        matches = re.findall(r'\[([\d.,\s]+)\]', raw_text)
        
        if not matches:
            descriptive_patterns = [
                # Basic formats
                r'\[([\d\., ]+)\]',  # Basic bracketed numbers
                
                # Common ratio descriptions
                r'ratios are.*?\[([\d\., ]+)\]',
                r'values are.*?\[([\d\., ]+)\]',
                r'approximately.*?\[([\d\., ]+)\]',
                
                # Specific descriptions
                r'moving left to right.*?\[([\d\., ]+)\]',
                r'largest bar.*?\[([\d\., ]+)\]',
                r'maximum are:.*?\[([\d\., ]+)\]',
                
                # Full sentence patterns
                r'Based on the (?:image|bar chart|chart).*?maximum are:\s*\[([\d\., ]+)\]',
                r'estimate the ratios.*?\[([\d\., ]+)\]',
                r'estimated ratios.*?maximum are:\s*\[([\d\., ]+)\]',
                
                # New patterns for narrative style responses
                r'My estimate is:\s*`\[([\d\., ]+)\]`',  # Simple backtick format
                r'To solve this[\s\S]*?My estimate is:\s*`\[([\d\., ]+)\]`',  # Full narrative pattern
                r'estimate is:\s*`?\[?([\d\., ]+)\]?`?',  # Flexible format
                r'(?:^|[^\d])(1\.0\s*,\s*0\.6\s*,\s*0\.4\s*,\s*0\.25\s*,\s*0\.1)(?:[^\d]|$)'  # Exact sequence
            ]
            
            for pattern in descriptive_patterns:
                match_pattern = re.search(pattern, raw_text, re.IGNORECASE | re.DOTALL)
                if match_pattern:
                    matches = [match_pattern.group(1)]
                    break
        
        if matches:
            valid_lists = []
            for match in matches:
                # Clean up any extra spaces and split by comma
                numbers = [
                    float(num.strip()) 
                    for num in match.split(',') 
                    if re.match(r'^\d*\.?\d+$', num.strip())
                ]
                if len(numbers) == 5:  # Only keep lists of exactly 5 numbers
                    valid_lists.append(numbers)
            return valid_lists[-1] if valid_lists else None
        return None
    
    # Example loop to clean the data
    rows_to_delete = []
    
    for index, row in df.iterrows():
        model_name = row.get('model', '')  # Assuming model name is in a 'model' column
        raw_text = row.get('raw_answer', '')  # Assuming raw answers are in a 'raw_answer' column

        # Extract digits
        cleaned_answer = extract_digits_exp2(raw_text, model_name)

        # Check if answer is valid, otherwise mark for deletion
        if cleaned_answer is None:
            deleted_rows.append(row)
            rows_to_delete.append(index)

    # Drop rows marked for deletion
    df.drop(rows_to_delete, inplace=True)

    # Print deleted rows and total tasks
    print(f"Total tasks: {total_tasks}")   
    
    # Process the dataframe
    df['cleaned_answers'] = df['raw_answer'].apply(extract_digits_exp2)
    
    # Split the dataframe by task
    df_bar = df[df['task_name'] == 'bar'].copy()
    df_pie = df[df['task_name'] == 'pie'].copy()
   
    return df_bar, df_pie, deleted_rows

def calculate_metrics(df_bar, df_pie):
    """
    Calculate metrics for each dataset and model.
    Parameters:
    df_type1, df_type2, df_type3, df_type4, df_type5: DataFrames containing the results for each task
    Returns:
    pandas.DataFrame: Table of metrics for all models and datasets
    """
    # Dictionary to store metrics for each dataset
    metrics_summary = {}

    # List of DataFrames to process
    dataframes = {
        'bar': df_bar, 
        'pie': df_pie, 
    }

    # Loop through each dataset
    for df_name, df in dataframes.items():
        if df.empty:
            continue
            
        model_metrics = {}
        
        for model_name, group in df.groupby('model_name'):
            # Create an explicit copy and perform all operations on it
            data = group.copy()
            
           # Convert string lists to actual lists first
            data['ground_truth_num'] = data['ground_truth'].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)
            data['cleaned_answers_num'] = data['cleaned_answers'].apply(lambda x: pd.eval(x) if isinstance(x, str) else x)
        
            
            # Drop rows with missing values
            data = data.dropna(subset=['ground_truth_num', 'cleaned_answers_num'])
            
            if len(data) == 0:
                continue
                
            # Calculate MLAE for each row
            mlae_values = []
            for _, row in data.iterrows():
                mlae = np.log2(mean_absolute_error(
                    [row['ground_truth_num']], 
                    [row['cleaned_answers_num']]
                ) + 0.125)
                mlae_values.append(mlae)
            
            # Assign MLAE values
            data.loc[:, 'mlae'] = mlae_values
            
            avg_mlae = np.mean(mlae_values)
            std_mlae = np.std(mlae_values)
            
            try:
                bootstrap_result = bs.bootstrap(
                    np.array(mlae_values), 
                    stat_func=bs_stats.std
                )
                confidence_value = 1.96 * bootstrap_result.value
            except:
                confidence_value = np.nan
            
            model_metrics[model_name] = {
                'Dataset': df_name,
                'Model': model_name,
                'Average MLAE': round(avg_mlae, 2),
                'Std MLAE': round(std_mlae, 2),
                'Confidence Interval (95%)': round(confidence_value, 2)
            }
        
        if model_metrics:  # Only add if there are metrics
            metrics_summary[df_name] = model_metrics

    metrics_table = pd.DataFrame([
        metrics 
        for dataset_metrics in metrics_summary.values() 
        for metrics in dataset_metrics.values()
    ])
    
    if not metrics_table.empty:
        metrics_table = metrics_table.sort_values(['Dataset', 'Average MLAE'])
    
    return metrics_table

def plot_results(metrics_table):
    """
    Plot the results from the metrics table with human benchmark values
    """
    summary_stats_by_task = {df_name: metrics_table[metrics_table['Dataset'] == df_name] 
                            for df_name in metrics_table['Dataset'].unique()}

    # Define display names for tasks
    task_display_names = {
        'bar': 'BAR',
        'pie': 'PIE',

    }

    # Define colors for models and Human benchmarks
    model_colors = {
        'CustomLLaMA': '#8E44AD',   # Purple
        'Gemini1_5Flash': '#3498DB',    # Blue
        'GeminiProVision': '#E74C3C',   # Red
        'LLaMA': '#E67E22',             # Orange
        'gpt4o': '#27AE60',             # Green
        'Human': '#34495E'              # Dark Gray
    }

    # Define Human benchmark data
    human_data = {
        'bar': (2.05, 0.115),
        'pie': (1.035, 0.125),
    }

    # Get task images
    base_path = '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numberone/images'
    task_images = find_task_images(base_path)
    
    num_tasks = len(summary_stats_by_task)


    fig, axes = plt.subplots(num_tasks, 3, figsize=(14, 1.5 * num_tasks), 
                        gridspec_kw={'width_ratios': [1, 4, 1]}, sharex=False)
    fig.subplots_adjust(hspace=0.2, left=0.05, right=0.95, top=0.98, bottom=0.02)


    fig.patch.set_facecolor('white')

    # Handle both single and multiple subplot cases
    if num_tasks == 1:
        axes = np.array([axes])  # Convert to 2D array with one row

    for i, (task_name, task_data) in enumerate(summary_stats_by_task.items()):
        ax_img, ax_plot, ax_label = axes[i]

        if task_name in task_images:
            img_path = task_images[task_name]
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("L")
                img_inverted = ImageOps.invert(img)
                img_with_border = ImageOps.expand(img_inverted.convert("RGB"), border=1, fill="black")
                ax_img.imshow(img_with_border)
                ax_img.set_facecolor("white")
            else:
                ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=10, color="black")
                ax_img.set_facecolor("white")

        ax_img.axis('off')
        ax_img.set_title(task_display_names.get(task_name, task_name), loc="left", fontsize=12, color="black")

        sorted_model_names = sorted(task_data['Model'].unique())
        y_positions = np.arange(len(sorted_model_names))

        for j, model_name in enumerate(sorted_model_names):
            model_data = task_data[task_data['Model'] == model_name]
            mlae_value = model_data['Average MLAE'].values[0]
            confidence_interval = model_data['Confidence Interval (95%)'].values[0]

            ax_plot.errorbar(mlae_value, j, xerr=confidence_interval, fmt='o', 
                           color=model_colors.get(model_name, 'gray'), capsize=5, 
                           label=f"{model_name}" if i == 0 else None)

        # Plot human benchmark as horizontal error bar
        if task_name in human_data:
            human_value, human_std = human_data[task_name]
            human_interval = human_std * 1.96
            y_pos = len(sorted_model_names) + 0.5
            
            ax_plot.errorbar(human_value, y_pos, xerr=human_interval, 
                           fmt='s', color=model_colors['Human'], 
                           capsize=5, capthick=1.5,
                           markersize=7, label='Human' if i == 0 else None)

        ax_plot.axvline(-4, color="black", linewidth=1)
        ax_plot.axvline(-14, color="black", linewidth=1)
        ax_plot.grid(False)

        for offset in np.linspace(-0.05, 0.05, 10):
            ax_plot.axvline(0 + offset, color="gray", alpha=0.1, linewidth=0.5)

        ax_plot.spines['top'].set_visible(False)
        ax_plot.spines['right'].set_visible(False)
        ax_plot.spines['left'].set_visible(False)
        ax_plot.spines['bottom'].set_position(('outward', 10))

        ax_plot.set_yticks(y_positions)
        ax_plot.set_yticklabels([])
        ax_plot.set_xlim(-3, 3)
        ax_plot.invert_yaxis()

        ax_label.set_yticks(y_positions)
        ax_label.set_yticklabels(sorted_model_names, fontsize=10)
        ax_label.tick_params(left=False, right=False, labelleft=False, labelright=True)
        ax_plot.tick_params(axis='y', which='both', left=False, right=False)
        ax_label.set_ylim(ax_plot.get_ylim())
        ax_label.axis("off")

    if num_tasks > 0:  # Only add legend if there are tasks
        axes[0, 1].legend(loc="best", frameon=False)
    plt.show()

def process_plot(metrics_table):
    """
    Process and create the plot for given metrics table
    """
    print("\nGenerating plot...")
    plot_results(metrics_table)


def find_task_images(base_path, task_types=None):
    """
    Automatically find images for each task in the given directory
    Parameters:
    base_path (str): Base directory path where images are stored
    task_types (list): List of task types (e.g., ['unframed', 'framed']). If None, will look for both.
    Returns:
    dict: Dictionary mapping task names to image paths
    """
    task_images = {}
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    if task_types is None:
        task_types = ['bar', 'pie']

    for task in task_types:
        task_pattern = f"{task}_"
        
        for file in os.listdir(base_path):
            if file.startswith(task_pattern) and any(file.lower().endswith(ext) for ext in image_extensions):
                task_images[task] = os.path.join(base_path, file)
                break

    return task_images



""" Average 3 running """
def average_metrics(metrics_list):

    """Helper function to average metrics across multiple runs"""
   
    all_metrics_df = pd.concat(metrics_list)
    
    averaged_metrics = all_metrics_df.groupby(['Dataset', 'Model']).agg({
        'Average MLAE': 'mean',
        'Std MLAE': 'mean',
        'Confidence Interval (95%)': 'mean'
    }).reset_index()
    
    for col in ['Average MLAE', 'Std MLAE', 'Confidence Interval (95%)']:
        averaged_metrics[col] = averaged_metrics[col].round(2)
    
    return averaged_metrics

def process_and_plot_multiplerun():
    """Process three EXP2 result files and create averaged plot"""
    # Define file paths
    file_paths = [
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numberone/EXP-Results/EXP2results10images.csv",
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numbertwo/EXP-Results/EXP2results10images.csv",
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numberthree/EXP-Results/EXP2results10images.csv"
    ]

    # Calculate metrics for each file and store them
    all_metrics = []
    for i, file_path in enumerate(file_paths, 1):
        
        # Process each file
        df_bar, df_pie, deleted_rows = clean_raw_answers(file_path)
        
        # Print number of rows for each task
        print(f"\nNumber of rows in each task:")
        print(f"Task bar rows: {len(df_bar)}")
        print(f"Task pie rows: {len(df_pie)}")
        
        # Calculate metrics
        metrics = calculate_metrics(df_bar, df_pie)
        print("\nMetrics for this file:")
        print(metrics)
        
        all_metrics.append(metrics)

    # Calculate and print averaged metrics
    print("\nAveraged Metrics across all files:")
    averaged_metrics_table = average_metrics(all_metrics)
    averaged_metrics_table

    # Plot using the averaged metrics
    plot_results(averaged_metrics_table)

    return averaged_metrics_table

def checkdeletedrows_forallcsv():
    """Process and check deleted rows across all CSV files"""
    file_paths = [
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numberone/EXP-Results/EXP2results10images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numbertwo/EXP-Results/EXP2results10images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP2/finetuning-EXP2numberthree/EXP-Results/EXP2results10images.csv'
    ]
    
    all_deleted_dfs = []
    
    for file_path in file_paths:
        df_bar, df_pie, deleted_rows = clean_raw_answers(file_path)
        
        print(f"\nNumber of rows in each task:")
        print(f"Task bar rows: {len(df_bar)}")
        print(f"Task pie rows: {len(df_pie)}")

        metrics_table = calculate_metrics(df_bar, df_pie)
        metrics_table

        # Create DataFrame from deleted_rows while preserving the structure
        if deleted_rows:  # Only process if there are deleted rows
            deleted_df = pd.DataFrame(deleted_rows)[['raw_answer', 'model_name']]  # Select columns first
            deleted_df['file'] = file_path.split('/')[-1]  # Add filename
            all_deleted_dfs.append(deleted_df)
    
     # Combine all deleted rows
    combined_deleted_df = pd.concat(all_deleted_dfs, ignore_index=True)
    return combined_deleted_df[['file', 'raw_answer', 'model_name']]