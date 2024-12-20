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

def clean_raw_answers(file_path):
    """
    Clean raw answers from CSV file, focusing only on extracting digits.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    tuple: (df_framed, df_unframed, deleted_rows) - Two DataFrames and list of deleted rows
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    deleted_rows = []
    total_tasks = len(df)
        
    def extract_digits_exp5(x):
        if pd.isna(x):
            return np.nan
        # Convert to string
        x = str(x)
        # Remove newline characters and whitespace
        x = x.strip().replace('\n', '')
        # If string starts with "user", extract the last number
        if x.startswith('user'):
            numbers = re.findall(r'\d+\.?\d*', x)
            return float(numbers[-1]) if numbers else np.nan
        # Extract first number found
        numbers = re.findall(r'\d+\.?\d*', x)
        return float(numbers[0]) if numbers else np.nan

    # Verify column names in DataFrame
    answer_column = 'answer' if 'answer' in df.columns else 'raw_answer'

    
    # Create new column with cleaned values
    df['parsed_answers'] = df[answer_column].apply(extract_digits_exp5)
    
    # Drop rows with NaN in parsed_answers
    df = df.dropna(subset=['parsed_answers'])
    
    # Format cleaned values as strings with one decimal point
    df['parsed_answers'] = df['parsed_answers'].apply(
        lambda x: '{:.1f}'.format(x) if not pd.isna(x) else x
    )
    
    # Process rows for deletion
    rows_to_delete = []
    for index, row in df.iterrows():
        raw_text = row[answer_column]
        
        # Extract digits
        cleaned_answer = extract_digits_exp5(raw_text)
        
        # Check if answer is valid, otherwise mark for deletion
        if cleaned_answer is None:
            deleted_rows.append(row)
            rows_to_delete.append(index)
    
    # Drop rows marked for deletion
    df.drop(rows_to_delete, inplace=True)
    
    # Print deletion summary
    print(f"Total tasks: {total_tasks}")
    print(f"Rows deleted: {len(deleted_rows)}")
    print("Deleted rows:")
    
    # Process the dataframe with extracted digits (no need for model_name)
    df['cleaned_answers'] = df[answer_column].apply(extract_digits_exp5)
    
    # Split the dataframe by task
    df_10 = df[df['task_name'] == 10].copy()
    df_100 = df[df['task_name'] == 100].copy()
    df_1000 = df[df['task_name'] == 1000].copy()
    
    return df_10, df_100, df_1000

def calculate_metrics(df_10, df_100, df_1000):
    """
    Calculate metrics for each dataset and model.
    Parameters:
    df_10, df_100, df_1000: DataFrames containing the results for each task
    Returns:
    pandas.DataFrame: Table of metrics for all models and datasets
    """
    # Dictionary to store metrics for each dataset
    metrics_summary = {}

    # List of DataFrames to process
    dataframes = {
        'Task_10': df_10, 
        'Task_100': df_100, 
        'Task_1000': df_1000
    }

    # Loop through each dataset
    for df_name, df in dataframes.items():
        model_metrics = {}
        
        for model_name, data in df.groupby('model_name'):
            data['ground_truth'] = pd.to_numeric(data['ground_truth'], errors='coerce')
            data['parsed_answers'] = pd.to_numeric(data['parsed_answers'], errors='coerce')
            data = data.dropna(subset=['ground_truth', 'parsed_answers'])
            
            if len(data) == 0:
                continue

            data['mlae'] = data.apply(
                lambda row: np.log2(mean_absolute_error(
                    [row['ground_truth']], 
                    [row['parsed_answers']]
                ) + 0.125),
                axis=1
            )
            
            avg_mlae = data['mlae'].mean()
            std_mlae = data['mlae'].std()
            
            try:
                mlae_values = data['mlae'].dropna().values
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
        'Task_10': '10 dots',
        'Task_100': '100 dots',
        'Task_1000': '1000 dots'
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
        'Task_10': (4.0149, 0.5338),
        'Task_100': (5.3891, 0.1945),
        'Task_1000': (5.4612, 0.2509)
    }

    # Define task images
    # Auto-detect task images
    def find_task_images(base_path):
        task_images = {}
        image_extensions = ['.jpg', '.jpeg', '.png']
        
        try:
            for task in [10, 100, 1000]:
                task_pattern = f"{task}_"
                image_dir = os.path.join(base_path, 'images')
                if os.path.exists(image_dir):
                    for file in os.listdir(image_dir):
                        if file.startswith(task_pattern) and any(file.lower().endswith(ext) for ext in image_extensions):
                            task_images[f'Task_{task}'] = os.path.join(image_dir, file)
                            break
        except Exception as e:
            print(f"Warning: Error finding images: {str(e)}")
        
        return task_images

    # Get task images
    base_path = '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberone'
    task_images = find_task_images(base_path)
    
    num_tasks = len(summary_stats_by_task)
    fig, axes = plt.subplots(num_tasks, 3, figsize=(12, 3 * num_tasks), 
                            gridspec_kw={'width_ratios': [1, 4, 1]}, sharex=False)
    fig.subplots_adjust(hspace=0.8, left=0.05, right=0.95)
    fig.patch.set_facecolor('white')

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

        # Add reference lines
        ax_plot.axvline(-5, color="black", linewidth=1)
        ax_plot.axvline(12, color="black", linewidth=1)
        ax_plot.grid(False)

        # Add blurred line at 0
        for offset in np.linspace(-0.05, 0.05, 10):
            ax_plot.axvline(0 + offset, color="gray", alpha=0.1, linewidth=0.5)

        # Customize plot appearance
        ax_plot.spines['top'].set_visible(False)
        ax_plot.spines['right'].set_visible(False)
        ax_plot.spines['left'].set_visible(False)
        ax_plot.spines['bottom'].set_position(('outward', 10))

        ax_plot.set_yticks(y_positions)
        ax_plot.set_yticklabels([])
        ax_plot.set_xlim(-5, 12)
        ax_plot.invert_yaxis()

        # Display model names
        ax_label.set_yticks(y_positions)
        ax_label.set_yticklabels(sorted_model_names, fontsize=10)
        ax_label.tick_params(left=False, right=False, labelleft=False, labelright=True)
        ax_plot.tick_params(axis='y', which='both', left=False, right=False)
        ax_label.set_ylim(ax_plot.get_ylim())
        ax_label.axis("off")

    axes[0, 1].legend(loc="best", frameon=False)
    plt.show()

def process_plot(metrics_table):
    """
    Process and create the plot for given metrics table
    """
    print("\nGenerating plot...")
    plot_results(metrics_table)
   
def find_task_images(base_path):

    """
    Automatically find images for each task in the given directory
    Parameters:
    base_path (str): Base directory path where images are stored
    Returns:
    dict: Dictionary mapping task names to image paths
    """
    task_images = {}
    image_extensions = ['.jpg', '.jpeg', '.png']

    try:
        # For each task (10, 100, 1000)
        for task in [10, 100, 1000]:
            # Look for images that start with the task number
            task_pattern = f"{task}_"
            
            # Search in the images directory
            image_dir = os.path.join(base_path, 'images')
            if os.path.exists(image_dir):
                for file in os.listdir(image_dir):
                    if file.startswith(task_pattern) and any(file.lower().endswith(ext) for ext in image_extensions):
                        task_images[f'Task_{task}'] = os.path.join(image_dir, file)
                        break  # Take the first matching image for each task
    except Exception as e:
        print(f"Warning: Error finding images: {str(e)}")

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
    """Process three EXP5 result files and create averaged plot"""
    # Define file paths
    file_paths = [
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberone/EXP-Results/EXP5results10images.csv",
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numbertwo/EXP-Results/EXP5results10images.csv",
        "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberthree/EXP-Results/EXP5results10images.csv"
    ]

    # Calculate metrics for each file and store them
    all_metrics = []
    for i, file_path in enumerate(file_paths, 1):
        print(f"\nProcessing File {i}: {os.path.basename(file_path)}")
        
        # Process each file
        df_10, df_100, df_1000 = clean_raw_answers(file_path)
        
        # Print number of rows for each task
        print(f"\nNumber of rows in each task:")
        print(f"Task 10 rows: {len(df_10)}")
        print(f"Task 100 rows: {len(df_100)}")
        print(f"Task 1000 rows: {len(df_1000)}")
        
        # Calculate metrics
        metrics = calculate_metrics(df_10, df_100, df_1000)
        print("\nMetrics for this file:")
        print(metrics)
        
        all_metrics.append(metrics)

    # Calculate and print averaged metrics
    print("\nAveraged Metrics across all files:")
    averaged_metrics_table = average_metrics(all_metrics)
    print(averaged_metrics_table)

    # Plot using the averaged metrics
    plot_results(averaged_metrics_table)

    return averaged_metrics_table



def checkdeletedrows_forallcsv():
    file_paths = [
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberone/EXP-Results/EXP5results10images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numbertwo/EXP-Results/EXP5results10images.csv',
        '/hpcstor6/scratch01/h/huuthanhvy.nguyen001/EXP5/finetuning-EXP5numberthree/EXP-Results/EXP5results10images.csv'
    ]
    
    all_deleted_dfs = []
    
    for file_path in file_paths:
        df_10, df_100, df_1000, deleted_rows = clean_raw_answers(file_path)
        
        print(f"\nNumber of rows in each task:")
        print(f"Task 10 rows: {len(df_10)}")
        print(f"Task 100 rows: {len(df_100)}")
        print(f"Task 1000 rows: {len(df_1000)}")

        metrics_table = calculate_metrics(df_10, df_100, df_1000)
        metrics_table

        # Create DataFrame from deleted_rows while preserving the structure
        if deleted_rows:  # Only process if there are deleted rows
            deleted_df = pd.DataFrame(deleted_rows)[['raw_answer', 'model_name']]  # Select columns first
            deleted_df['file'] = file_path.split('/')[-1]  # Add filename
            all_deleted_dfs.append(deleted_df)
    
    # Combine all deleted rows if there are any
    if all_deleted_dfs:
        combined_deleted_df = pd.concat(all_deleted_dfs, ignore_index=True)
        return combined_deleted_df[['file', 'raw_answer', 'model_name']]
    else:
        # Return empty DataFrame with correct columns if no deleted rows
        return pd.DataFrame(columns=['file', 'raw_answer', 'model_name'])