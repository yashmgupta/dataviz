import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# Set the page configuration
st.set_page_config(page_title="Professional Data Analysis App", layout="wide")

# Title of the app
st.title("📊 Professional Data Analysis and Visualization App")

# File upload
uploaded_file = st.file_uploader("📁 Upload a CSV or Excel file", type=["csv", "xlsx"])

# Load data with caching for efficiency
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file, engine='openpyxl')

# Auto-analysis function
@st.cache_data
def perform_auto_analysis(df):
    analysis = {}
    analysis['shape'] = df.shape
    analysis['columns'] = df.columns.tolist()
    analysis['missing_values'] = df.isnull().sum().to_dict()
    analysis['data_types'] = df.dtypes.to_dict()
    analysis['summary_statistics'] = df.describe(include='all').to_dict()
    analysis['correlation'] = df.select_dtypes(include=[np.number]).corr().to_dict()
    return analysis

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = load_data(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Display data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        # Perform automatic analysis
        analysis = perform_auto_analysis(df)

        # Sidebar for analysis options
        st.sidebar.header("⚙️ Analysis Options")
        show_analysis = st.sidebar.checkbox("Show Automatic Analysis Details")

        if show_analysis:
            # Display auto-analysis results if selected
            st.subheader("📊 Automatic Data Analysis")
            st.write(f"**Dataset Shape:** {analysis['shape']}")
            st.write(f"**Columns:** {', '.join(analysis['columns'])}")
            st.write("**Missing Values:")
            st.write(analysis['missing_values'])
            st.write("**Data Types:**")
            st.write(analysis['data_types'])
            st.write("**Summary Statistics:**")
            st.write(pd.DataFrame(analysis['summary_statistics']))

        # Sidebar for selecting advanced analysis options
        st.sidebar.header("🔍 Advanced Analysis Options")
        if st.sidebar.checkbox("Perform PCA (Principal Component Analysis)"):
            st.subheader("🔍 Principal Component Analysis (PCA)")
            n_components = st.sidebar.slider("Number of Components", 1, min(len(df.columns), 10), 2)
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(scaled_data)
            pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
            st.write(pca_df.head())
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x='PC1', y='PC2', data=pca_df)
            plt.title("PCA Result - First Two Principal Components")
            st.pyplot(plt)

        if st.sidebar.checkbox("Perform Clustering (K-Means)"):
            st.subheader("🔍 K-Means Clustering")
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(scaled_data)
            df['Cluster'] = kmeans.labels_
            st.write(df[['Cluster']].value_counts().reset_index(name='Count'))
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=numeric_df.columns[0], y=numeric_df.columns[1], hue=df['Cluster'], palette='viridis', data=df)
            plt.title("K-Means Clustering Result")
            st.pyplot(plt)

        if st.sidebar.checkbox("Outlier Detection"):
            st.subheader("🔍 Outlier Detection")
            selected_column = st.sidebar.selectbox("Select Column for Outlier Detection", df.select_dtypes(include=[np.number]).columns)
            threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0)
            z_scores = np.abs(stats.zscore(df[selected_column].dropna()))
            outliers = df[z_scores > threshold]
            st.write(f"Number of Outliers in {selected_column}: {len(outliers)}")
            st.write(outliers)

        if st.sidebar.checkbox("Perform Regression Analysis"):
            st.subheader("🔍 Regression Analysis")
            target = st.sidebar.selectbox("Select Target Variable", options=all_columns)
            features = st.sidebar.multiselect("Select Feature Variables", options=[col for col in all_columns if col != target])
            if len(features) > 0:
                X = df[features].select_dtypes(include=[np.number]).dropna()
                y = df[target].dropna()
                model = LinearRegression()
                model.fit(X, y)
                st.write(f"Intercept: {model.intercept_}")
                st.write(f"Coefficients: {dict(zip(features, model.coef_))}")
                st.write(f"R^2 Score: {model.score(X, y)}")

        # Sidebar for selecting columns and plot type
        st.sidebar.header("📈 Plotting Options")

        # Select X-axis
        all_columns = df.columns.tolist()
        x_axis = st.sidebar.selectbox("Select X-axis", options=all_columns)

        # Select Y-axis (multiple selections for some plot types)
        y_axis = st.sidebar.multiselect("Select Y-axis (one or more columns)", options=all_columns)

        # Select plot type
        plot_type = st.sidebar.selectbox("Select Plot Type",
                                         options=["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Heatmap", "Pair Plot", "Violin Plot", "Area Chart", "Count Plot"])

        # Optional: Select color (for applicable plots)
        color_option = st.sidebar.selectbox("Select Color (optional)", options=[None] + all_columns) if plot_type in ["Scatter Plot", "Line Chart", "Bar Chart", "Violin Plot", "Area Chart"] else None

        # Optional: Select grouping (for bar plot)
        group_option = st.sidebar.selectbox("Select Grouping (optional)", options=[None] + all_columns) if plot_type == "Bar Chart" else None

        # Generate Plot button
        if st.sidebar.button("Generate Plot"):
            st.subheader(f"📊 {plot_type}")
            
            # Plotting with Seaborn for enhanced visuals
            plt.figure(figsize=(12, 6))

            try:
                if plot_type == "Scatter Plot":
                    sns.scatterplot(data=df, x=x_axis, y=y_axis[0], hue=color_option, palette='viridis', alpha=0.7)
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis[0])
                    plt.title(f"{y_axis[0]} vs {x_axis}")

                elif plot_type == "Line Chart":
                    sns.lineplot(data=df, x=x_axis, y=y_axis[0], hue=color_option)
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis[0])
                    plt.title(f"{y_axis[0]} over {x_axis}")

                elif plot_type == "Bar Chart":
                    sns.barplot(data=df, x=x_axis, y=y_axis[0], hue=group_option, alpha=0.7)
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis[0])
                    plt.title(f"{y_axis[0]} by {x_axis}")
                    plt.xticks(rotation=45, ha='right')

                elif plot_type == "Histogram":
                    sns.histplot(df[x_axis], bins=20, color='skyblue', edgecolor='black')
                    plt.xlabel(x_axis)
                    plt.ylabel("Frequency")
                    plt.title(f"Histogram of {x_axis}")

                elif plot_type == "Box Plot":
                    sns.boxplot(data=df[y_axis])
                    plt.ylabel("Values")
                    plt.title(f"Box Plot of {', '.join(y_axis)}")

                elif plot_type == "Heatmap":
                    correlation = df.select_dtypes(include=[np.number]).corr()
                    sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
                    plt.title("Correlation Heatmap")

                elif plot_type == "Pair Plot":
                    sns.pairplot(df[y_axis])
                    st.pyplot()  # Special handling since pairplot creates multiple subplots
                    plt.clf()
                
                elif plot_type == "Violin Plot":
                    sns.violinplot(data=df, x=x_axis, y=y_axis[0], hue=color_option, split=True, palette='muted')
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis[0])
                    plt.title(f"Violin Plot of {y_axis[0]} by {x_axis}")

                elif plot_type == "Area Chart":
                    plt.fill_between(df[x_axis], df[y_axis[0]], color='skyblue', alpha=0.4)
                    plt.plot(df[x_axis], df[y_axis[0]], color='Slateblue', alpha=0.6)
                    plt.xlabel(x_axis)
                    plt.ylabel(y_axis[0])
                    plt.title(f"Area Chart of {y_axis[0]} over {x_axis}")

                elif plot_type == "Count Plot":
                    sns.countplot(data=df, x=x_axis, hue=color_option, palette='viridis')
                    plt.xlabel(x_axis)
                    plt.ylabel("Count")
                    plt.title(f"Count Plot of {x_axis}")

                st.pyplot(plt)
                plt.clf()  # Clear the figure after plotting to avoid overlap

            except ValueError as ve:
                st.error(f"❌ Plotting error: {ve}. Please check if the selected columns are suitable for the chosen plot type.")

        # Additional Features: Display statistics
        st.subheader("📈 Data Statistics")
        st.write(df.describe(include='all'))

        # Optionally, allow users to download the data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Data as CSV",
            data=csv,
            file_name='data.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("🗒️ Please upload a CSV or Excel file to get started.")
