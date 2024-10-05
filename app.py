import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

# Set the page configuration
st.set_page_config(page_title="Data Visualization App", layout="wide")

# Title of the app
st.title("📊 Data Visualization App with Pandas and Matplotlib")

# File upload
uploaded_file = st.file_uploader("📁 Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        st.success("✅ File uploaded successfully!")

        # Display data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        # Sidebar for selecting columns and plot type
        st.sidebar.header("📈 Plotting Options")

        # Select X-axis
        all_columns = df.columns.tolist()
        x_axis = st.sidebar.selectbox("Select X-axis", options=all_columns)

        # Select Y-axis
        y_axis = st.sidebar.selectbox("Select Y-axis", options=all_columns)

        # Select plot type
        plot_type = st.sidebar.selectbox("Select Plot Type", 
                                         options=["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Heatmap"])

        # Optional: Select color (for applicable plots)
        if plot_type in ["Scatter Plot", "Line Chart", "Bar Chart"]:
            color_option = st.sidebar.selectbox("Select Color (optional)", 
                                               options=[None] + all_columns)
        else:
            color_option = None

        # Generate Plot button
        if st.sidebar.button("Generate Plot"):
            st.subheader(f"📊 {plot_type}")

            plt.figure(figsize=(10, 6))
            
            if plot_type == "Scatter Plot":
                if color_option:
                    plt.scatter(df[x_axis], df[y_axis], c=df[color_option], cmap='viridis', alpha=0.7)
                    plt.colorbar(label=color_option)
                else:
                    plt.scatter(df[x_axis], df[y_axis], alpha=0.7)
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
                plt.title(f"{y_axis} vs {x_axis}")
            
            elif plot_type == "Line Chart":
                if color_option:
                    plt.plot(df[x_axis], df[y_axis], label=y_axis, color=df[color_option].astype('category').cat.codes)
                else:
                    plt.plot(df[x_axis], df[y_axis], label=y_axis)
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
                plt.title(f"{y_axis} over {x_axis}")
                plt.legend()
            
            elif plot_type == "Bar Chart":
                if color_option:
                    plt.bar(df[x_axis], df[y_axis], color=df[color_option].astype('category').cat.codes, alpha=0.7)
                else:
                    plt.bar(df[x_axis], df[y_axis], alpha=0.7)
                plt.xlabel(x_axis)
                plt.ylabel(y_axis)
                plt.title(f"{y_axis} by {x_axis}")
                plt.xticks(rotation=45, ha='right')
            
            elif plot_type == "Histogram":
                plt.hist(df[x_axis], bins=20, color='skyblue', edgecolor='black')
                plt.xlabel(x_axis)
                plt.ylabel("Frequency")
                plt.title(f"Histogram of {x_axis}")
            
            elif plot_type == "Box Plot":
                plt.boxplot(df[y_axis].dropna())
                plt.ylabel(y_axis)
                plt.title(f"Box Plot of {y_axis}")
            
            elif plot_type == "Heatmap":
                correlation = df.corr()
                plt.imshow(correlation, cmap='coolwarm', interpolation='none', aspect='auto')
                plt.colorbar()
                plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=90)
                plt.yticks(range(len(correlation.columns)), correlation.columns)
                plt.title("Correlation Heatmap")
            
            st.pyplot(plt)
            plt.clf()  # Clear the figure after plotting to avoid overlap
            
        # Additional Features: Display statistics
        st.subheader("📈 Data Statistics")
        st.write(df.describe())

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
    st.info("📝 Please upload a CSV or Excel file to get started.")
