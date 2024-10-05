import streamlit as st
import pandas as pd
from autoviz.AutoViz_Class import AutoViz
import plotly.express as px

# Set the page configuration
st.set_page_config(page_title="Auto Data Plotting App", layout="wide")

# Title of the app
st.title("Auto Data Plotting with AutoViz and Streamlit")

# File upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        st.success("File uploaded successfully!")
        
        # Display the dataframe
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Automatic Visualization using AutoViz
        st.subheader("Automatic Visualization")
        AV = AutoViz()
        # AutoViz will display the plots automatically in the console, but in Streamlit, we need to capture the plots.
        # However, AutoViz is primarily designed for Jupyter Notebooks and may not directly integrate with Streamlit.
        # As an alternative, you can use Streamlit's native features or other visualization libraries.
        # For demonstration, we'll skip AutoViz automatic plots and focus on custom visualization.

        # Custom Visualization
        st.subheader("Custom Visualization")
        all_columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Sidebar for user inputs
        st.sidebar.header("Plotting Options")
        plot_type = st.sidebar.selectbox("Select Plot Type", ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Heatmap"])
        
        if plot_type in ["Scatter Plot", "Line Chart", "Bar Chart"]:
            x_axis = st.sidebar.selectbox("Select X-axis", all_columns)
            y_axis = st.sidebar.selectbox("Select Y-axis", all_columns)
            color = st.sidebar.selectbox("Select Color (optional)", [None] + all_columns)
            
            if st.sidebar.button("Generate Plot"):
                if plot_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=color)
                elif plot_type == "Line Chart":
                    fig = px.line(df, x=x_axis, y=y_axis, color=color)
                elif plot_type == "Bar Chart":
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Histogram":
            hist_col = st.sidebar.selectbox("Select Column for Histogram", numeric_columns)
            bins = st.sidebar.slider("Number of Bins", min_value=5, max_value=100, value=20)
            if st.sidebar.button("Generate Histogram"):
                fig = px.histogram(df, x=hist_col, nbins=bins)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box Plot":
            box_col = st.sidebar.selectbox("Select Column for Box Plot", numeric_columns)
            box_group = st.sidebar.selectbox("Group By (optional)", [None] + categorical_columns)
            if st.sidebar.button("Generate Box Plot"):
                fig = px.box(df, x=box_group, y=box_col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Heatmap":
            corr_method = st.sidebar.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
            if st.sidebar.button("Generate Heatmap"):
                corr = df.corr(method=corr_method)
                fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV or Excel file to get started.")
